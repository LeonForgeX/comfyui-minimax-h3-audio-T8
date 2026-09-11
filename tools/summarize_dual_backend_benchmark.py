"""Compare only matched, independently audited cold/repeat backend jobs."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

try:
    from .progressive_probe_control import file_identity, summarize_execution_events
except ImportError:
    from progressive_probe_control import file_identity, summarize_execution_events


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalized_graph(graph, backend):
    value = deepcopy(graph)
    expected = {
        'pytorch': ('ModelAttentionBackend', {'attention': 'pytorch attention'}),
        'kj': ('PathchSageAttentionKJ', {'sage_attention': 'auto', 'allow_compile': False}),
        'sol': ('SolAttentionPatch', {'enabled': True, 'tau': 1.3, 'min_tokens': 4096,
            'strict': True, 'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}),
    }
    cls, options = expected[backend]
    for node_id, model in [('21', '2'), ('22', '3')]:
        if value[node_id] != {'class_type': cls, 'inputs': {'model': [model, 0], **options}}:
            raise ValueError('Backend recipe differs from the measured preset')
        value[node_id] = {'class_type': 'BENCHMARK_BACKEND', 'inputs': {'model': [model, 0]}}
    if value['8']['inputs']['total_duration_seconds'] != 3:
        raise ValueError('Benchmark duration differs')
    value['8']['inputs']['chain_id'] = 'BENCHMARK_NAMESPACE'
    return value


def memory_recipe(root, terminal, expected):
    """Compare explicit observed reservations and the actual child command, not defaults."""
    options = {key: terminal.get(key) for key in ('reserve_vram_gib', 'headroom_gib')}
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in options.values()):
        raise ValueError('Current benchmark needs explicit finite runtime memory settings')
    live = terminal.get('environment', {})
    if (live.get('status') != 'pass' or live.get('device') == 'cpu'
            or not live.get('device') or live.get('pid') != terminal['server_stop'].get('pid')
            or live.get('runtime_options') != options or expected.get('runtime_options') != options):
        raise ValueError('Expected and actual runtime memory settings differ')
    unpinned = terminal.get('disable_pinned_memory')
    command = read(root / 'server-command.json')
    if (type(unpinned) is not bool or not isinstance(command, list)
            or any(not isinstance(value, str) for value in command)
            or command.count('--disable-pinned-memory') != int(unpinned)
            or command.count('--cache-none') != 1):
        raise ValueError('Actual pinning/cache command differs from measured recipe')
    for flag, key in (('--reserve-vram', 'reserve_vram_gib'), ('--vram-headroom', 'headroom_gib')):
        count = command.count(flag)
        if not count and key == 'headroom_gib' and options[key] == 0:
            continue  # Native default0 is also verified by the live environment.
        if count != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError('Missing or duplicate runtime memory argument')
        if float(command[command.index(flag) + 1]) != options[key]:
            raise ValueError('Runtime memory command differs from observed settings')
    return {**options, 'disable_pinned_memory': unpinned}


def validate_current_audio(audit):
    contract = audit.get('audio_contract', {})
    if not isinstance(contract, dict):
        raise ValueError('Current audio contract missing')
    delta = contract.get('max_absolute_delta')
    if (contract.get('mode') != 'joint_second_pass_continuation'
            or contract.get('lock_tolerance') is not None
            or audit.get('audio_input_bit_exact') is not True
            or audit.get('audio_output_bit_exact') is not False
            or type(delta) not in (int, float) or not math.isfinite(delta) or delta <= 1e-6
            or delta != audit.get('audio_output_max_absolute_delta')):
        raise ValueError('Current4+4 requires audited joint audio continuation; historical locked audio is not comparable')


def validate_timing(root, folder, graph, timing):
    path = root / folder
    events = [json.loads(line) for line in (path / 'events.jsonl').read_text(encoding='utf8').splitlines()]
    prompt_id = read(path / 'submission.json')['prompt_id']
    if (type(timing['elapsed_seconds']) not in (int, float)
            or not read(path / 'history.json')['status']['completed']
            or summarize_execution_events(events, prompt_id, graph, timing['elapsed_seconds']) != timing):
        raise ValueError('Execution events/history do not support the claimed timing')


def invocation_output(root, folder):
    """Use native history, including safe hashed chain aliases; never guess a folder."""
    images = read(root / folder / 'history.json').get('outputs', {}).get('8', {}).get('images', [])
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError('Exactly one measured node output is required')
    entry = images[0]
    relative = Path(entry.get('subfolder', ''))
    filename = entry.get('filename', '')
    if (entry.get('type') != 'output' or relative.is_absolute() or '..' in relative.parts
            or len(relative.parts) < 2 or relative.parts[0] != 'minimax_h3_t8_long_video'
            or not filename or Path(filename).name != filename):
        raise ValueError('Measured history output leaves its owned chain')
    chain_root = (root / 'output' / relative.parts[0] / relative.parts[1]).resolve(strict=True)
    media = (root / 'output' / relative / filename).resolve(strict=True)
    if not chain_root.is_relative_to((root / 'output').resolve(strict=True)) or not media.is_relative_to(chain_root):
        raise ValueError('Measured output path escaped its chain')
    return chain_root, media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs=3, required=True, help='PyTorch, KJ, Sol order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite measurements')
    reference_graph = reference_sources = reference_core = None
    reference_memory = reference_device = None
    rows = []
    measured_outputs = set()
    for backend, root in zip(('pytorch', 'kj', 'sol'), args.roots):
        if (root / 'benchmark-exclusion.json').exists():
            raise ValueError('Benchmark pair explicitly excluded; preserve evidence and do not compare its timings')
        terminal, pair, expected = [read(root / name) for name in ('terminal.json', 'benchmark-pair.json', 'expected.json')]
        if (pair.get('status') != 'two_uncached_jobs_completed_independent_audit_pending'
                or pair['process_warm']['chain_id'] != pair['cold']['chain_id'] + '_process_warm'):
            raise ValueError('Repeat job must use its independent uncached namespace')
        if (terminal['backend'] != backend or not terminal['status'].startswith('generation_completed')
                or terminal['server_stop']['owned_children_remaining']
                or terminal['resources']['status'] != 'observations_within_policy'
                or terminal.get('mode') != 'gpu' or terminal.get('second_base_file')):
            raise ValueError('Backend pair is incomplete or has different residency settings')
        memory = memory_recipe(root, terminal, expected)
        device = {key: terminal['environment'].get(key) for key in ('device', 'torch_version', 'torch_cuda')}
        device['gpu_uuid'] = terminal['resources'].get('gpu_uuid')
        if not all(device.values()):
            raise ValueError('Actual device/runtime identity missing')
        if reference_memory is None:
            reference_memory, reference_device = memory, device
        elif memory != reference_memory or device != reference_device:
            raise ValueError('Backend pairs use different memory/device/runtime recipes')
        sources, core = expected['sources'], expected['core']
        if reference_sources is None:
            reference_sources, reference_core = sources, core
        elif sources != reference_sources or core != reference_core:
            raise ValueError('Measured runtime/Core source snapshots differ')
        for label, folder in [('cold', 'generation'), ('process_warm', 'generation-process-warm')]:
            graph = read(root / folder / 'prompt.json')
            if label == 'cold' and graph != expected['pilot_graphs']['dual_short']:
                raise ValueError('Measured graph differs from requested graph')
            normalized = normalized_graph(graph, backend)
            if reference_graph is None:
                reference_graph = normalized
            elif normalized != reference_graph:
                raise ValueError('Models, inputs, seed or non-backend parameters differ')
            timing = read(root / folder / 'timing.json')
            if timing != pair[label]['timing'] or not timing['complete_uncached_graph']:
                raise ValueError('Timing receipt mismatch or cached graph')
            validate_timing(root, folder, graph, timing)
            chain = graph['8']['inputs']['chain_id']
            if pair[label]['chain_id'] != chain:
                raise ValueError('Timing bound to the wrong chain')
            audit = read(root / f'audit-{label}.json')
            chain_root, delivered_media = invocation_output(root, folder)
            if delivered_media in measured_outputs:
                raise ValueError('Two measured invocations reused the same output')
            measured_outputs.add(delivered_media)
            if (audit['selected_chain'] != chain_root.name or audit['backend'] != backend
                    or audit['actual_forwards'] != [4, 4]
                    or audit['status'] != 'short_mechanical_audit_pass_human_pending'):
                raise ValueError('Independent stage/media qualification missing')
            validate_current_audio(audit)
            media = Path(audit['media']['path']).resolve(strict=True)
            if (media != delivered_media or not media.is_relative_to(chain_root.resolve(strict=True))
                    or file_identity(media)['sha256'] != audit['media']['sha256']):
                raise ValueError('Audited media changed or is outside its measured chain')
            stages = {}
            for stage in ('low_x0', 'high_output'):
                paths = list(chain_root.rglob(stage + '-*.json'))
                if len(paths) != 1:
                    raise ValueError('Ambiguous measured stage')
                record = read(paths[0])
                if record['tensor_sha256'] != audit['stage_sha256'][stage]:
                    raise ValueError('Timing stage differs from audited media source')
                stages[stage] = record['report']
            measured = [timing['elapsed_seconds'], stages['low_x0']['sample_seconds'],
                stages['high_output']['preparation']['upscale_seconds'], stages['high_output']['sample_seconds']]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in measured):
                raise ValueError('Invalid measured stage duration')
            rows.append({'backend': backend, 'invocation': label,
                'total_graph_seconds': timing['elapsed_seconds'],
                'first_sampling_including_load_seconds': stages['low_x0']['sample_seconds'],
                'learned_upscale_seconds': stages['high_output']['preparation']['upscale_seconds'],
                'second_sampling_including_load_seconds': stages['high_output']['sample_seconds'],
                'observed_device_peak_bytes_pair_window': terminal['resources']['maximum_observed_device_used_bytes'],
                'observed_minimum_free_system_ram_pair_window': terminal['resources']['minimum_ram_available_bytes'],
                'actual_kernel_calls': audit['backend_completed_calls'], 'media': audit['media'],
                'audit_sha256': hashlib.sha256((root / f'audit-{label}.json').read_bytes()).hexdigest()})
    result = {'status': 'matched_six_job_timing_and_kernel_evidence', 'rows': rows,
        'audio_policy': 'joint_second_pass_continuation', 'runtime_options': reference_memory,
        'device': reference_device,
        'limitations': ['One cold and one repeat observation per backend, not statistical medians.',
            'Cold means first graph in a new process; disk cache was not purged.',
            'Warm means same process; normal stage offload and --cache-none are retained.',
            'Sampling timers include lazy model initialization/movement, not pure kernel timings.',
            'Detailed conditioning/VAE/mux timings, when present in sidecars, are not aggregated in this table.',
            'Historical pre-repair locked-audio results are excluded, not requalified by this summary.',
            'Memory is periodically sampled whole-device/system data over both runs, not exact per-stage peaks.',
            'Human visual/audio acceptance remains separate.']}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
