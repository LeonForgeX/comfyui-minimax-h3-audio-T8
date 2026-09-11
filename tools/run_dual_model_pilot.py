"""One isolated serial short dual-MODEL job, guarded, no automatic retry."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def core_memory_options(command, disable_pinned_memory=False):
    """Explicit test-process options, never a user launcher/global setting edit."""
    result = list(command)
    if '--disable-pinned-memory' in result:
        raise ValueError('Core memory options are already configured')
    if disable_pinned_memory:
        result.append('--disable-pinned-memory')
    return result


def attach_first_frame(graph, core, filename):
    """Bind a real Core input file without copying it or altering the user's UI."""
    if filename is None:
        return None
    relative = Path(filename)
    input_root = (Path(core) / 'input').resolve(strict=True)
    if relative.is_absolute() or '..' in relative.parts or not filename.strip():
        raise ValueError('First frame must be a Core input-relative filename')
    source = (input_root / relative).resolve(strict=True)
    if not source.is_relative_to(input_root) or not source.is_file():
        raise ValueError('First frame must remain inside the Core input directory')
    if '23' in graph or 'first_frame' in graph['8']['inputs']:
        raise ValueError('Reference probe would overwrite an existing input')
    graph['23'] = {'class_type': 'LoadImage', 'inputs': {'image': relative.as_posix()}}
    graph['8']['inputs']['first_frame'] = ['23', 0]
    return file_identity(source)


def attach_memory_chunks(graph, backend, head_chunks=1, ffn_chunks=1):
    if head_chunks not in (1, 4) or ffn_chunks not in (1, 2):
        raise ValueError('Only explicit tested memory chunk recipes are supported by this harness')
    if head_chunks == ffn_chunks == 1:
        return
    if backend != 'kj-memory':
        raise ValueError('Memory chunk recipe requires the authenticated KJ memory backend')
    if any(key in graph for key in ('24', '25', '26', '27')):
        raise ValueError('Memory recipe would overwrite graph nodes')
    for branch, source, head_id, ffn_id in (('model_pass1', '21', '24', '26'),
                                            ('model_pass2', '22', '25', '27')):
        if head_chunks != 1:
            graph[head_id] = {'class_type': 'MiniMaxLowVRAMAttention',
                             'inputs': {'model': [source, 0], 'head_chunks': head_chunks}}
            source = head_id
        if ffn_chunks != 1:
            graph[ffn_id] = {'class_type': 'MiniMaxChunkFeedForward',
                            'inputs': {'model': [source, 0], 'chunks': ffn_chunks, 'seq_threshold': 4096}}
            source = ffn_id
        graph['8']['inputs'][branch] = [source, 0]


def attach_eav_stock20(graph, enabled=False):
    """An explicit separate recipe; never apply EAV to a four-step Turbo model."""
    if not enabled:
        return
    if (graph['1']['class_type'] != 'UNETLoader'
            or graph['2']['class_type'] != 'MiniMaxH3LoRACompatibilityLoaderT8Advanced'
            or graph['21']['inputs']['model'] != ['2', 0]
            or graph['22']['inputs']['model'] != ['3', 0]):
        raise ValueError('Unexpected graph for the Stock20 first-model recipe')
    graph['21']['inputs']['model'] = ['1', 0]
    del graph['2']
    graph['8']['inputs'].update(coarse_steps=20, eav_mode='apply_exp')


def attach_second_base(graph, core, filename):
    """Explicit existing native base for pass2; never replace the pass1 loader."""
    if filename is None:
        return None
    relative = Path(filename)
    root = (Path(core) / 'models/diffusion_models').resolve(strict=True)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.safetensors':
        raise ValueError('Second base must be a diffusion_models-relative safetensors file')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Second base must remain in the explicit model directory')
    if (relative.as_posix() == graph['1']['inputs']['unet_name'] or '28' in graph
            or graph['3']['inputs']['model'] != ['1', 0]):
        raise ValueError('Distinct second base would overwrite an unexpected graph')
    graph['28'] = {'class_type': 'UNETLoader', 'inputs': {
        'unet_name': relative.as_posix(), 'weight_dtype': 'default'}}
    graph['3']['inputs']['model'] = ['28', 0]
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--port', type=int, default=8208)
    parser.add_argument('--relay', action='store_true')
    parser.add_argument('--backend', choices=('kj', 'kj-memory', 'sol', 'pytorch'), default='kj')
    parser.add_argument('--sol-tau', type=float, choices=(0.0, 0.5, 1.3), default=1.3,
                        help='Explicit Sol-only diagnostic threshold; never silently change connected settings')
    parser.add_argument('--duration', type=int, choices=(3, 8, 24), default=3)
    parser.add_argument('--video-context-mode', choices=('reference_only','high_native_mask_exp'), default='reference_only',
                        help='Explicit single-variable continuation experiment; existing workflow default remains reference_only')
    parser.add_argument('--content', choices=('portrait', 'game'), default='portrait')
    parser.add_argument('--query-rows', type=int, choices=(256, 512, 1024), default=256)
    parser.add_argument('--first-frame', help='Existing filename relative to Core input; no UI changes')
    parser.add_argument('--memory-head-chunks', type=int, choices=(1, 4), default=1)
    parser.add_argument('--memory-ffn-chunks', type=int, choices=(1, 2), default=1)
    parser.add_argument('--eav-stock20', action='store_true', help='Separate base20 first pass plus EMA4 second pass; Relay/KJ only')
    parser.add_argument('--second-base', help='Existing diffusion_models-relative native H3 base for pass2 only')
    parser.add_argument('--disable-pinned-memory', action='store_true',
                        help='Explicit isolated Core option; no global/user configuration change')
    parser.add_argument('--headroom-gib', type=int, choices=(0, 2), default=0,
                        help='Additional native DynamicVRAM headroom, preserving the existing5GiB reserve and resource guards')
    args = parser.parse_args()
    if args.backend == 'kj-memory' and not args.relay:
        raise ValueError('This measured memory-composition probe requires Relay; plain direct-forward counting is separate')
    if args.eav_stock20 and (not args.relay or args.backend not in ('kj', 'kj-memory')):
        raise ValueError('EAV probe requires the explicit KJ/Relay combination')
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new artifact directory in this worktree')
    transport.CORE = args.core.resolve()
    transport.PROJECT = project
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        idx = result.index('--whitelist-custom-nodes') + 1
        result.insert(idx, 'ComfyUI-sol-attn' if args.backend == 'sol' else 'ComfyUI-KJNodes')
        return core_memory_options(result, args.disable_pinned_memory)

    transport.server_command = command
    template = project / 'artifacts/dual-workflows-cpu-v1' / ('Relay.prompt.json' if args.relay else 'Plain.prompt.json')
    graph = json.loads(template.read_text(encoding='utf-8'))
    graph['8']['inputs'].update(total_duration_seconds=float(args.duration), chain_id='dual_short_' + root.name,
        base_seed=2609032101, filename_prefix='Dual_Model_Pilot', query_chunk_rows=args.query_rows)
    for index, model in [('21', '2'), ('22', '3')]:
        if args.backend == 'kj':
            graph[index] = {'class_type': 'PathchSageAttentionKJ', 'inputs': {
                'model': [model, 0], 'sage_attention': 'auto', 'allow_compile': False}}
        elif args.backend == 'kj-memory':
            graph[index] = {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch',
                            'inputs': {'model': [model, 0]}}
        elif args.backend == 'sol':
            graph[index] = {'class_type': 'SolAttentionPatch', 'inputs': {
                'model': [model, 0], 'enabled': True, 'tau': args.sol_tau, 'min_tokens': 4096,
                'strict': True, 'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}}
        else:
            graph[index] = {'class_type': 'ModelAttentionBackend', 'inputs': {
                'model': [model, 0], 'attention': 'pytorch attention'}}
    graph['8']['inputs'].update(model_pass1=['21', 0], model_pass2=['22', 0])
    if args.video_context_mode != 'reference_only':
        if args.duration < 8:
            raise ValueError('Video context experiment needs a real continuation segment')
        graph['8']['inputs']['video_context_mode'] = args.video_context_mode
    attach_eav_stock20(graph, args.eav_stock20)
    attach_memory_chunks(graph, args.backend, args.memory_head_chunks, args.memory_ffn_chunks)
    second_base = attach_second_base(graph, args.core, args.second_base)
    if args.relay:
        graph['7']['inputs'].update(length=args.duration * 24 + 1, local_prompts='She looks toward the camera.\nShe speaks the sentence and listens.',
                                   time_ranges='0-50\n50-100')
        if args.duration >= 8:
            from tools.build_dual_model_workflows import SCENE_PROMPT, RELAY_EVENTS
            graph['7']['inputs'].update(global_prompt=SCENE_PROMPT, local_prompts=RELAY_EVENTS,
                                       time_ranges='0-15\n15-40\n40-75\n75-100')
            if args.duration == 24:
                graph['7']['inputs']['length'] = 583
    if args.content == 'game':
        if args.relay:
            raise ValueError('Game recipe currently qualified for plain timeline only; do not reuse portrait events')
        graph['8']['inputs']['global_prompt'] = transport.GAME_PROMPT
    reference = attach_first_frame(graph, args.core, args.first_frame)
    root.mkdir(parents=True)
    result = {'status': 'incomplete', 'mode': args.mode, 'relay': args.relay,
        'backend': args.backend, 'duration': args.duration, 'content': args.content,
        'first_frame_file': reference, 'memory_head_chunks': args.memory_head_chunks,
        'memory_ffn_chunks': args.memory_ffn_chunks, 'eav_stock20': args.eav_stock20,
        'disable_pinned_memory': args.disable_pinned_memory,
        'reserve_vram_gib': 5, 'headroom_gib': args.headroom_gib,
        'human_review': 'pending'}
    server = transport.OwnedServer(root, args.port, args.mode == 'cpu', args.headroom_gib)
    monitor = None
    guard = ResourceGuard()
    # Match the established host-wide acceleration lease; no unrelated process
    # is closed. Only this controller's server handle/children can be stopped.
    lease = args.core / 'custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock'
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            result['second_base_file'] = file_identity(second_base) if second_base else None
            source = transport.source_snapshot()
            source['tools/run_dual_model_pilot.py'] = file_identity(Path(__file__))['sha256']
            expected = {'core': verify_core_source(args.core), 'sources': source,
                        'mode': 'cpu-smoke' if args.mode == 'cpu' else 'gpu', 'pilot_graphs': {'dual_short': graph}}
            expected['runtime_options'] = {'reserve_vram_gib': 5, 'headroom_gib': args.headroom_gib}
            transport.write_json(root / 'paths.json', probe_resource_config(args.core, project))
            transport.write_json(root / 'expected.json', expected)
            if args.mode == 'gpu':
                reason = guard.observe(reader.sample(), startup=True)
                if reason:
                    raise RuntimeError('Startup resource guard: ' + reason)
            server.start()
            if args.mode == 'gpu':
                monitor = transport.ContinuousGuard(reader, guard, root / 'resources.jsonl', server)
                # Join the observer before NVML exits, including error paths.
                cleanup.callback(monitor.close)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            transport.wait_ready(server, check)
            history, _ = transport.execute_graph(server, {
                '1': {'class_type': 'T8ProgressiveEnvironmentAudit', 'inputs': {'expected_json': json.dumps(expected)}},
                '2': {'class_type': 'PreviewAny', 'inputs': {'source': ['1', 0]}}}, root / 'environment', check)
            result['environment'] = transport.preview_report(history, '2')
            transport.write_json(root / 'object-info.json', server.request('GET', '/object_info'))
            if args.mode == 'gpu':
                started = time.perf_counter()
                history, timing = transport.execute_graph(server, graph, root / 'generation', check, timeout=2400)
                result.update(status='generation_completed_independent_media_and_human_review_pending',
                              wall_seconds=time.perf_counter()-started, timing=timing)
            else:
                result['status'] = 'actual_owned_CPU_server_and_KJ_dual_graph_validation_pass_no_generation'
            if transport.source_snapshot() != {k: v for k, v in source.items() if k != 'tools/run_dual_model_pilot.py'}:
                raise RuntimeError('Runtime sources changed during the owned run')
            if reference and file_identity(Path(reference['path'])) != reference:
                raise RuntimeError('Reference image changed during the owned run')
            if second_base and file_identity(second_base) != result['second_base_file']:
                raise RuntimeError('Second base file changed during the owned run')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(root / 'terminal.json', result)
        print(json.dumps({'status': result['status'], 'root': str(root)}), flush=True)


if __name__ == '__main__':
    main()
