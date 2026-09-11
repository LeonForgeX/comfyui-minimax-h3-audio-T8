"""Current audio continuation and matched-memory benchmark receipts; no inference."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools import summarize_dual_backend_benchmark as summary
from tools.progressive_probe_control import summarize_execution_events
from test_dual_backend_benchmark_summary import graph


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf8')


def make_pair(root, backend, *, unpinned=True, long_name=False):
    memory = {'reserve_vram_gib': 5, 'headroom_gib': 2}
    terminal = {'status': 'generation_completed_independent_media_and_human_review_pending',
        'mode': 'gpu', 'backend': backend, 'disable_pinned_memory': unpinned, **memory,
        'second_base_file': None, 'server_stop': {'pid': 123, 'owned_children_remaining': []},
        'resources': {'status': 'observations_within_policy', 'gpu_uuid': 'gpu-fixture',
            'maximum_observed_device_used_bytes': 100, 'minimum_ram_available_bytes': 200},
        'environment': {'status': 'pass', 'device': 'cuda:0', 'pid': 123,
            'torch_version': 'fixture', 'torch_cuda': 'fixture', 'runtime_options': memory}}
    save(root / 'terminal.json', terminal)
    command = ['python', 'main.py', '--cache-none', '--reserve-vram', '5', '--vram-headroom', '2']
    if unpinned:
        command.append('--disable-pinned-memory')
    save(root / 'server-command.json', command)
    cold_graph = graph(backend)
    if long_name:
        cold_graph['8']['inputs']['chain_id'] = backend + '-long-measured-namespace' * 4
    save(root / 'expected.json', {'sources': {'runtime': 'fixture'}, 'core': {'commit': 'fixture'},
        'runtime_options': memory, 'pilot_graphs': {'dual_short': cold_graph}})
    pair = {'status': 'two_uncached_jobs_completed_independent_audit_pending'}
    for label, folder in [('cold', 'generation'), ('process_warm', 'generation-process-warm')]:
        submitted = deepcopy(cold_graph)
        chain = cold_graph['8']['inputs']['chain_id'] + ('_process_warm' if label == 'process_warm' else '')
        submitted['8']['inputs']['chain_id'] = chain
        save(root / folder / 'prompt.json', submitted)
        events = [{'type': 'executing', 'elapsed_seconds': i + 1,
            'data': {'prompt_id': chain, 'node': node}} for i, node in enumerate(submitted)]
        events.append({'type': 'execution_success', 'elapsed_seconds': len(events) + 1,
            'data': {'prompt_id': chain}})
        elapsed = len(events) + 1
        timing = summarize_execution_events(events, chain, submitted, elapsed)
        save(root / folder / 'timing.json', timing)
        save(root / folder / 'submission.json', {'prompt_id': chain})
        save(root / folder / 'history.json', {'status': {'completed': True}})
        (root / folder / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events), encoding='utf8')
        pair[label] = {'chain_id': chain, 'timing': timing}
        from h3_audio_t8_pkg.long_video import sanitize_chain_id
        actual_chain = sanitize_chain_id(chain)
        chain_root = root / 'output/minimax_h3_t8_long_video' / actual_chain
        chain_root.mkdir(parents=True)
        media = chain_root / 'video.mp4'
        media.write_bytes(b'fixture-only-not-a-real-video')
        save(root / folder / 'history.json', {'status': {'completed': True},
            'outputs': {'8': {'images': [{'type': 'output', 'filename': 'video.mp4',
                'subfolder': 'minimax_h3_t8_long_video/' + actual_chain}]}}})
        audit = {'selected_chain': actual_chain, 'backend': backend, 'actual_forwards': [4, 4],
            'status': 'short_mechanical_audit_pass_human_pending', 'audio_input_bit_exact': True,
            'audio_output_bit_exact': False, 'audio_output_max_absolute_delta': 1.9,
            'audio_contract': {'mode': 'joint_second_pass_continuation', 'max_absolute_delta': 1.9,
                'lock_tolerance': None, 'quality': 'requires_human_review'},
            'stage_sha256': {}, 'backend_completed_calls': [{'fixture': 200}, {'fixture': 200}],
            'media': {'path': str(media), 'sha256': hashlib.sha256(media.read_bytes()).hexdigest()}}
        for stage in ('low_x0', 'high_output'):
            digest = hashlib.sha256(stage.encode()).hexdigest()
            audit['stage_sha256'][stage] = digest
            save(chain_root / (stage + '-fixture.json'), {'tensor_sha256': digest,
                'report': {'sample_seconds': 1.0, 'preparation': {'upscale_seconds': 0.1}}})
        save(root / ('audit-' + label + '.json'), audit)
    save(root / 'benchmark-pair.json', pair)


def invoke(tmp_path, monkeypatch, *, unpinned=True, mutate=None, long_name=False):
    roots = [tmp_path / name for name in ('pytorch', 'kj', 'sol')]
    for root, name in zip(roots, ('pytorch', 'kj', 'sol')):
        make_pair(root, name, unpinned=unpinned, long_name=long_name)
    if mutate:
        mutate(roots[1])
    output = tmp_path / 'summary.json'
    monkeypatch.setattr('sys.argv', ['summary', '--roots', *map(str, roots), '--output', str(output)])
    summary.main()
    return json.loads(output.read_text())


@pytest.mark.parametrize('unpinned', [True, False])
def test_current_joint_audio_qualifies_when_memory_recipe_matches(tmp_path, monkeypatch, unpinned):
    result = invoke(tmp_path, monkeypatch, unpinned=unpinned)
    assert len(result['rows']) == 6
    assert result['audio_policy'] == 'joint_second_pass_continuation'
    assert result['runtime_options']['disable_pinned_memory'] is unpinned


def test_native_safe_chain_alias_is_bound_through_actual_history(tmp_path, monkeypatch):
    assert len(invoke(tmp_path, monkeypatch, long_name=True)['rows']) == 6


def test_known_external_interference_excludes_entire_pair(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='excluded'):
        invoke(tmp_path, monkeypatch, mutate=lambda root: save(root / 'benchmark-exclusion.json',
            {'status': 'excluded_from_comparative_performance'}))


def change(root, file, mutate):
    path = root / file
    value = json.loads(path.read_text())
    mutate(value)
    save(path, value)


@pytest.mark.parametrize('case', ['locked_audio', 'missing_audio', 'unchanged_audio', 'nan_audio',
    'handoff_changed', 'headroom_mismatch', 'pinned_command', 'duplicate_reserve',
    'live_mismatch', 'negative_time', 'cached_events', 'changed_media', 'wrong_graph',
    'null_audio', 'reused_chain', 'different_matched_memory'])
def test_current_benchmark_rejects_stale_or_unmatched_evidence(tmp_path, monkeypatch, case):
    def corrupt(root):
        if case in ('locked_audio', 'missing_audio', 'unchanged_audio', 'nan_audio', 'handoff_changed', 'null_audio'):
            def update(a):
                if case == 'locked_audio':
                    a['audio_contract']['mode'] = 'first_pass_locked'
                elif case == 'missing_audio':
                    a.pop('audio_contract')
                elif case == 'null_audio':
                    a['audio_contract'] = None
                elif case == 'handoff_changed':
                    a['audio_input_bit_exact'] = False
                else:
                    delta = 0.0 if case == 'unchanged_audio' else float('nan')
                    a['audio_contract']['max_absolute_delta'] = delta
                    a['audio_output_max_absolute_delta'] = delta
            change(root, 'audit-cold.json', update)
        elif case == 'reused_chain':
            change(root, 'benchmark-pair.json', lambda p: p['process_warm'].update(chain_id=p['cold']['chain_id']))
        elif case == 'different_matched_memory':
            change(root, 'terminal.json', lambda t: t.update(disable_pinned_memory=False))
            change(root, 'server-command.json', lambda c: c.remove('--disable-pinned-memory'))
        elif case == 'headroom_mismatch':
            change(root, 'terminal.json', lambda t: t.update(headroom_gib=0))
        elif case == 'live_mismatch':
            change(root, 'terminal.json', lambda t: t['environment']['runtime_options'].update(headroom_gib=0))
        elif case == 'pinned_command':
            change(root, 'server-command.json', lambda c: c.remove('--disable-pinned-memory'))
        elif case == 'duplicate_reserve':
            change(root, 'server-command.json', lambda c: c.extend(['--reserve-vram', '0']))
        elif case == 'negative_time':
            change(root, 'generation/timing.json', lambda t: t.update(elapsed_seconds=-1))
        elif case == 'cached_events':
            path = root / 'generation/events.jsonl'
            path.write_text(json.dumps({'type': 'execution_cached', 'elapsed_seconds': 0,
                'data': {'prompt_id': 'kj', 'nodes': ['8']}}) + '\n' + path.read_text())
        elif case == 'changed_media':
            (root / 'output/minimax_h3_t8_long_video/kj/video.mp4').write_bytes(b'changed')
        else:
            change(root, 'expected.json', lambda e: e['pilot_graphs']['dual_short']['8']['inputs'].update(base_seed=9))
    with pytest.raises(ValueError):
        invoke(tmp_path, monkeypatch, mutate=corrupt)
    assert not (tmp_path / 'summary.json').exists()
