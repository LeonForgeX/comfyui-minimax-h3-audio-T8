from copy import deepcopy
import hashlib
import json

import pytest

from tools.audit_dual_execution_timings import audit, sha, timing_rows


def ledger(names, start):
    return {'clock': 'perf_counter', 'gpu_synchronization_added': False,
        'historical_cache_time_counted_as_current': False,
        'events': [{'phase': name, 'start': start + i * .2, 'end': start + i * .2 + .1,
                    'seconds': .1, 'status': 'completed'} for i, name in enumerate(names)]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def sign(payload):
    value = deepcopy(payload)
    value.pop('audit_sha256', None)
    value['audit_sha256'] = hashlib.sha256(json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


@pytest.fixture
def completed(tmp_path):
    root = tmp_path
    chain = root / 'output/chain'
    chain.mkdir(parents=True)
    media = chain / 'final.mp4'
    media.write_bytes(b'fixture-not-real-media')
    segments = [{'index': i, 'candidate_id': f'c{i}'} for i in range(2)]
    write(root / 'terminal.json', {'status': 'generation_completed',
        'server_stop': {'owned_children_remaining': []}})
    write(chain / 'manifest.json', {'segments': segments})
    write(chain / 'in_node_loop_effects_state.json', {'status': 'complete', 'accepted_count': 2,
        'contract_sha256': 'job', 'final_video_path': 'final.mp4', 'final_video_sha256': sha(media)})
    write(root / 'independent-audit-v1.json', {'status': 'multi_segment_mechanical_pass_human_pending',
        'segments': segments, 'media': {'path': str(media), 'sha256': sha(media)}})
    write(chain / 'last_execution_report.json', {'status': 'complete', 'contract_sha256': 'job',
        'composition_execution_timings': ledger(['final_composition'], 10)})
    samples = [{'monotonic': i, 'clock': 'perf_counter', 'gpu_used_bytes': 100 + i,
                'ram_available_bytes': 1000 - i} for i in range(12)]
    (root / 'resources.jsonl').write_text('\n'.join(json.dumps(s) for s in samples))
    for i, segment in enumerate(segments):
        forward = {'completed_network_forwards': 4, 'execution_timings': ledger(
            ['prepare_sampling_including_model_load'] + ['forward_including_dynamic_transfers'] * 4, i * 5 + .1)}
        payload = {'segment_index': i, 'candidate_id': segment['candidate_id'], 'contract_sha256': 'job',
            'delivery_execution_timings': ledger(['av_decode', 'trim', 'candidate_encode_and_save'], i * 5 + 3),
            'sampling_plan': {'dual_model': {'low_reused': False, 'high_reused': False,
                'first_pass': forward, 'second_pass': deepcopy(forward),
                'execution_timings': ledger(['first_conditioning', 'first_sampling', 'second_conditioning',
                    'learned_upscale', 'reconcile', 'second_sampling'], i * 5 + .1)}}}
        write(chain / f'candidates/segment_{i:05d}/c{i}/effects_audit.json', sign(payload))
    return root


def test_all_segments_and_final_composition_are_measured(completed):
    result = audit(completed)
    assert result['segments'] == 2
    assert len(result['rows']) == 39
    assert sum(row['scope'] == 'composition' for row in result['rows']) == 1
    assert {row['segment'] for row in result['rows']} == {None, 0, 1}


@pytest.mark.parametrize('change', ['checksum', 'wrong_job', 'cache', 'missing_phase', 'forwards'])
def test_incomplete_or_reused_segment_evidence_is_not_speed_evidence(completed, change):
    path = completed / 'output/chain/candidates/segment_00001/c1/effects_audit.json'
    payload = json.loads(path.read_text())
    if change in ('checksum', 'wrong_job'):
        payload['contract_sha256'] = 'other'
    elif change == 'cache':
        payload['sampling_plan']['dual_model']['low_reused'] = True
    elif change == 'missing_phase':
        payload['sampling_plan']['dual_model']['execution_timings']['events'].pop()
    else:
        payload['sampling_plan']['dual_model']['second_pass']['completed_network_forwards'] = 3
    write(path, payload if change == 'checksum' else sign(payload))
    with pytest.raises(ValueError):
        audit(completed)


def test_changed_final_media_invalidates_timing_bundle(completed):
    (completed / 'output/chain/final.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        audit(completed)


@pytest.mark.parametrize('field,value', [('start', -1), ('end', 30), ('seconds', 99),
    ('start', float('nan')), ('start', True), ('status', 'failed')])
def test_invalid_times_never_enter_report(field, value):
    data = ledger(['example'], 1)
    data['events'][0][field] = value
    with pytest.raises(ValueError):
        timing_rows(data, [{'monotonic': 0}, {'monotonic': 10}], 'fixture', 0)


def test_missing_resource_sample_is_unobserved_not_zero():
    rows = timing_rows(ledger(['short_event'], 1), [{'monotonic': 0}, {'monotonic': 10}], 'fixture', 0)
    assert rows[0]['resource_sample_count'] == 0
    assert rows[0]['maximum_observed_device_used_bytes'] is None
    assert rows[0]['minimum_observed_system_ram_available_bytes'] is None
