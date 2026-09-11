import hashlib
import json

import pytest

from tools.audit_dual_model_pilot import audit_first_frame, read_bound_high_condition


@pytest.mark.parametrize('change', [None, 'source', 'task', 'warning', 'context', 'segment'])
def test_first_frame_requires_both_stage_conditioning_and_current_bytes(tmp_path, change):
    path = tmp_path / 'image.png'
    path.write_bytes(b'image')
    terminal = {'first_frame_file': {'path': str(path), 'bytes': 5,
                                   'sha256': hashlib.sha256(b'image').hexdigest()}}
    records = {stage: {'report': {'conditioning': {'task': 'i2va', 'context_active': False,
                'segment_index': 0, 'warnings': []}}} for stage in ('low_x0', 'high_output')}
    condition = records['high_output']['report']['conditioning']
    if change == 'source':
        path.write_bytes(b'other')
    elif change == 'task':
        condition['task'] = 't2va'
    elif change == 'warning':
        condition['warnings'] = ['ignored reference']
    elif change == 'context':
        condition['context_active'] = True
    elif change == 'segment':
        condition['segment_index'] = 1
    if change:
        with pytest.raises(ValueError):
            audit_first_frame(terminal, records)
    else:
        assert audit_first_frame(terminal, records)['actual_stage_tasks'] == ['i2va', 'i2va']


def test_no_image_does_not_claim_i2va_audit():
    assert audit_first_frame({}, {}) is None


@pytest.mark.parametrize('change', [None, 'checksum', 'job', 'candidate', 'stage', 'segment'])
def test_high_condition_sidecar_is_bound_to_actual_stage_records(tmp_path, change):
    folder = tmp_path / 'output/candidate'
    folder.mkdir(parents=True)
    records = {'low_x0': {'contract': {'job': 'job'}, 'report': {'actual': 'low'}},
               'high_output': {'report': {'actual': 'high'}}}
    payload = {'schema': 1, 'format': 'minimax_h3_t8_in_node_loop_effects_segment',
               'contract_sha256': 'job', 'candidate_id': 'candidate', 'segment_index': 0,
               'conditioning': {'task': 'i2va'}, 'sampling_plan': {'dual_model': {
                   'first_pass': {'actual': 'low'}, 'second_pass': {'actual': 'high'}}}}
    if change == 'job':
        payload['contract_sha256'] = 'other'
    elif change == 'candidate':
        payload['candidate_id'] = 'other'
    elif change == 'segment':
        payload['segment_index'] = 1
    elif change == 'stage':
        payload['sampling_plan']['dual_model']['second_pass'] = {'actual': 'other'}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    payload['audit_sha256'] = 'incorrect' if change == 'checksum' else digest
    (folder / 'effects_audit.json').write_text(json.dumps(payload), encoding='utf8')
    if change:
        with pytest.raises(ValueError):
            read_bound_high_condition(tmp_path, records)
    else:
        condition, proof = read_bound_high_condition(tmp_path, records)
        assert condition == {'task': 'i2va'} and proof['audit_sha256'] == digest
