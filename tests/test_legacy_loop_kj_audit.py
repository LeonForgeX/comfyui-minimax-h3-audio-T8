import hashlib
import json

import pytest

from tools.audit_legacy_loop_kj_probe import check_effects


def evidence():
    return {'contract_sha256': 'contract', 'segment_index': 0, 'candidate_id': 'candidate',
        'sampling_plan': {'second_pass_nfe': 0}, 'enhance_a_video_audit': {
            'status': 'apply_exp_long_video_segment_verified', 'aborted': None,
            'model_forward_count': 20, 'attention_measurement_count': 1000,
            'attention_calls_per_active_forward': [50] * 20,
            'config': {'composed_attention_backend': {'kind': 'audited_kj_selector',
                'completed_calls': {'sage:biased': 76000, 'sage:unbiased': 4000},
                'memory_composition': {'head_chunks': 4, 'ffn_settings': [2, 4096], 'kind': 'kj_memory_sage'}}}}}


def sign(data):
    data['audit_sha256'] = hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return data


def test_bound_single_stock20_evidence():
    backend = check_effects(sign(evidence()), {'contract_sha256': 'contract'}, {'candidate_id': 'candidate'})
    assert backend['completed_calls']['sage:biased'] == 76000


@pytest.mark.parametrize('change', ['hash', 'candidate', 'nfe', 'fallback', 'ffn', 'second_pass'])
def test_incomplete_or_unbound_evidence_rejected(change):
    data = evidence()
    if change == 'candidate':
        data['candidate_id'] = 'different'
    elif change == 'nfe':
        data['enhance_a_video_audit']['model_forward_count'] = 19
    elif change == 'fallback':
        data['enhance_a_video_audit']['config']['composed_attention_backend']['completed_calls']['pytorch:fallback'] = 1
    elif change == 'ffn':
        data['enhance_a_video_audit']['config']['composed_attention_backend']['memory_composition']['ffn_settings'] = [1, 4096]
    elif change == 'second_pass':
        data['sampling_plan']['second_pass_nfe'] = 4
    sign(data)
    if change == 'hash':
        data['audit_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        check_effects(data, {'contract_sha256': 'contract'}, {'candidate_id': 'candidate'})
