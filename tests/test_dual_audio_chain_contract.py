from copy import deepcopy

import pytest
import torch

from tools.dual_audio_chain_contract import audit_audio_stages, audit_saved_contexts


def fixture():
    policy = {'effective_source': 'legacy_policy',
              'context_handoff': 'coarse_unlocked_template_locked_v1'}
    records = {k: {'contract': {'audio_policy_version': 2, 'audio_policy': deepcopy(policy)}}
               for k in ('low_x0', 'high_input', 'high_output')}
    records['low_x0']['report'] = {'schedule': {'coarse_audio_sigmas': [1., .75]}}
    records['high_output']['report'] = {'schedule': {'refine_audio_sigmas': [.7, 0.]},
        'audio_delivery': {'source': 'completed_second_pass_output'}}
    tensors = {'low_x0': {'samples_audio': torch.tensor([2., 2., 2., 2.])},
        'high_input': {'samples_audio': torch.tensor([9., 9., 2., 2.]),
                      'mask_audio': torch.tensor([0., 0., 1., 1.])},
        'high_output': {'samples_audio': torch.tensor([9., 9., 5., 5.])}}
    return records, tensors


def test_context_audio_and_generated_audio_have_separate_contracts():
    records, tensors = fixture()
    result = audit_audio_stages(records, tensors)
    assert result['generated_delta'] == 3 and result['locked_delta'] == 0


@pytest.mark.parametrize('fault', ['freeze', 'discard', 'context_changed', 'no_update',
                                 'partial_mask', 'stale_policy', 'policy_mismatch', 'nonzero_end'])
def test_invalid_joint_audio_is_rejected(fault):
    records, tensors = fixture()
    prepared, high = tensors['high_input'], tensors['high_output']
    if fault == 'freeze':
        prepared['mask_audio'].zero_()
    elif fault == 'discard':
        prepared['samples_audio'][2:] = 0
    elif fault == 'context_changed':
        high['samples_audio'][0] = 8
    elif fault == 'no_update':
        high['samples_audio'][2:] = 2
    elif fault == 'partial_mask':
        prepared['mask_audio'][0] = .5
    elif fault == 'stale_policy':
        for record in records.values():
            record['contract']['audio_policy'].pop('context_handoff')
    elif fault == 'policy_mismatch':
        records['high_input']['contract']['audio_policy'] = {}
    else:
        records['high_output']['report']['schedule']['refine_audio_sigmas'][-1] = .1
    with pytest.raises(ValueError):
        audit_audio_stages(records, tensors)


def test_old_evidence_retains_exact_audio_requirement():
    records, tensors = fixture()
    for record in records.values():
        record['contract'] = {}
    for tensor in tensors.values():
        tensor['samples_audio'].fill_(2)
    tensors['high_input']['mask_audio'].zero_()
    assert audit_audio_stages(records, tensors)['mode'] == 'historical_first_pass_locked'
    tensors['high_output']['samples_audio'][0] += .01
    with pytest.raises(ValueError):
        audit_audio_stages(records, tensors)


@pytest.mark.parametrize('version', [None, 1, 3, '2'])
def test_unknown_policy_cannot_masquerade_as_old_locked_evidence(version):
    records, tensors = fixture()
    for record in records.values():
        record['contract']['audio_policy_version'] = version
    for tensor in tensors.values():
        tensor['samples_audio'].fill_(2)
    tensors['high_input']['mask_audio'].zero_()
    with pytest.raises(ValueError, match='Unknown audio policy'):
        audit_audio_stages(records, tensors)


@pytest.mark.parametrize('fault', [None, 'coarse_audio', 'video_geometry', 'different_tail', 'dtype'])
def test_saved_contexts_must_use_completed_audio(fault):
    low = {'samples_video': torch.ones(1, 2, 4, 2, 2)}
    high = {'samples_video': torch.ones(1, 2, 4, 4, 4) * 3,
            'samples_audio': torch.ones(1, 2, 2, 8) * 7}
    lo = {'video_tail': low['samples_video'][:, :, -2:],
          'audio_tail': high['samples_audio'][..., -4:].clone()}
    hi = {'video_tail': high['samples_video'][:, :, -2:],
          'audio_tail': high['samples_audio'][..., -4:].clone()}
    if fault == 'coarse_audio':
        lo['audio_tail'].fill_(1)
    elif fault == 'video_geometry':
        lo['video_tail'] = hi['video_tail']
    elif fault == 'different_tail':
        lo['audio_tail'] = lo['audio_tail'][..., -2:]
    elif fault == 'dtype':
        lo['audio_tail'] = lo['audio_tail'].double()
    if fault is None:
        audit_saved_contexts(low, high, lo, hi)
    else:
        with pytest.raises(ValueError):
            audit_saved_contexts(low, high, lo, hi)
