from copy import deepcopy

import pytest
import torch

from tools.audit_dual_model_pilot import validate_audio_stage_contract


def fixture():
    policy = {'effective_source': 'legacy_policy'}
    records = {
        'low_x0': {'contract': {'audio_policy': policy},
                   'report': {'schedule': {'coarse_audio_sigmas': [1., .75]}}},
        'high_output': {'contract': {'audio_policy_version': 2, 'audio_policy': policy},
                        'report': {'schedule': {'refine_audio_sigmas': [.7, 0.]},
                                   'audio_delivery': {'source': 'completed_second_pass_output'}}}}
    tensors = {'low_x0': {'samples_audio': torch.ones(3)},
               'high_input': {'samples_audio': torch.ones(3)},
               'high_output': {'samples_audio': torch.full((3,), 2.)}}
    return records, tensors


def test_joint_audio_audit_requires_actual_continuation():
    records, tensors = fixture()
    assert validate_audio_stage_contract(records, tensors)['mode'] == 'joint_second_pass_continuation'


@pytest.mark.parametrize('error', ['mask', 'unchanged', 'handoff', 'policy', 'zero_end', 'replacement'])
def test_joint_audio_audit_rejects_wrong_or_frozen_audio(error):
    records, tensors = deepcopy(fixture())
    if error == 'mask':
        tensors['high_input']['mask_audio'] = torch.zeros(3)
    elif error == 'unchanged':
        tensors['high_output']['samples_audio'] = torch.ones(3)
    elif error == 'handoff':
        tensors['high_input']['samples_audio'] = torch.zeros(3)
    elif error == 'policy':
        records['low_x0']['contract']['audio_policy'] = {}
    elif error == 'zero_end':
        records['high_output']['report']['schedule']['refine_audio_sigmas'][-1] = .1
    else:
        records['high_output']['report']['audio_delivery']['source'] = 'first_pass_exact_latent'
    with pytest.raises(ValueError):
        validate_audio_stage_contract(records, tensors)
