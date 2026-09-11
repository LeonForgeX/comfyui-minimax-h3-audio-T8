from copy import deepcopy

import pytest

from tools.run_legacy_loop_kj_probe import NODE, legacy_graph


def recipe():
    value = {key: {'class_type': 'fixture', 'inputs': {}} for key in
             ('1', '3', '4', '5', '6', '7', '22', '25', '27')}
    value.update({
        '21': {'class_type': 'MiniMaxH3MemoryEfficientSageAttentionPatch', 'inputs': {'model': ['1', 0]}},
        '24': {'class_type': 'MiniMaxLowVRAMAttention', 'inputs': {'model': ['21', 0], 'head_chunks': 4}},
        '26': {'class_type': 'MiniMaxChunkFeedForward', 'inputs': {'model': ['24', 0], 'chunks': 2, 'seq_threshold': 4096}},
        '8': {'inputs': {'coarse_steps': 20, 'refine_steps': 4, 'total_duration_seconds': 3,
            'eav_mode': 'apply_exp', 'prompt_relay_mode': 'apply_exp', 'model_pass1': ['26', 0],
            'chain_id': 'test', 'width': 1024, 'height': 512}},
    })
    return value


def schema():
    return {NODE: {'input': {'required': {
        key: ['STRING', {'default': default}] for key, default in
        [('eav_mode', 'disabled'), ('prompt_relay_mode', 'disabled'), ('total_duration_seconds', 30)]}}}}


def test_old_node_only_no_second_model_or_upscale():
    value = recipe()
    before = deepcopy(value)
    result = legacy_graph(value, schema())
    assert value == before
    assert result['8']['class_type'] == NODE
    settings = result['8']['inputs']
    assert settings['model'] == ['26', 0]
    assert (settings['width'], settings['height'], settings['steps']) == (512, 256, 20)
    assert settings['eav_mode'] == settings['prompt_relay_mode'] == 'apply_exp'
    assert not {'3', '22', '25', '27'} & result.keys()
    assert 'model_pass2' not in settings and 'upscaler_model' not in settings


@pytest.mark.parametrize('change', ['turbo', 'head', 'ffn', 'duration'])
def test_invalid_combination_stops_before_execution(change):
    value = recipe()
    if change == 'turbo':
        value['2'] = {'inputs': {}}
    elif change == 'head':
        value['24']['inputs']['head_chunks'] = 1
    elif change == 'ffn':
        value['26']['inputs']['chunks'] = 1
    else:
        value['8']['inputs']['total_duration_seconds'] = 24
    with pytest.raises(ValueError):
        legacy_graph(value, schema())
