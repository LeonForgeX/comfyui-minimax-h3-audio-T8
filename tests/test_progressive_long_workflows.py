"""Public candidate connection contracts, independent of model execution."""

import pytest

from tools.build_progressive_long_workflows import (
    NODE,
    REVIEWED_EAV_END,
    REVIEWED_EAV_START,
    REVIEWED_EAV_TAU,
    build_public_graph,
)


@pytest.mark.parametrize('backend', ['pytorch', 'kj-memory', 'sol'])
@pytest.mark.parametrize('tst', ['disabled', 'apply_exp'])
def test_public_models_and_full_sigma_wiring(backend, tst):
    info = {NODE: {'input': {'required': {
        'low_evaluations': ['INT', {'default': 4}],
        'resume_existing': ['BOOLEAN', {'default': True}]}}}}
    graph = build_public_graph(info, 'new_chain', backend, True, 'apply_exp', tst)
    assert graph['8']['class_type'] == NODE
    inputs = graph['8']['inputs']
    assert inputs['sampler'] == ['12', 2] and inputs['sigmas'] == ['12', 3]
    assert inputs['low_evaluations'] == 4 and inputs['resume_existing'] is True
    assert inputs['eav_mode'] == 'apply_exp'
    assert inputs['eav_tau'] == REVIEWED_EAV_TAU == 8.0
    assert inputs['eav_start_video_progress'] == REVIEWED_EAV_START == 0.15
    assert inputs['eav_end_video_progress'] == REVIEWED_EAV_END == 0.90
    assert 'tst_mode' not in inputs  # Independent MODEL nodes, not double installation.
    assert graph['12']['inputs']['steps'] == 8
    assert graph['12']['inputs']['model'] == ['21', 0]
    assert graph['12']['inputs']['model_hires'] == ['22', 0]
    if tst == 'disabled':
        assert inputs['model'] == ['12', 0] and inputs['model_hires'] == ['12', 1]
        assert not {'13', '14'}.intersection(graph)
    else:
        for key in ('13', '14'):
            assert graph[key]['class_type'] == 'MiniMaxH3TSTModelEXPT8'
            assert graph[key]['inputs']['sigmas'] == ['12', 3]
        assert inputs['model'] == ['13', 0] and inputs['model_hires'] == ['14', 0]
    assert graph['9']['inputs']['source'] == ['8', 2]
    assert graph['10']['inputs']['source'] == ['8', 1]
    assert inputs['prompt_relay_plan'] == ['11', 0]
