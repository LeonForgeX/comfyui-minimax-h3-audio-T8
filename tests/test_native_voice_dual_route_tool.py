import copy
import json
from pathlib import Path

import pytest

from tools.qualify_native_voice_dual_gpu import prepared_probe_graph, with_query_route


def test_formal_joint_frontend_explicit_route_and_generated_audio_only():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root / 'examples/workflows/36-avatar-voice/2026-09-18_T8_voice_dual_joint_EXP.json').read_text(encoding='utf8'))
    by_type = {node['type']: node for node in workflow['nodes']}
    plan = by_type['MiniMaxH3PromptRelayPlanT8Advanced']
    route = by_type['MiniMaxH3PromptRelayQueryRouteT8Advanced']
    loop = by_type['MiniMaxH3DualModelLongVideoEXPT8']
    links = {link[0]: link for link in workflow['links']}
    assert route['widgets_values'] == ['joint_av_exp']
    assert links[route['inputs'][0]['link']][1:5] == [plan['id'], 0, route['id'], 0]
    pins = {pin['name']: pin for pin in loop['inputs']}
    assert links[pins['prompt_relay_plan']['link']][1:3] == [route['id'], 0]
    assert pins['drive_audio']['link'] is None and pins['final_audio']['link'] is None
    assert pins['ref_audios.ref_audio_0']['link'] is not None
    assert '指定样片已获用户验收' in by_type['MarkdownNote']['widgets_values'][0]
    assert workflow['extra']['t8_release'] == '1.85.0'
    assert workflow['extra']['t8_seam_event_acceptance'] == 'NA_not_promoted'


def test_formal_single_model_joint_candidate_keeps_reference_only_and_no_upscale():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root / 'examples/workflows/36-avatar-voice/2026-09-18_T8_voice_long_joint_EXP.json').read_text(encoding='utf8'))
    by_type = {node['type']: node for node in workflow['nodes']}
    loop = by_type['MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced']
    route = by_type['MiniMaxH3PromptRelayQueryRouteT8Advanced']
    assert route['widgets_values'] == ['joint_av_exp']
    assert not any('DualModel' in name or 'Upscaler' in name for name in by_type)
    pins = {pin['name']: pin for pin in loop['inputs']}
    links = {link[0]: link for link in workflow['links']}
    assert links[pins['prompt_relay_plan']['link']][1:3] == [route['id'], 0]
    assert pins['drive_audio']['link'] is None and pins['final_audio']['link'] is None
    assert pins['ref_audios.ref_audio_0']['link'] is not None
    chunk = by_type['MiniMaxH3ChunkFeedForwardT8Advanced']
    assert links[pins['model']['link']][1:3] == [chunk['id'], 0]
    assert '指定样片已获用户验收' in by_type['MarkdownNote']['widgets_values'][0]
    assert workflow['extra']['t8_release'] == '1.85.0'
    assert workflow['extra']['t8_seam_event_acceptance'] == 'NA_not_promoted'


def recipe():
    return {'7': {'inputs': {'global_prompt': 'identity', 'local_prompts': 'new words', 'length': 193}},
            '8': {'inputs': {'prompt_relay_plan': ['7', 0], 'coarse_steps': 20, 'refine_steps': 4,
                             'second_audio_source': 'auto', 'second_audio_strength': 0,
                             'context_frames': 22, 'audio_seam_policy': 'cosine_bridge'}}}


def test_paper_default_is_exact_independent_copy():
    original = recipe()
    observed = with_query_route(original, 'video_only_paper')
    assert observed == original and observed is not original


def test_joint_route_only_connects_explicit_existing_node_no_math_change():
    original = recipe()
    frozen = copy.deepcopy(original)
    observed = with_query_route(original, 'joint_av_exp')
    assert original == frozen
    assert observed['12']['inputs'] == {'prompt_relay_plan': ['7', 0], 'query_route': 'joint_av_exp'}
    assert observed['8']['inputs'].pop('prompt_relay_plan') == ['12', 0]
    assert observed['8']['inputs'] == {k: v for k, v in original['8']['inputs'].items() if k != 'prompt_relay_plan'}
    assert observed['7'] == original['7']


@pytest.mark.parametrize('fault', ['unknown', 'occupied', 'foreign'])
def test_unknown_owners_rejected_without_mutating_recipe(fault):
    original = recipe()
    if fault == 'occupied':
        original['12'] = {'inputs': {}}
    elif fault == 'foreign':
        original['8']['inputs']['prompt_relay_plan'] = ['foreign', 0]
    frozen = copy.deepcopy(original)
    with pytest.raises(ValueError):
        with_query_route(original, 'unknown' if fault == 'unknown' else 'joint_av_exp')
    assert original == frozen


@pytest.mark.parametrize('single_model', [False, True])
def test_probe_chunks_only_change_known_model_pins(single_model):
    original = recipe()
    original['8']['class_type'] = ('MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced' if single_model
                                 else 'MiniMaxH3DualModelLongVideoEXPT8')
    pins = ('model',) if single_model else ('model_pass1', 'model_pass2')
    original['8']['inputs'].update({pin: ['1', 0] for pin in pins})
    frozen = copy.deepcopy(original)
    observed = prepared_probe_graph(original, single_model=single_model)
    assert original == frozen
    for pin in pins:
        assert observed['8']['inputs'].pop(pin) == ['11', 0]
    assert observed['8']['inputs'] == {key: value for key, value in frozen['8']['inputs'].items() if key not in pins}
    assert observed['7'] == frozen['7']
    assert observed['10']['inputs']['head_chunks'] == 4
    assert observed['11']['inputs']['chunks'] == 2
    assert prepared_probe_graph(prepared_probe_graph(original, single_model=single_model),
                                single_model=single_model) == prepared_probe_graph(original, single_model=single_model)


@pytest.mark.parametrize('fault', ['wrong_mode', 'foreign_model', 'foreign_chunks'])
def test_probe_rejects_unknown_owners_without_mutation(fault):
    original = recipe()
    original['8']['class_type'] = 'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'
    original['8']['inputs']['model'] = ['1', 0]
    if fault == 'wrong_mode':
        original['8']['class_type'] = 'other'
    elif fault == 'foreign_model':
        original['8']['inputs']['model'] = ['foreign', 0]
    else:
        original['10'] = {'class_type': 'other', 'inputs': {}}
    frozen = copy.deepcopy(original)
    with pytest.raises(ValueError):
        prepared_probe_graph(original, single_model=True)
    assert original == frozen


def test_window_graph_only_adds_explicit_text_owner_and_does_not_retime_dialogue():
    from tools.qualify_native_voice_dual_gpu import with_window_text_graph
    original = with_query_route(recipe(), 'joint_av_exp')
    frozen = copy.deepcopy(original)
    observed = with_window_text_graph(original)
    assert original == frozen and observed['7'] == frozen['7']
    assert observed['8']['inputs'].pop('prompt_relay_plan') == ['13', 0]
    assert observed['8']['inputs'] == {k:v for k,v in frozen['8']['inputs'].items() if k != 'prompt_relay_plan'}
    observed['8']['inputs']['prompt_relay_plan'] = ['13', 0]
    assert with_window_text_graph(observed) == observed
    observed['13']['inputs']['text_policy'] = 'unknown'
    with pytest.raises(ValueError):
        with_window_text_graph(observed)


def test_dialogue_owner_graph_is_separate_allowlisted_connection_not_silent_migration():
    from tools.qualify_native_voice_dual_gpu import with_window_text_graph
    original = with_query_route(recipe(), 'joint_av_exp')
    frozen = copy.deepcopy(original)
    owned = with_window_text_graph(original, text_policy='dialogue_start_owner_exp')
    accepted = with_window_text_graph(original)
    assert original == frozen and owned['7'] == frozen['7']
    assert owned['8'] == accepted['8']
    assert owned['13']['inputs']['text_policy'] == 'dialogue_start_owner_exp'
    assert with_window_text_graph(owned, text_policy='dialogue_start_owner_exp') == owned
    with pytest.raises(ValueError):
        with_window_text_graph(accepted, text_policy='dialogue_start_owner_exp')
    with pytest.raises(ValueError):
        with_window_text_graph(owned)
    with pytest.raises(ValueError):
        with_window_text_graph(original, text_policy='unknown')
