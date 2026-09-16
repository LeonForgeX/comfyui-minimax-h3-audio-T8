from pathlib import Path
import sys

import pytest


def test_candidate_recipes_are_full_students_with_native_samplers_and_no_probe_nodes():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from build_fast_h3_v2_workflows import recipes
    from run_fast_h3_v2_probe import MODEL
    cases = recipes()
    assert len(cases) == 4
    for graph in cases.values():
        assert graph['1']['inputs']['unet_name'] == MODEL
        assert graph['13']['class_type'] == 'SamplerCustomAdvanced'
        assert graph['10']['inputs']['min_tokens'] == 12288
        assert graph['16']['inputs']['codec'] == 'h264'
        assert not any(node['class_type'].startswith('T8FastH3V2') for node in graph.values())
        assert not any('LoRA' in node['class_type'] for node in graph.values())
    portrait = cases['FastH3_V2_Dense_First_Frame_T8_Memory_EXP']['9']['inputs']
    assert (portrait['width'], portrait['height']) == (512, 768)
    assert portrait['first_frame'] == ['23', 0]
    from h3_audio_t8_pkg.conditioning import resolve_task_type
    assert resolve_task_type(portrait['task_type'], object(), None, False) == 'i2va'


def test_native_dynamic_save_codec_is_expanded_from_selected_schema_not_guessed():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    from build_fast_h3_v2_workflows import selected_frontend_schema
    codec = ['COMFY_DYNAMICCOMBO_V3', {'options': [
        {'key': 'auto', 'inputs': {'required': {}}},
        {'key': 'h264', 'inputs': {'required': {}, 'optional': {
            'encoding': ['COMFY_DYNAMICCOMBO_V3', {'options': [
                {'key': 'auto', 'inputs': {'required': {}}}]}]}}}]}]
    info = {'SaveVideo': {'input': {'required': {
        'filename_prefix': ['STRING', {'default': 'video/test'}],
        'format': ['COMFY_DYNAMICCOMBO_V3', {'options': [
            {'key': 'mp4', 'inputs': {'required': {'codec': codec}}}]}]},
        'optional': {'codec': [codec[0], {**codec[1], 'hidden': True}]}}}}
    graph = {'1': {'class_type': 'SaveVideo', 'inputs': {
        'filename_prefix': 'V2', 'format': 'mp4', 'codec': 'h264'}}}
    expanded, schema = selected_frontend_schema(graph, info)
    assert expanded['1']['inputs']['format.codec'] == 'h264'
    assert expanded['1']['inputs']['format.codec.encoding'] == 'auto'
    assert 'codec' not in expanded['1']['inputs']
    assert schema['SaveVideo']['input']['required']['format.codec'][0] == ['auto', 'h264']
    assert graph['1']['inputs']['codec'] == 'h264'  # No mutation of old recipes.
    graph['1']['inputs']['codec'] = 'invalid'
    with pytest.raises(ValueError, match='dynamic choice'):
        selected_frontend_schema(graph, info)
