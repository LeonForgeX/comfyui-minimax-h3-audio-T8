"""Audit the actual shipped JSONs, not only freshly generated artifact copies."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from comfy_extras.nodes_video import LoadVideo

from h3_audio_t8_pkg import nodes_topaz
from tools.audit_progressive_workflows import audit_candidate


@pytest.mark.parametrize('name', ['Environment', 'Video'])
def test_delivered_topaz_schema_values_and_edges(name):
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root / 'examples/workflows/31-topaz' /
        f'2026-09-11_H3_Topaz_{name}_EXP.json').read_text(encoding='utf8'))
    classes = [*nodes_topaz.TOPAZ_NODE_CLASSES, LoadVideo]
    info = json.loads(json.dumps({cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}))
    graph = {'1': {'class_type': 'MiniMaxH3TopazEnvironmentEXPT8', 'inputs': {
        'install_directory': '', 'model_definitions_directory': '', 'model_data_directory': ''}}}
    if name == 'Video':
        graph.update({
            '2': {'class_type': 'LoadVideo', 'inputs': {'file': 'source.mkv'}},
            '3': {'class_type': 'MiniMaxH3TopazVideoEXPT8', 'inputs': {
                'topaz_runtime': ['1', 0], 'source_video': ['2', 0], 'model_id': 'iris-3',
                'scale': '2x', 'vram_fraction': .8, 'parameters_json': '{}',
                'size_mode': 'scale', 'target_width': 0, 'target_height': 0}}})
    result = audit_candidate(graph, workflow, info)
    assert result['edges'] == (2 if name == 'Video' else 0)
    if name == 'Video':
        for offset, value in ((4, 'target_dimensions'), (5, 1536), (6, 768)):
            broken = deepcopy(workflow)
            node = next(n for n in broken['nodes'] if n['type'] == 'MiniMaxH3TopazVideoEXPT8')
            node['widgets_values'][offset] = value
            with pytest.raises(ValueError):
                audit_candidate(graph, broken, info)


def test_delivered_upscale_then_interpolation_has_real_two_stage_edges():
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root / 'examples/workflows/31-topaz' /
        '2026-09-14_H3_Topaz_Upscale_Then_Interpolation_EXP.json').read_text(encoding='utf8'))
    classes = [*nodes_topaz.TOPAZ_NODE_CLASSES, LoadVideo]
    info = json.loads(json.dumps({cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}))
    graph = {
        '1': {'class_type': 'MiniMaxH3TopazEnvironmentEXPT8', 'inputs': {
            'install_directory': '', 'model_definitions_directory': '', 'model_data_directory': ''}},
        '2': {'class_type': 'LoadVideo', 'inputs': {'file': 'source.mp4'}},
        '3': {'class_type': 'MiniMaxH3TopazVideoEXPT8', 'inputs': {
            'topaz_runtime': ['1', 0], 'source_video': ['2', 0], 'model_id': 'iris-3',
            'scale': '2x', 'vram_fraction': .8, 'parameters_json': '{}'}},
        '4': {'class_type': 'MiniMaxH3TopazFrameInterpolationEXPT8', 'inputs': {
            'topaz_runtime': ['1', 0], 'source_video': ['3', 0], 'model_id': 'apf-2',
            'multiplier': '2x'}},
    }
    assert audit_candidate(graph, workflow, info)['edges'] == 4
    regular = next(node for node in workflow['nodes'] if node['id'] == 3)
    interpolator = next(node for node in workflow['nodes'] if node['id'] == 4)
    assert regular['outputs'][0]['links'] == [4]
    assert interpolator['inputs'][1]['link'] == 4
