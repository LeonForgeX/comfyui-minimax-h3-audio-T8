"""Public H16 example contract and non-overlapping layout, no assets required."""
import json
from pathlib import Path

from tools.build_h16_latent_adapter_workflow import NAME, NODE, graph

ROOT = Path(__file__).resolve().parents[1]


def workflow():
    return json.loads((ROOT / 'examples/workflows/35-h3-ltx-latent' / (NAME + '.json')).read_text(encoding='utf8'))


def test_template_keeps_conversion_only_explicit_boundaries():
    data = graph()
    assert len(data) == 6
    inputs = data['2']['inputs']
    assert inputs['source_frames'] == 73 and inputs['source_fps'] == 24
    assert inputs['frame_policy'] == 'exact'
    assert inputs['source_directory'] == inputs['model_directory'] == ''
    assert (inputs['device'], inputs['precision']) == ('cpu', 'float32')
    assert data['3']['class_type'] == 'SaveLatent'
    assert data['3']['inputs']['samples'] == ['2', 0]
    assert data['5']['inputs']['source'] == ['2', 2]
    assert data['6']['inputs']['source'] == ['2', 3]


def test_public_template_has_no_local_paths_or_overlapping_nodes():
    data = workflow()
    nodes = data['nodes']
    assert len(nodes) == 7 and len(data['links']) == 5
    assert sum(n['type'] == NODE for n in nodes) == 1
    assert 'G:/' not in json.dumps(data) and 'F:/' not in json.dumps(data)
    for i, left in enumerate(nodes):
        lx, ly = left['pos']
        lw, lh = left['size']
        for right in nodes[i+1:]:
            rx, ry = right['pos']
            rw, rh = right['size']
            assert lx + lw <= rx or rx + rw <= lx or ly + lh + 30 <= ry or ry + rh + 30 <= ly
    assert data['extra']['t8_h16_status'] == 'development_exp_human_pending'
