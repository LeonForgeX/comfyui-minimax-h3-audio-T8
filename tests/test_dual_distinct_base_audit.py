import hashlib
import json

import pytest

from tools.audit_dual_model_pilot import audit_distinct_base


def fixture(root, drift=None):
    source = root / 'second.safetensors'
    source.write_bytes(b'identity fixture only, never loaded as weights')
    graph = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'first.safetensors'}},
             '28': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'second.safetensors'}},
             '2': {'inputs': {'model': ['1', 0]}}, '3': {'inputs': {'model': ['28', 0]}},
             '21': {'inputs': {'model': ['2', 0]}}, '22': {'inputs': {'model': ['3', 0]}},
             '8': {'inputs': {'model_pass1': ['21', 0], 'model_pass2': ['22', 0]}}}
    if drift == 'same_loader':
        graph['28']['inputs']['unet_name'] = 'first.safetensors'
    elif drift == 'same_branch':
        graph['3']['inputs']['model'] = ['1', 0]
    elif drift == 'swapped_stages':
        graph['8']['inputs']['model_pass2'] = ['21', 0]
    (root / 'generation').mkdir()
    (root / 'output').mkdir()
    (root / 'generation/prompt.json').write_text(json.dumps(graph))
    (root / 'expected.json').write_text(json.dumps({'pilot_graphs': {'dual_short': graph}}))
    ids = 'a' * 16 + ':' + ('a' if drift == 'same_identity' else 'b') * 16
    (root / 'output/manifest.json').write_text(json.dumps({'segments': [{'model_id': ids}]}))
    terminal = {'second_base_file': {'path': str(source), 'bytes': source.stat().st_size,
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    if drift == 'weight_bytes':
        source.write_bytes(b'changed')
    return terminal


def test_separate_loader_content_bindings(tmp_path):
    report = audit_distinct_base(tmp_path, fixture(tmp_path))
    assert report['separate_UNET_loader_and_LoRA_bindings']


@pytest.mark.parametrize('drift', ['same_loader', 'same_branch', 'swapped_stages', 'same_identity', 'weight_bytes'])
def test_false_distinct_base_claim_rejected(tmp_path, drift):
    with pytest.raises(ValueError):
        audit_distinct_base(tmp_path, fixture(tmp_path, drift))
