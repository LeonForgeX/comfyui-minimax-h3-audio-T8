"""Pinned accepted controls, not fresh full-model inference or browser proof."""
import asyncio
import json
import os
from pathlib import Path

import pytest

from tools.build_fast_h3_v2_accepted_workflows import CASES, native_sampler_graph, sha
from tools.package_fast_h3_v2_candidate import ACCEPTED_V2_WORKFLOWS, PROJECT


@pytest.mark.parametrize('case', CASES)
def test_saved_six_controls_bind_exact_source_media_and_native_sockets(case):
    name, directory, _, _, media_sha = case
    evidence = Path(os.environ.get('T8_FASTH3V2_EVIDENCE_ROOT', str(PROJECT))) / 'artifacts'
    source_path = evidence / directory / 'generation/prompt.json'
    if not source_path.is_file():
        pytest.skip('Private original GPU evidence is local-only; public control hashes have separate tests')
    source = json.loads(source_path.read_text(encoding='utf8'))
    graph = native_sampler_graph(source)
    path = PROJECT / 'examples/workflows/10-speed' / (name + '.json')
    assert sha(path) == ACCEPTED_V2_WORKFLOWS[path.relative_to(PROJECT).as_posix()]
    workflow = json.loads(path.read_text(encoding='utf8'))
    artifact_root = evidence / 'fasth3-v2-six-accepted-workflows-v1'
    original = artifact_root / (name + '.json')
    original_record = json.loads((artifact_root / 'receipt.json').read_text(encoding='utf8'))['cases'][name]
    assert sha(original) == original_record['frontend_sha256']
    assert original.read_bytes().replace(b'\r\n', b'\n') == path.read_bytes()
    assert json.loads(original.read_text(encoding='utf8')) == workflow
    # Independent edge/widget audit with live current339 schema, not just SHA.
    import h3_audio_t8_pkg
    from tools.audit_progressive_workflows import audit_candidate
    from tools.build_fast_h3_v2_workflows import selected_frontend_schema
    info_path = evidence / directory / 'object-info.json'
    if not info_path.is_file():
        info_path = evidence / 'fasth3-v2-native-trained-interrupt-8s-headroom4-gpu-v2/object-info.json'
    info = json.loads(info_path.read_text(encoding='utf8'))
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    assert len(classes) == 339
    info.update({c.define_schema().node_id: c.GET_NODE_INFO_V1() for c in classes})
    selected_graph, selected_info = selected_frontend_schema(graph, info)
    assert audit_candidate(selected_graph, workflow, selected_info)['nodes'] == len(graph)
    binding = workflow['extra']['t8_bound_review']
    assert binding['media_sha256'] == media_sha
    assert binding['status'] == 'accepted_in_this_review_scope'
    assert binding['not_universal_quality_claim'] is True
    assert not any(n['type'].startswith('T8FastH3V2') for n in workflow['nodes'])
    if '13' in source:
        assert graph['11']['inputs']['noise_seed'] == source['13']['inputs']['seed']
        assert graph['12']['inputs']['model'] == source['13']['inputs']['model']
        assert graph['12']['inputs']['conditioning'] == source['13']['inputs']['positive']
        assert graph['13']['inputs']['sampler'] == source['13']['inputs']['sampler']
        assert graph['13']['inputs']['sigmas'] == source['13']['inputs']['sigmas']
        assert graph['13']['inputs']['latent_image'] == source['13']['inputs']['av_latent']
        assert graph['10']['inputs']['min_tokens'] == 0
        assert graph['9']['inputs']['length'] == (124 if 'Template' in name else 73)
    else:
        assert graph == source
        assert graph['8']['inputs']['total_duration_seconds'] == 8.0
        assert (graph['8']['inputs']['low_width'], graph['8']['inputs']['low_height']) == (256, 384)
        assert (graph['8']['inputs']['width'], graph['8']['inputs']['height']) == (512, 768)
        assert graph['8']['inputs']['model_pass1'] != graph['8']['inputs']['model_pass2']
        assert graph['8']['inputs']['context_frames'] == 22
        assert graph['8']['inputs']['render_window_frames'] == 124


@pytest.mark.parametrize('mutation', ['unknown', 'field', 'collision'])
def test_unknown_diagnostic_removal_fails_closed(mutation):
    source = {'13': {'class_type': 'T8FastH3V2SamplerProbe', 'inputs': dict(
        model=['10', 0], positive=['9', 0], av_latent=['9', 1], sampler=['10', 1], sigmas=['10', 2], seed=1)}}
    if mutation == 'unknown':
        source['13']['class_type'] = 'T8FastH3V2Unknown'
    elif mutation == 'field':
        source['13']['inputs']['new_sampler_clock'] = 1
    else:
        source['11'] = dict(class_type='RandomNoise', inputs={})
    with pytest.raises(ValueError, match='Unknown|collision'):
        native_sampler_graph(source)


def test_formal_core_has_identical_six_controls_without_migrating_old_workflows():
    location = os.environ.get('T8_FORMAL_NODE_ROOT')
    if not location:
        pytest.skip('Explicit local formal deployment required; not shipped in a source checkout')
    formal = Path(location)
    for name, expected in ACCEPTED_V2_WORKFLOWS.items():
        assert sha(PROJECT / name) == sha(formal / name) == expected


@pytest.mark.parametrize('case', CASES)
def test_public_six_controls_bind_canonical_hash_and_accepted_media_without_private_files(case):
    name, _, _, _, media_sha = case
    path = PROJECT / 'examples/workflows/10-speed' / (name + '.json')
    assert sha(path) == ACCEPTED_V2_WORKFLOWS[path.relative_to(PROJECT).as_posix()]
    workflow = json.loads(path.read_text(encoding='utf8'))
    binding = workflow['extra']['t8_bound_review']
    assert binding['media_sha256'] == media_sha
    assert binding['status'] == 'accepted_in_this_review_scope'
    assert binding['not_universal_quality_claim'] is True
    assert not any(n['type'].startswith('T8FastH3V2') for n in workflow['nodes'])
