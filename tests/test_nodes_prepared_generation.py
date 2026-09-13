"""Actual Comfy node schema and lazy handoff tests, no inference claim."""
import asyncio
import json
import math
from pathlib import Path

import pytest
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import nodes_prepared_generation as nodes
from h3_audio_t8_pkg import prepared_generation_runtime as runtime
from h3_audio_t8_pkg.nodes import comfy_entrypoint
from tools.build_prepared_workflows import build_prompt, build_workflow
from tools.audit_progressive_workflows import audit_candidate
from tests.test_prepared_generation_contract import bundle  # noqa: F401


def test_nodes_registered_once_and_lazy_schema():
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    for cls in nodes.PREPARED_GENERATION_NODE_CLASSES:
        assert classes.count(cls) == 1
        assert cls.define_schema().is_experimental
        assert math.isnan(cls.fingerprint_inputs())
    assert nodes.MiniMaxH3PreparedVideoEXPT8.define_schema().is_output_node


def test_bundle_node_reads_only_structure(bundle, tmp_path, monkeypatch):  # noqa: F811
    file = tmp_path / 'bundle.json'
    file.write_text(json.dumps(bundle))
    monkeypatch.setattr(runtime, 'verify_assets', lambda *args: pytest.fail('Load node hashed models'))
    result = nodes.MiniMaxH3PreparedGenerationBundleEXPT8.execute(str(file))
    assert result.result[0] == bundle
    report = json.loads(result.result[1])
    assert report['status'] == 'structure_checked_only'
    assert not report['models_loaded'] and not report['content_hashes_verified']


@pytest.mark.parametrize('path', ['', 'not-existing-bundle.json'])
def test_missing_bundle_is_not_default_current_directory(path):
    with pytest.raises((ValueError, FileNotFoundError)):
        nodes.MiniMaxH3PreparedGenerationBundleEXPT8.execute(path)


def test_video_node_forwards_settings_and_previews_original_file(bundle, tmp_path, monkeypatch):  # noqa: F811
    observed = {}
    movie = tmp_path / 'MiniMaxH3-Prepared/fixture/result.mp4'
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b'synthetic video fixture, not media validation')
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    def run(value, **settings):
        observed.update(settings)
        assert value == bundle and value is not bundle
        return movie, {'status': 'synthetic_handoff_only'}
    monkeypatch.setattr(runtime, 'run_prepared', run)
    result = nodes.MiniMaxH3PreparedVideoEXPT8.execute(bundle, 8301, 'trial', True, str(tmp_path / 'shared.lock'))
    assert observed['noise_seed'] == 8301 and observed['chain_id'] == 'trial' and observed['resume_existing']
    assert observed['lease_path'] == tmp_path / 'shared.lock'
    assert isinstance(result.result[0], InputImpl.VideoFromFile)
    assert result.result[1] == str(movie) and result.ui is not None


def test_lease_path_is_required_before_execution(bundle, monkeypatch):  # noqa: F811
    monkeypatch.setattr(runtime, 'run_prepared', lambda *a, **kw: pytest.fail('Worker started without shared lease'))
    with pytest.raises(ValueError, match='absolute'):
        nodes.MiniMaxH3PreparedVideoEXPT8.execute(bundle, 8301, 'trial', True, '')


@pytest.mark.parametrize('route', ['tao5s', 'ltx_refine'])
def test_generation_workflow_is_not_fixed_clip_playback(route):
    info = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in nodes.PREPARED_GENERATION_NODE_CLASSES}
    info = json.loads(json.dumps(info))
    graph = build_prompt(route=route)
    workflow = build_workflow(info, route=route)
    result = audit_candidate(graph, workflow, info)
    assert result['nodes'] == 2 and result['edges'] == 1
    assert graph['2']['class_type'] == 'MiniMaxH3PreparedVideoEXPT8'
    assert not any(n['type'] in ('LoadVideo', 'SaveVideo') for n in workflow['nodes'])
    assert not graph['1']['inputs']['prepared_bundle_path']
    saved = Path(__file__).parents[1] / 'examples/workflows/32-prepared-generation' / f'2026-09-13_H3_Prepared_{route}_EXP.json'
    assert json.loads(saved.read_text(encoding='utf8')) == workflow
