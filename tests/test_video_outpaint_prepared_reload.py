from __future__ import annotations

import json

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_prepared_reload import load_prepared_outpaint, _ExistingSourceStore, _ExistingAudioStore
from h3_audio_t8_pkg.nodes_video_outpaint_reload import MiniMaxH3VideoOutpaintLoadPreparedT8
from h3_audio_t8_pkg.video_outpaint_candidate_archive import load_candidate_archive
from test_video_outpaint_candidate_archive import archived as archived_fixture

archived = archived_fixture


def _reload(handle, **kwargs):
    prepared = handle["prepared"]
    return load_prepared_outpaint({"plan": prepared["plan"], "inspection": prepared["inspection"]}, prepared["root"], **kwargs)


def test_loaded_providers_are_new_readonly_objects_with_identical_values(archived):
    handle, image = archived
    old = handle["prepared"]
    loaded, report = _reload(handle)
    assert loaded["source"] is not old["source"] and loaded["audio"] is not old["audio"]
    assert report["source_chunks_verified"] == len(old["source"].expected)
    assert not report["vae_or_clip_called"] and not report["cache_manifests_rewritten"]
    for shot, start, stop in old["source"].expected:
        assert torch.equal(loaded["source"].read_range(shot, start, stop), old["source"].read_range(shot, start, stop))
    assert loaded["conditioning"].verify() == old["conditioning"].verify()
    assert loaded["audio"].verify() == old["audio"].verify()
    restored, picture = load_candidate_archive(loaded, handle["cache_root"], handle["archive_id"])
    assert torch.equal(picture, image) and restored["prepared"] is loaded
    assert isinstance(loaded["source"], _ExistingSourceStore)
    assert isinstance(loaded["audio"].store, _ExistingAudioStore)


@pytest.mark.parametrize("kind", ["source", "audio", "text"])
def test_missing_cache_manifest_is_never_recreated(archived, kind):
    handle, _ = archived
    prepared = handle["prepared"]
    path = {"source": prepared["source"].path, "audio": prepared["audio"].store.path,
            "text": prepared["conditioning"].path}[kind]
    path.rename(path.with_suffix(".saved"))
    with pytest.raises(FileNotFoundError):
        _reload(handle)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["source", "text"])
def test_corrupted_source_or_text_asset_rejected(archived, kind):
    handle, _ = archived
    prepared = handle["prepared"]
    root, pattern = (prepared["source"].root, "video-*.safetensors") if kind == "source" else (
        prepared["conditioning"].root, "conditioning-*.safetensors")
    target = next(root.glob(pattern))
    target.write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated|missing"):
        _reload(handle)
    assert target.read_bytes() == b"bad"


def test_mid_reload_manifest_change_rejected(archived):
    handle, _ = archived
    path = handle["prepared"]["conditioning"].path

    def mutate(report):
        if report["chunks_verified"] == report["total_chunks"]:
            path.write_bytes(b"bad")

    with pytest.raises(ValueError):
        _reload(handle, progress=mutate)


def test_cancelled_reload_can_retry_without_changing_cache(archived):
    handle, _ = archived
    before = handle["prepared"]["source"].path.read_bytes()

    def cancel():
        raise RuntimeError("cancel reload")

    with pytest.raises(RuntimeError, match="cancel reload"):
        _reload(handle, interrupt_check=cancel)
    assert handle["prepared"]["source"].path.read_bytes() == before
    assert _reload(handle)[1]["saved_conditioning_reused"]


def test_reload_node_has_no_model_inputs_and_returns_prepared_handle(archived):
    handle, _ = archived
    schema = MiniMaxH3VideoOutpaintLoadPreparedT8.define_schema()
    assert [i.id for i in schema.inputs] == ["plan", "run_name"]
    prepared = handle["prepared"]
    result = MiniMaxH3VideoOutpaintLoadPreparedT8.execute(
        {"plan": prepared["plan"], "inspection": prepared["inspection"]}, "archived")
    assert result.result[0]["root"] == prepared["root"]
    assert not json.loads(result.result[1])["models_loaded_by_reload"]


def test_readonly_store_constructors_refuse_missing_manifest_creation(tmp_path):
    from test_video_outpaint_sampling import _plan
    plan = _plan()
    with pytest.raises(FileNotFoundError, match="cannot create"):
        _ExistingSourceStore(tmp_path / "source", plan, video_vae_sha256="b"*64)
    assert not (tmp_path / "source/outpaint_source_latents.json").exists()
    with pytest.raises(FileNotFoundError, match="cannot create"):
        _ExistingAudioStore(tmp_path / "audio", plan, audio_vae_sha256=None, pcm_sha256=None,
                            stream_position=None, encoding_device=None)
    assert not (tmp_path / "audio/outpaint_source_audio.json").exists()


def test_audio_asset_validation_checks_cancel_before_each_asset():
    store = object.__new__(_ExistingAudioStore)
    store.snapshot = lambda: {"chunks": [1, 2, 3]}
    loaded = []
    store._load = loaded.append

    def checkpoint():
        if loaded:
            raise RuntimeError("cancel second audio asset")

    store._reload_interrupt = checkpoint
    with pytest.raises(RuntimeError, match="second audio asset"):
        store.verify_assets()
    assert loaded == [1]


def test_conditioning_restore_checks_cancel_before_loading_shot_tensor(archived):
    from h3_audio_t8_pkg.video_outpaint_prepared_reload import _ExistingConditioningProvider
    handle, _ = archived
    prepared = handle["prepared"]

    def cancel():
        raise RuntimeError("cancel shot tensor")

    with pytest.raises(RuntimeError, match="cancel shot tensor"):
        _ExistingConditioningProvider(prepared["conditioning"].root, prepared["plan"], interrupt_check=cancel)
