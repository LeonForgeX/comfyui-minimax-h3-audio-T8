from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import h3_audio_t8_pkg.nodes_video_outpaint_candidates as nodes
import h3_audio_t8_pkg.nodes_video_outpaint as stages
from h3_audio_t8_pkg.video_outpaint_candidate_archive import (
    save_candidate_archive, load_candidate_archive, load_selection_archive, save_selection_archive,
)
from h3_audio_t8_pkg.video_outpaint_candidates import select_candidate
from test_video_outpaint_compose import CombinedVAE
from test_video_outpaint_execution import TinyClip, _model
from test_video_outpaint_media import _clip
from test_video_outpaint_sampling import _backend


@pytest.fixture
def archived(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path / "outputs"))
    generate = nodes.sample_verified_first_candidate
    monkeypatch.setattr(nodes, "sample_verified_first_candidate", lambda **kw: generate(**kw, sample_function=_backend))
    monkeypatch.setattr(nodes.ui, "PreviewImage", lambda *a, **kw: {"images": []})
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=90, sound=False)
    plan = stages.MiniMaxH3VideoOutpaintPlanT8.execute(source, "custom", 32, 0, 32, 0,
        0.5, 0.5, 0.5, "73", "[]").result[0]
    prepared = stages.MiniMaxH3VideoOutpaintPrepareT8.execute(plan, TinyClip(), CombinedVAE(),
        "scene", "[]", "archived", 0, 64, False).result[0]
    generated = nodes.MiniMaxH3VideoOutpaintCandidateT8.execute(_model(), prepared, CombinedVAE(),
        "candidate_a", 42, 20, False, True, source_mode="preserve_source")
    return generated.result[0], generated.result[1]


def _load(handle):
    return load_candidate_archive(handle["prepared"], handle["cache_root"], handle["archive_id"])


def test_archive_reopens_new_objects_exact_png_without_implicit_selection(archived):
    handle, image = archived
    before = handle["windows"].path.read_bytes()
    restored, cached_image = _load(handle)
    assert restored["windows"] is not handle["windows"] and torch.equal(cached_image, image)
    assert restored["candidate"] == handle["candidate"] and "selection" not in restored
    assert handle["windows"].path.read_bytes() == before
    assert save_candidate_archive(handle, image) == handle["archive_id"]


@pytest.mark.parametrize("identifier", ["", "../escape", "a"*63, "A"*64, None, False])
def test_load_requires_explicit_sha_not_path_or_latest(tmp_path, identifier):
    with pytest.raises(ValueError, match="archive id"):
        load_candidate_archive({}, tmp_path / "absent", identifier)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("asset", ["metadata", "png", "window"])
def test_corrupt_archive_or_sample_is_rejected_without_replacement(archived, asset):
    handle, _ = archived
    root = handle["cache_root"]
    metadata = root / f"candidate-{handle['archive_id']}.json"
    record = json.loads(metadata.read_text())
    target = {"metadata": metadata, "png": root / f"preview-{record['png_sha256']}.png",
              "window": root / f"window-{handle['candidate']['prefix'][0]['sha256']}.safetensors"}[asset]
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        _load(handle)
    assert target.read_bytes() == b"corrupt"


def test_missing_sampling_manifest_does_not_create_replacement(archived):
    handle, _ = archived
    path = handle["windows"].path
    path.rename(path.with_suffix(".saved"))
    with pytest.raises(FileNotFoundError, match="no replacement"):
        _load(handle)
    assert not path.exists()


def test_wrong_image_cannot_be_saved_under_existing_preview(archived):
    handle, image = archived
    changed = image.clone()
    changed[0, 0, 0] = 1 - changed[0, 0, 0]
    with pytest.raises(ValueError, match="image differs"):
        save_candidate_archive(handle, changed)


def test_selection_pins_saved_preview_and_settings(archived):
    handle, _ = archived
    selected = {**handle, "selection": select_candidate(handle["candidate"], handle["windows"])}
    selection_id = save_selection_archive(selected)
    recovered, _ = load_selection_archive(handle["prepared"], handle["cache_root"], selection_id)
    assert recovered["selection"] == selected["selection"]
    broken = {**selected, "preview_report": deepcopy(selected["preview_report"])}
    broken["preview_report"]["color_settings"]["enabled"] = False
    with pytest.raises(ValueError, match="no longer matches"):
        save_selection_archive(broken)


def test_new_process_restores_selection_after_continuation_without_loading_models(archived, tmp_path):
    from h3_audio_t8_pkg.video_outpaint_candidate_execution import continue_verified_candidate
    handle, image = archived
    prepared = handle["prepared"]
    selected = {**handle, "selection": select_candidate(handle["candidate"], handle["windows"])}
    selection_id = save_selection_archive(selected)
    continue_verified_candidate(model=_model(), conditioning=prepared["conditioning"], source_store=prepared["source"],
        audio=prepared["audio"], cache_root=handle["cache_root"], selection=selected["selection"],
        sample_function=_backend, **handle["settings"])
    request = {"prepared_root": str(prepared["root"]), "plan": prepared["plan"],
        "inspection": prepared["inspection"], "video_vae_sha256": prepared["source"].identity["video_vae_sha256"],
        "audio_identity": prepared["audio"].store.identity, "selection_id": selection_id}
    path = tmp_path / "reload.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("outpaint_archive_reload_worker.py")), str(path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    assert receipt["committed"] == 2 and receipt["image_shape"] == list(image.shape)
    assert receipt["selection_id"] == selection_id and receipt["candidate_id"] == handle["archive_id"]
    assert not receipt["model_loaded"] and not receipt["sampler_called"]


def test_cancelled_archive_load_releases_lease_and_does_not_change_record(archived):
    handle, image = archived
    path = handle["cache_root"] / f"candidate-{handle['archive_id']}.json"
    before = path.read_bytes()

    def cancel():
        raise RuntimeError("cancel archive load")

    with pytest.raises(RuntimeError, match="cancel archive load"):
        load_candidate_archive(handle["prepared"], handle["cache_root"], handle["archive_id"], interrupt_check=cancel)
    assert path.read_bytes() == before
    assert torch.equal(_load(handle)[1], image)
