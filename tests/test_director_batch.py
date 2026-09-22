from __future__ import annotations

import hashlib
import json
import uuid

import av
import numpy as np
import pytest

from h3_audio_t8_pkg.director_batch import (
    batch_status, capture_resources, create_batch, load_batch, retry_batch_item,
    verify_resources,
)
from h3_audio_t8_pkg.director_project import ProjectStore, atomic_json, sha


def _setup(tmp_path):
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    batch_id = str(uuid.uuid4())
    shot_ids = [str(uuid.uuid4()) for _ in range(2)]
    project = {
        "id": str(uuid.uuid4()), "revision": 5, "title": "frozen",
        "current": shot_ids[0], "assets": [],
        "doc": {"shots": [{"id": shot_ids[0], "simplePrompt": "first"},
                           {"id": shot_ids[1], "simplePrompt": "second"}]},
    }
    output = tmp_path / "output"
    output.mkdir()
    return store, batch_id, project, output


def _create(store, batch_id, project, seed, *, delivery=None):
    delivery = delivery or {"frames": 3, "width": 32, "height": 32, "seconds": 0.125}
    plans = [{"prompt": {"1": {"class_type": "Test", "inputs": {"seed": seed + i}}},
              "recipe": "test", "seed": seed + i, "report": {"ready": True},
              "sampling": {"mode": "single"}, "turbo_lora": None,
              "shot": {"time": {"delivery_trim_frames": delivery["frames"],
                                "requested_seconds": delivery["seconds"]},
                       "canvas": {"width": delivery["width"], "height": delivery["height"]}}}
             for i, _ in enumerate(project["doc"]["shots"])]
    return create_batch(store, batch_id, project, seed, plans, {"models": {}, "assets": {}})


def _write_video(path, value=25, *, audio_samples=6144):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        sound = container.add_stream("aac", rate=48000)
        sound.layout = "mono"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value + index, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        samples = av.AudioFrame.from_ndarray(
            np.zeros((1, audio_samples), dtype=np.float32), format="fltp", layout="mono")
        samples.sample_rate = 48000
        for packet in sound.encode(samples):
            container.mux(packet)
        for packet in sound.encode():
            container.mux(packet)


def _write_av(path):
    _write_video(path, 40)


def _receipt(store, batch, index, state="queued", terminal=None):
    item = batch["items"][index]
    project = {**batch["project"], "current": item["shot_id"]}
    record = {
        "state": state,
        "fingerprint": sha({"project": project, "shot_id": item["shot_id"], "seed": item["seed"]}),
        "prompt_id": str(uuid.uuid4()),
        "project_id": project["id"], "shot_id": item["shot_id"],
        "result": {"recipe": "director"},
    }
    if terminal is not None:
        record["terminal"] = terminal
    path = store.root / "requests" / f"{item['request_id']}.json"
    atomic_json(path, record)
    return path, record


def test_batch_freezes_order_seeds_and_project_and_rejects_id_reuse(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 45)
    assert [item["seed"] for item in batch["items"]] == [45, 46]
    assert _create(store, batch_id, project, 45) == batch
    project["doc"]["shots"][0]["simplePrompt"] = "edited afterwards"
    assert load_batch(store, batch_id)["project"]["doc"]["shots"][0]["simplePrompt"] == "first"
    with pytest.raises(ValueError, match="不同"):
        _create(store, batch_id, project, 45)
    status = batch_status(store, batch_id, lambda _id: None, output)
    assert status["next_index"] == 0
    assert [row["state"] for row in status["items"]] == ["not_submitted"] * 2


def test_batch_resumes_only_after_verified_media_and_detects_tampering(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    path, receipt = _receipt(store, batch, 0)
    video = output / "complete.mp4"
    _write_video(video)
    done = {"state": "success", "outputs": {"save": {"images": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    status = batch_status(store, batch_id, lambda _id: done, output)
    assert [row["state"] for row in status["items"]] == ["success", "not_submitted"]
    assert status["next_index"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["terminal"]["state"] == "success"
    assert batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output)["next_index"] == 1
    _write_video(video, 100)
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output)
    assert status["items"][0]["state"] == "needs_review"
    assert status["next_index"] == 0


@pytest.mark.parametrize("field,value", [
    ("file", "wrong.mp4"), ("frames", 0), ("width", -1),
    ("audio_stream", False), ("audio_frames", 0),
])
@pytest.mark.parametrize("keep_digest", [False, True])
def test_batch_rejects_corrupt_cached_media_fields_after_core_restart(tmp_path, field, value, keep_digest):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    path, _ = _receipt(store, batch, 0)
    video = output / "complete.mp4"
    _write_av(video)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    assert batch_status(store, batch_id, lambda _id: done, output)["items"][0]["state"] == "success"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["media_evidence"][0][field] = value
    if not keep_digest:
        receipt.pop("media_evidence_digest")
    atomic_json(path, receipt)
    state = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                         core_epoch="restarted-core")
    assert state["items"][0]["state"] == "needs_review"
    assert state["next_index"] == 0


def test_batch_verifies_mp4_with_video_and_audio_track(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    _receipt(store, batch, 0)
    video = output / "with-audio.mp4"
    _write_av(video)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    status = batch_status(store, batch_id, lambda _id: done, output)
    assert status["items"][0]["state"] == "success"
    assert status["items"][0]["media"][0]["audio_stream"] is True
    assert status["items"][0]["media"][0]["audio_frames"] > 0


@pytest.mark.parametrize("field,value", [
    ("frames", 4), ("width", 64), ("height", 64), ("seconds", 1.0),
])
def test_batch_rejects_decodable_av_that_disagrees_with_frozen_shot(tmp_path, field, value):
    store, batch_id, project, output = _setup(tmp_path)
    delivery = {"frames": 3, "width": 32, "height": 32, "seconds": 0.125}
    delivery[field] = value
    batch = _create(store, batch_id, project, 123, delivery=delivery)
    _receipt(store, batch, 0)
    video = output / "wrong-delivery.mp4"
    _write_av(video)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    status = batch_status(store, batch_id, lambda _id: done, output)
    assert status["items"][0]["state"] == "needs_review"
    assert status["next_index"] == 0


def test_batch_rejects_short_but_decodable_audio_for_short_scene(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    _receipt(store, batch, 0)
    video = output / "short-audio.mp4"
    _write_video(video, audio_samples=1024)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    status = batch_status(store, batch_id, lambda _id: done, output)
    assert status["items"][0]["state"] == "needs_review"


@pytest.mark.parametrize("field,value", [
    ("frames", 4), ("width", 64), ("height", 64), ("audio_samples", 48000),
])
def test_batch_rejects_forged_positive_cached_delivery_after_restart(tmp_path, field, value):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    receipt_path, _ = _receipt(store, batch, 0)
    video = output / "complete.mp4"
    _write_av(video)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    assert batch_status(store, batch_id, lambda _id: done, output)["items"][0]["state"] == "success"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["media_evidence"][0][field] = value
    receipt["media_evidence_digest"] = sha(receipt["media_evidence"])
    atomic_json(receipt_path, receipt)
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                          core_epoch="restarted-core")
    assert status["items"][0]["state"] == "needs_review"
    assert status["next_index"] == 0


def test_batch_upgrades_old_cached_media_receipt_after_full_redecode(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    receipt_path, _ = _receipt(store, batch, 0)
    video = output / "complete.mp4"
    _write_av(video)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": video.name, "subfolder": ""}
    ]}}}
    assert batch_status(store, batch_id, lambda _id: done, output)["items"][0]["state"] == "success"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    old_evidence = receipt["media_evidence"][0]
    old_evidence.pop("audio_samples")
    old_evidence.pop("audio_sample_rate")
    receipt["media_evidence_digest"] = sha(receipt["media_evidence"])
    atomic_json(receipt_path, receipt)
    restored = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                            core_epoch="restarted-core")
    assert restored["items"][0]["state"] == "success"
    upgraded = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert upgraded["media_evidence"][0]["audio_samples"] > 0
    assert upgraded["media_evidence"][0]["audio_sample_rate"] == 48000


def test_batch_does_not_mark_video_without_director_audio_as_complete(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    _receipt(store, batch, 0)
    path = output / "silent.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    done = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": path.name, "subfolder": ""}
    ]}}}
    status = batch_status(store, batch_id, lambda _id: done, output)
    assert status["items"][0]["state"] == "needs_review"
    assert status["next_index"] == 0


@pytest.mark.parametrize("broken", ["not-json", "[]"])
@pytest.mark.parametrize("index", [0, 1])
def test_unreadable_receipt_requires_review_without_retry_or_cross_shot_leak(tmp_path, broken, index):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 123)
    if index == 1:
        _receipt(store, batch, 0, terminal={"state": "error", "outputs": {}})
    path, _ = _receipt(store, batch, index)
    path.write_text(broken, encoding="utf-8")
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                          core_epoch="new-owned-core")
    assert status["items"][index]["state"] == "needs_review"
    assert status["items"][index]["retry_available"] is False


@pytest.mark.parametrize("state", ["submitting", "queued"])
def test_batch_lost_core_history_does_not_become_unsubmitted(tmp_path, state):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    _receipt(store, batch, 0, state=state)
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output)
    assert status["items"][0]["state"] == "unknown"
    assert status["next_index"] == 0


def test_confirmed_queue_deletion_is_durable_and_allows_explicit_retry_same_core(tmp_path):
    from h3_audio_t8_pkg.director_routes import _record_deleted_queue_receipt

    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    path, receipt = _receipt(store, batch, 0)
    assert _record_deleted_queue_receipt(store, receipt["prompt_id"]) is True
    assert json.loads(path.read_text(encoding="utf-8"))["terminal"]["state"] == "cancelled"
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                          core_epoch="same-core")
    assert status["items"][0]["state"] == "cancelled"
    assert status["items"][0]["retry_available"] is True
    assert status["next_index"] == 0
    retry = retry_batch_item(store, batch_id, 0)
    assert retry["items"][0]["attempt"] == 1
    assert path.is_file(), "the cancelled attempt remains auditable"


def test_batch_bad_receipt_and_escape_media_require_review(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    path, receipt = _receipt(store, batch, 0)
    receipt["fingerprint"] = "wrong"
    atomic_json(path, receipt)
    status = batch_status(store, batch_id, lambda _id: {"state": "success"}, output)
    assert status["items"][0]["state"] == "needs_review"
    receipt["fingerprint"] = sha({"project": {**batch["project"], "current": batch["items"][0]["shot_id"]}, "seed": 77, "shot_id": batch["items"][0]["shot_id"]})
    atomic_json(path, receipt)
    dangerous = {"state": "success", "outputs": {"save": {"videos": [
        {"type": "output", "filename": "../../other.mp4", "subfolder": ""}
    ]}}}
    assert batch_status(store, batch_id, lambda _id: dangerous, output)["items"][0]["state"] == "needs_review"


def test_batch_corrupt_request_identity_cannot_resubmit_finished_shot(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    file = store.root / "batches" / f"{batch_id}.json"
    batch["items"][0]["request_id"] = str(uuid.uuid4())
    atomic_json(file, batch)
    with pytest.raises(ValueError, match="请求身份"):
        load_batch(store, batch_id)


def test_unknown_from_old_core_can_explicitly_retry_only_first_incomplete_item(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    path, receipt = _receipt(store, batch, 0)
    receipt["core_epoch"] = "previous-owned-core"
    atomic_json(path, receipt)
    same = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                        core_epoch="previous-owned-core")
    assert same["items"][0]["retry_available"] is False
    restarted = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output,
                             core_epoch="new-owned-core")
    assert restarted["items"][0]["retry_available"] is True
    revised = retry_batch_item(store, batch_id, restarted["next_index"])
    assert revised["items"][0]["attempt"] == 1
    assert revised["items"][0]["previous_request_ids"] == [batch["items"][0]["request_id"]]
    assert path.is_file(), "old receipt remains available for investigation"
    assert revised["items"][1]["request_id"] == batch["items"][1]["request_id"]
    resumed = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output)
    assert [item["state"] for item in resumed["items"]] == ["not_submitted", "not_submitted"]


@pytest.mark.parametrize("bad", ["broken", {"state": "success", "outputs": []}])
def test_corrupt_terminal_record_needs_review_instead_of_breaking_batch(tmp_path, bad):
    store, batch_id, project, output = _setup(tmp_path)
    batch = _create(store, batch_id, project, 77)
    _receipt(store, batch, 0, terminal=bad)
    status = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output)
    assert status["items"][0]["state"] == "needs_review"
    assert status["items"][0]["retry_available"] is False


def test_batch_freezes_compiled_graph_and_model_bytes(tmp_path):
    store, batch_id, project, output = _setup(tmp_path)
    project["doc"]["sharedRefs"] = []
    for shot in project["doc"]["shots"]:
        shot.update(tray=[], refs=[], first=None, last=None, audio=None, selected=None)
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"exact-weights")
    plans = [{"prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}}},
              "recipe": "test", "seed": i, "sampling": {}, "report": {}}
             for i in range(2)]
    def resolver(folder, name):
        return weight
    resources = capture_resources(store, project, plans, resolver)
    batch = create_batch(store, batch_id, project, 9, plans, resources)
    assert len(batch["resources"]["models"]) == 1
    verify_resources(store, load_batch(store, batch_id)["resources"], resolver)
    weight.write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="模型已变化"):
        verify_resources(store, batch["resources"], resolver)
    path = store.root / "batches" / f"{batch_id}.json"
    batch["items"][0]["built"]["prompt"]["1"]["inputs"]["unet_name"] = "wrong.safetensors"
    atomic_json(path, batch)
    with pytest.raises(ValueError, match="快照身份"):
        load_batch(store, batch_id)


def test_batch_freezes_dedicated_hyperflow_file_not_generic_lora(tmp_path):
    store, _, project, _ = _setup(tmp_path)
    project["doc"]["sharedRefs"] = []
    for shot in project["doc"]["shots"]:
        shot.update(tray=[], refs=[], first=None, last=None, audio=None, selected=None)
    weight = tmp_path / "hyperflow.safetensors"
    weight.write_bytes(b"original-hf-artifact")
    plans = [{"prompt": {"2": {"class_type": "MiniMaxH3HyperFlowLoaderT8Advanced",
                               "inputs": {"hyperflow_file": "hyperflow/hyperflow.safetensors", "model": ["1", 0]}}}}
             for _ in project["doc"]["shots"]]
    def resolver(folder, name):
        assert (folder, name) == ("hyperflow_selection", "hyperflow/hyperflow.safetensors")
        return weight
    resources = capture_resources(store, project, plans, resolver)
    assert len(resources["models"]) == 1
    frozen = resources["models"]["hyperflow_selection:hyperflow/hyperflow.safetensors"]
    assert frozen["sha256"] == hashlib.sha256(b"original-hf-artifact").hexdigest()
    assert frozen["size"] == len(b"original-hf-artifact")
    batch = create_batch(store, str(uuid.uuid4()), project, 9, plans, resources)
    assert load_batch(store, batch["id"])["resources"]["models"] == resources["models"]
    verify_resources(store, resources, resolver)
    weight.write_bytes(b"replaced-hf-artifact")
    with pytest.raises(ValueError, match="模型已变化"):
        verify_resources(store, resources, resolver)
