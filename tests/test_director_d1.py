from copy import deepcopy
import json
import uuid

import pytest
from PIL import Image

from h3_audio_t8_pkg.director_project import (
    ProjectConflict,
    ProjectStore,
    compile_project,
    new_project,
    validate_project,
)
from h3_audio_t8_pkg.director_routes import export_preflight_workflow


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "user", tmp_path / "input")


def asset(store, kind="image", name="same.png", **values):
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (512, 768), (20, 100, 60)).save(path)
    result = store.register_asset(path, aid, name)
    return {**result, "kind": kind, **values}


def test_actual_same_name_assets_persist_restart_and_cas(store):
    a, b = asset(store), asset(store)
    assert a["id"] != b["id"] and a["server_path"] != b["server_path"]
    p = new_project()
    p["assets"] = [a, b]
    p["doc"]["shots"][0]["simplePrompt"] = "原稿\r\n中文\t🙂 𠀀"
    saved = store.save(p, 0)
    restarted = ProjectStore(store.root.parent, store.input_root)
    assert restarted.load(p["id"]) == saved
    assert restarted.list()[0]["revision"] == 1
    assert saved["doc"] == p["doc"]
    with pytest.raises(ProjectConflict):
        restarted.save(p, 0)
    updated = deepcopy(saved)
    updated["title"] = "new"
    assert store.save(updated, 1)["revision"] == 2
    assert restarted.asset(a["id"], verify=True)["sha256"] == a["sha256"]


def test_hyperflow_global_and_local_sampling_survive_save_reload_and_project_json(store):
    p = new_project()
    first = p["doc"]["shots"][0]
    second = deepcopy(first)
    second["id"] = str(uuid.uuid4())
    second["name"] = "独立采样镜头"
    second["samplingInherit"] = False
    second["sampling"] = {
        "mode": "hyperflow", "variant": "upscale4plus4",
        "hyperflow_file": "hyperflow/weight.safetensors", "output_mp": 0.5,
        "upscaler": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "low_loras": [{"id": "local-low", "name": "portrait.safetensors", "strength": 0.15, "enabled": True}],
        "high_loras": [{"id": "local-high", "name": "motion.safetensors", "strength": 0.25, "enabled": True}],
    }
    p["doc"]["shots"].append(second)
    p["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "upscale8plus4",
        "hyperflow_file": "hyperflow/weight.safetensors", "output_mp": 0.4,
        "upscaler": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "low_loras": [
            {"id": "global-a", "name": "portrait.safetensors", "strength": 0.1, "enabled": True},
            {"id": "global-b", "name": "motion.safetensors", "strength": 0.2, "enabled": True},
        ],
        "high_loras": [{"id": "global-high", "name": "motion.safetensors", "strength": 0.3, "enabled": True}],
    }
    saved = store.save(p, 0)
    reopened = ProjectStore(store.root.parent, store.input_root).load(p["id"])
    assert reopened == saved
    assert reopened["doc"]["sampling"] == p["doc"]["sampling"]
    assert reopened["doc"]["shots"][0]["samplingInherit"] is True
    assert reopened["doc"]["shots"][1]["samplingInherit"] is False
    assert reopened["doc"]["shots"][1]["sampling"] == second["sampling"]

    # The browser's exported project JSON is the envelope, while its separate
    # API snapshot is deliberately only a D1 CPU preflight graph.
    imported = validate_project(json.loads(json.dumps(reopened, ensure_ascii=False)))
    assert imported == reopened
    exported = export_preflight_workflow(imported, second["id"])
    embedded = json.loads(exported["api_snapshot"]["1"]["inputs"]["project_json"])
    assert embedded["doc"]["sampling"] == p["doc"]["sampling"]
    assert embedded["doc"]["shots"][1]["sampling"] == second["sampling"]


def test_missing_and_corrupt_assets_are_not_silently_successful(store):
    a = asset(store)
    p = new_project()
    p["assets"] = [a]
    p["doc"]["shots"][0]["tray"] = [a["id"]]
    path = store.input_root / a["server_path"]
    path.write_bytes(b"broken")
    assert not compile_project(p, store)["ready"]
    assert any("字节改变" in e["message"] for e in compile_project(p, store)["errors"])
    path.unlink()
    with pytest.raises(ValueError, match="缺失"):
        store.save(p, 0)


def test_single_shot_scope_keeps_integrity_checks_but_isolates_other_assets(store):
    good, missing = asset(store), asset(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", refs=[good["id"]], tray=[good["id"]], simplePrompt="@image1 微笑")
    sibling = deepcopy(shot)
    sibling.update(id=str(uuid.uuid4()), refs=[missing["id"]], tray=[missing["id"]])
    project["doc"]["shots"].append(sibling)
    project["assets"] = [good, missing]
    (store.input_root / missing["server_path"]).unlink()
    assert not compile_project(project, store)["ready"]
    assert compile_project(project, store, shot_id=shot["id"])["ready"]
    assert not compile_project(project, store, shot_id=sibling["id"])["ready"]
    (store.input_root / good["server_path"]).write_bytes(b"changed")
    report = compile_project(project, store, shot_id=shot["id"])
    assert not report["ready"]
    assert any("字节改变" in error["message"] for error in report["errors"])


def test_missing_retired_library_asset_does_not_break_explicit_reconnection(store):
    old, replacement = asset(store), asset(store)
    (store.input_root / old["server_path"]).unlink()
    p = new_project()
    p["assets"] = [old, replacement]
    p["doc"]["shots"][0].update(
        mode="first",
        first=replacement["id"],
        tray=[replacement["id"]],
        simplePrompt="@image1",
    )
    report = compile_project(p, store)
    assert report["ready"] and any(
        x.get("asset_id") == old["id"] for x in report["warnings"]
    )
    saved = store.save(p, 0)
    assert saved["assets"][0]["missing"] is True
    assert compile_project(store.load(p["id"]), store)["ready"]


def test_reconnected_imported_ghost_preserves_only_nonexecutable_library_identity(
    store,
):
    replacement = asset(store)
    ghost = {
        "id": str(uuid.uuid4()),
        "name": "old.png",
        "kind": "image",
        "server_path": "../../private",
        "sha256": "fake",
    }
    p = new_project()
    p["assets"] = [ghost, replacement]
    p["doc"]["shots"][0].update(
        mode="first",
        first=replacement["id"],
        tray=[replacement["id"]],
        simplePrompt="@image1",
    )
    saved = store.save(p, 0)
    assert saved["assets"][0]["id"] == ghost["id"] and saved["assets"][0]["retired"]
    assert (
        saved["assets"][0]["server_path"] is None
        and saved["assets"][0]["sha256"] is None
    )
    assert compile_project(saved, store)["ready"]


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(id="../escape"),
        lambda p: p.update(version=99),
        lambda p: p.update(version=True),
        lambda p: p.update(revision=True),
        lambda p: p["doc"].update(shots=[]),
        lambda p: p["doc"]["shots"][0].update(duration=float("nan")),
        lambda p: p["doc"]["shots"].append(deepcopy(p["doc"]["shots"][0])),
    ],
)
def test_bad_contracts_rejected(change):
    p = new_project()
    change(p)
    with pytest.raises(ValueError):
        validate_project(p)


def test_explicit_migration_preserves_unknown_fields_and_prose():
    p = new_project()
    p["version"] = 0
    p["opaque_notes"] = "a\r\nb"
    migrated = validate_project(p)
    assert migrated["version"] == 2 and migrated["opaque_notes"] == "a\r\nb"
    with pytest.raises(ValueError, match="未知工作流"):
        validate_project({"nodes": [], "links": []})


def test_media_order_first_last_video_soundtrack_independent_audio(store):
    first, last, ref = asset(store), asset(store), asset(store)
    video = asset(store, "video", duration=3, has_audio=True)
    audio = asset(store, "audio", duration=2)
    p = new_project()
    p["assets"] = [first, last, ref, video, audio]
    s = p["doc"]["shots"][0]
    s.update(
        mode="ends",
        first=first["id"],
        last=last["id"],
        refs=[ref["id"], video["id"], audio["id"]],
        tray=[audio["id"], ref["id"], video["id"], last["id"], first["id"]],
        ratio="2:3",
        ownRatio="2:3",
        simplePrompt="@image3 @image2 @image1 @video1 @audio1\r\n原文  空格",
    )
    p["doc"]["ratio"] = "2:3"
    result = compile_project(p)
    assert result["ready"]
    out = result["shots"][0]
    assert (
        out["prompt"]
        == "<Picture 1> <Picture 2> <Picture 3> <Video 1> <Audio 2>\r\n原文  空格"
    )
    assert [x["role"] for x in out["media_map"]] == [
        "first_frame",
        "last_frame",
        "ref_image",
        "ref_video",
        "ref_video_audio",
        "ref_audio",
    ]
    assert out["source_drafts"]["simple"] == s["simplePrompt"]
    assert p["doc"]["shots"][0]["simplePrompt"].startswith("@")


def test_alias_manifest_prevents_retargeting_and_missing_is_explicit(store):
    a = asset(store)
    p = new_project()
    p["assets"] = [a]
    s = p["doc"]["shots"][0]
    s.update(mode="refs", tray=[a["id"]], refs=[a["id"]], simplePrompt="@image1")
    p["aliasMaps"] = {s["id"]: {"@image1": str(uuid.uuid4())}}
    assert not compile_project(p)["ready"]
    p.pop("aliasMaps")
    s["refs"] = []
    s["mode"] = "text"
    assert not compile_project(p)[
        "ready"
    ]  # Visible but unassigned does not imply a native slot.
    s["simplePrompt"] = "@missing_image1"
    assert any("missing" in e["message"] for e in compile_project(p)["errors"])


@pytest.mark.parametrize(
    "sound,recipe,mode,delivery",
    [
        (
            "record",
            "avatar_single_segment_progressive_lock_source",
            "lock_source",
            "original_selected_recording",
        ),
        ("voice", "native_ref_voice_stock20", "native", "generated"),
    ],
)
def test_audio_intents_never_share_drive_or_delivery_slot(
    store, sound, recipe, mode, delivery
):
    a = asset(store, "audio", duration=2)
    p = new_project()
    p["assets"] = [a]
    s = p["doc"]["shots"][0]
    s.update(
        sound=sound, audio=a["id"], end=2, simplePrompt="@audio1 新台词", tray=[a["id"]]
    )
    if sound == "record":
        picture = asset(store)
        p["assets"].append(picture)
        s.update(mode="first", first=picture["id"], ratio="2:3", ownRatio="2:3")
        p["doc"]["ratio"] = "2:3"
    result = compile_project(p)
    assert result["ready"]
    out = result["shots"][0]
    assert (
        out["recipe"] == recipe
        and out["audio_mode"] == mode
        and out["delivery_audio"] == delivery
    )
    assert out["drive_audio"] == (a["id"] if sound == "record" else None)
    assert out["final_audio"] == (a["id"] if sound == "record" else None)
    assert out["media_map"][-1]["role"] == (
        "drive_audio" if sound == "record" else "ref_audio"
    )


def test_actual_cpu_input_preview_is_contained_not_stretched_and_original_unchanged(
    store,
):
    a = asset(store)
    before = (store.input_root / a["server_path"]).read_bytes()
    prepared = store.prepare_image(a["id"], 768, 448)
    with Image.open(store.input_root / prepared["server_path"]) as image:
        assert image.size == (768, 448)
        assert image.getpixel((0, 0)) == (0, 0, 0)
        assert image.getpixel((384, 224)) == (20, 100, 60)
    assert (store.input_root / a["server_path"]).read_bytes() == before
    assert store.prepare_image(a["id"], 768, 448) == prepared
    p = new_project()
    p["assets"] = [a]
    s = p["doc"]["shots"][0]
    s.update(mode="first", first=a["id"], simplePrompt="raw")
    report = compile_project(p, store)
    assert report["ready"]
    assert report["shots"][0]["canvas"]["preprocessing"][0]["state"] == "cpu_processed"
    assert report["shots"][0]["media_map"][0]["source_server_path"] == a["server_path"]


def test_same_size_derived_preview_tamper_is_rejected(store):
    a = asset(store)
    prepared = store.prepare_image(a["id"], 768, 448)
    Image.new("RGB", (768, 448), (255, 0, 0)).save(
        store.input_root / prepared["server_path"]
    )
    with pytest.raises(ValueError, match="缓存身份或字节损坏"):
        store.prepare_image(a["id"], 768, 448)


@pytest.mark.parametrize("text", ["@image0", "<Picture 9>", "<Audio 0>", "<Video 1>"])
def test_unbound_direct_native_and_zero_alias_slots_are_not_silently_valid(text):
    p = new_project()
    p["doc"]["shots"][0]["simplePrompt"] = text
    assert not compile_project(p)["ready"]


def test_active_draft_only_global_once_and_event_grid():
    p = new_project()
    p["doc"]["global"] = "全片\r\n保持风格  "
    s = p["doc"]["shots"][0]
    s.update(
        simplePrompt="整段不解析 [3秒]",
        prompt="高级整镜",
        duration=4,
        manualDuration=4,
        events=[{"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "只说一次"}],
    )
    simple = compile_project(p)["shots"][0]
    assert (
        simple["prompt"] == "全片\r\n保持风格  \n\n整段不解析 [3秒]"
        and simple["events"] == []
    )
    assert simple["time"]["requested_seconds"] == 4
    s["writingMode"] = "advanced"
    out = compile_project(p)["shots"][0]
    assert "整段不解析" not in out["prompt"] and out["prompt"].count("全片") == 1
    assert out["events"][0]["end_frame"] == 48 and out["time"]["requested_seconds"] == 2
    s["events"][0]["end"] = 0.1
    assert not compile_project(p)["ready"]


def test_limits_apply_to_reference_slots_not_generation_frames(store):
    refs = [asset(store) for _ in range(10)]
    p = new_project()
    p["assets"] = refs
    s = p["doc"]["shots"][0]
    s.update(
        mode="refs",
        refs=[a["id"] for a in refs],
        simplePrompt="raw",
        manualDuration=6000,
    )
    result = compile_project(p)
    assert not result["ready"] and len(result["shots"][0]["media_map"]) == 10
    s["refs"].pop()
    result = compile_project(p)
    assert result["ready"] and result["shots"][0]["time"]["aligned_frames"] > 144000


def test_native_cpu_workflow_and_api_are_separate_artifacts():
    p = new_project()
    s = p["doc"]["shots"][0]
    s["simplePrompt"] = "Hello\r\n中文  "
    exports = export_preflight_workflow(p, s["id"])
    assert exports["workflow"]["nodes"][0]["type"] == "MiniMaxH3DirectorProjectT8"
    assert exports["api_snapshot"]["1"]["class_type"] == "MiniMaxH3DirectorProjectT8"
    assert json.loads(exports["workflow"]["nodes"][0]["widgets_values"][0]) == p
    assert exports["workflow"]["links"] == []
    assert (
        "nodes" not in exports["api_snapshot"] and not compile_project(p)["gpu_queued"]
    )


def test_traversal_and_asset_overwrite_are_refused(store):
    with pytest.raises(ValueError):
        store.load("../../anything")
    a = asset(store)
    with pytest.raises(ProjectConflict):
        store.register_asset(store.input_root / a["server_path"], a["id"], "same.png")
