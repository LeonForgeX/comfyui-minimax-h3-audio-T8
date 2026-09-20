from __future__ import annotations

import uuid
import wave

import pytest
from PIL import Image

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import ProjectStore, new_project


def _store(tmp_path):
    return ProjectStore(tmp_path / "user", tmp_path / "input")


def _image(store, name="frame.png"):
    asset_id = str(uuid.uuid4())
    path = store.input_root / "t8_director" / asset_id / "source.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (512, 768), (40, 80, 120)).save(path)
    return store.register_asset(path, asset_id, name)


def _audio(store, name="voice.wav", seconds=4.0):
    asset_id = str(uuid.uuid4())
    path = store.input_root / "t8_director" / asset_id / "source.wav"
    path.parent.mkdir(parents=True)
    frames = int(16_000 * seconds)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * frames)
    return store.register_asset(path, asset_id, name)


def test_d2a_builds_real_native_t2va_prompt_without_queue(tmp_path):
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet cinematic room, one continuous shot."
    built = build_director_generation_prompt(project, project["current"], store)
    graph = built["prompt"]
    assert graph["5"]["class_type"] == "MiniMaxH3AudioConditioningT8"
    assert graph["5"]["inputs"]["task_type"] == "T2VA"
    assert graph["6"]["class_type"] == "MiniMaxH3DualClockSamplerT8"
    assert graph["12"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
    assert built["recipe"].startswith("director_")


def test_d2a_binds_prepared_first_frame(tmp_path):
    store = _store(tmp_path)
    frame = _image(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=frame["id"], tray=[frame["id"]], simplePrompt="A portrait moves gently.")
    project["assets"] = [frame]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["14"]["class_type"] == "LoadImage"
    assert built["prompt"]["14"]["inputs"]["image"].startswith("t8_director/")


def test_director_recipe_contract_is_not_downgraded_for_unsupported_assets(tmp_path):
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", simplePrompt="A scene.")
    with pytest.raises(ValueError, match="参考素材"):
        build_director_generation_prompt(project, shot["id"], store)


def test_d2b_binds_ends_and_ref2va_with_native_task_labels(tmp_path):
    store = _store(tmp_path)
    first, last, ref = _image(store, "first.png"), _image(store, "last.png"), _image(store, "ref.png")
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="ends", first=first["id"], last=last["id"], tray=[first["id"], last["id"]], simplePrompt="A clean transition.")
    project["assets"] = [first, last]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "FL2VA"
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["5"]["inputs"]["last_frame"] == ["15", 0]

    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", refs=[ref["id"]], tray=[ref["id"]], simplePrompt="Keep the character identity.")
    project["assets"] = [ref]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "Ref2VA"
    assert built["prompt"]["5"]["inputs"]["ref_images.ref_image_0"] == ["14", 0]

    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=first["id"], refs=[ref["id"]],
                tray=[first["id"], ref["id"]], simplePrompt="A person turns toward the reference.")
    project["assets"] = [first, ref]
    built = build_director_generation_prompt(project, shot["id"], store)
    assert built["prompt"]["5"]["inputs"]["task_type"] == "Hybrid"
    assert "ref2va" in built["prompt"]["1"]["inputs"]["unet_name"].lower()
    assert built["prompt"]["5"]["inputs"]["first_frame"] == ["14", 0]
    assert built["prompt"]["5"]["inputs"]["ref_images.ref_image_0"] == ["15", 0]


def test_d2c_record_uses_windowed_drive_audio_as_delivery_track(tmp_path):
    store = _store(tmp_path)
    frame, audio = _image(store), _audio(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="first", first=frame["id"], tray=[frame["id"], audio["id"]],
                sound="record", audio=audio["id"], start=0, end=4, duration=4, manualDuration=4,
                simplePrompt="A person speaks naturally.")
    project["assets"] = [frame, audio]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert built["recipe"].endswith("record")
    assert built["turbo_lora"] is None
    assert graph["6"]["inputs"]["steps"] == 8
    assert graph["5"]["inputs"]["task_type"] == "Hybrid"
    assert "ref2va" in graph["1"]["inputs"]["unet_name"].lower()
    assert graph["5"]["inputs"]["drive_audio"] == ["16", 0]
    assert graph["5"]["inputs"]["final_audio"] == ["16", 0]
    assert graph["5"]["inputs"]["length"] == ["16", 1]
    assert graph["11"]["inputs"]["audio"] == ["5", 2]
    assert graph["16"]["class_type"] == "MiniMaxH3AudioWindowT8"


def test_d2c_reference_voice_uses_audio_reference_not_delivery_audio(tmp_path):
    store = _store(tmp_path)
    audio = _audio(store)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(sound="voice", audio=audio["id"], start=0, end=4,
                tray=[audio["id"]], simplePrompt="Say the new line.")
    project["assets"] = [audio]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert graph["5"]["inputs"]["task_type"] == "Ref2VA"
    assert graph["5"]["inputs"]["ref_audios.ref_audio_0"] == ["14", 0]
    assert graph["11"]["inputs"]["audio"] == ["10", 1]


def test_d2b_reference_video_extracts_images_and_audio_components(tmp_path, monkeypatch):
    store = _store(tmp_path)
    import uuid as _uuid

    video_id = str(_uuid.uuid4())
    video = {
        "id": video_id,
        "name": "reference.mp4",
        "kind": "video",
        "width": 512,
        "height": 512,
        "duration": 3.0,
        "has_audio": True,
        "server_path": "t8_director/reference.mp4",
        "sha256": "video-sha",
        "size": 1,
    }
    monkeypatch.setattr(store, "asset", lambda asset_id, verify=True: video if asset_id == video_id else None)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(mode="refs", refs=[video_id], tray=[video_id], simplePrompt="Use the motion reference.")
    project["assets"] = [video]
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    assert graph["14"]["class_type"] == "LoadVideo"
    assert graph["15"]["class_type"] == "GetVideoComponents"
    assert graph["15"]["inputs"]["video"] == ["14", 0]
    assert graph["5"]["inputs"]["ref_videos.ref_video_0"] == ["15", 0]
    assert graph["5"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["15", 1]


def _patch_director_models(monkeypatch):
    from h3_audio_t8_pkg import director_generation

    monkeypatch.setattr(
        director_generation,
        "_pick",
        lambda _folder, candidates, _label: candidates[0],
    )
    monkeypatch.setattr(director_generation, "_optional_turbo_lora", lambda: None)
    monkeypatch.setattr(
        director_generation,
        "_optional_semantic_bridge_model",
        lambda: "t8_compat/semantic_bridge.safetensors",
    )


def test_d3_semantic_bridge_is_compiled_into_native_conditioning(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A stable portrait with gentle motion."
    project["doc"]["shots"][0]["d3"] = {
        "semantic_bridge": {"enabled": True, "alpha": 0.12},
    }
    built = build_director_generation_prompt(project, project["current"], store)
    graph = built["prompt"]
    bridge = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3SemanticBridgeConfigT8")
    apply = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3SemanticBridgeApplyT8")
    assert bridge["inputs"]["alpha"] == 0.12
    assert apply["inputs"]["conditioning"] == ["5", 0]
    assert graph["7"]["inputs"]["conditioning"] == [next(k for k, v in graph.items() if v is apply), 0]


def test_semantic_bridge_prefers_converted_minimax_compat_asset(monkeypatch):
    from h3_audio_t8_pkg import director_generation, nodes_semantic_bridge

    monkeypatch.setattr(
        nodes_semantic_bridge,
        "model_paths",
        lambda: {
            "bunny/BUNNY_H3_ActionLogic_Bridge_V1.safetensors": "bunny",
            "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors": "bunny-compat",
            "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors": "minimax-compat",
        },
    )
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda _folder: [])
    assert director_generation._optional_semantic_bridge_model() == (
        "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors"
    )


def test_d3_prompt_relay_replaces_conditioning_and_preserves_media_slots(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(
        writingMode="advanced",
        prompt="",
        global_prompt="",
        events=[
            {"id": str(uuid.uuid4()), "start": 0, "end": 2, "text": "The subject looks left."},
            {"id": str(uuid.uuid4()), "start": 2, "end": 4, "text": "The subject looks right."},
        ],
        simplePrompt="",
        d3={"prompt_relay": {"enabled": True, "execution_mode": "apply_exp"}},
    )
    project["doc"]["global"] = "A cinematic continuous shot."
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    plan_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3PromptRelayPlanT8Advanced")
    relay_id = next(k for k, v in graph.items() if v["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced")
    assert graph[plan_id]["inputs"]["timing_mode"] == "frames"
    assert graph[relay_id]["inputs"]["prompt_relay_plan"] == [plan_id, 0]
    assert graph["6"]["inputs"]["model"] == [relay_id, 0]
    assert graph["7"]["inputs"]["conditioning"] == [relay_id, 1]
    assert "MiniMaxH3AudioConditioningT8" not in {node["class_type"] for node in graph.values()}


def test_d3_fast_h3_and_memory_nodes_are_chained_without_changing_default_graph(tmp_path, monkeypatch):
    _patch_director_models(monkeypatch)
    store = _store(tmp_path)
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A calm subject speaks to camera."
    shot["d3"] = {
        "fast_h3_v2": {"enabled": True, "profile": "dense_compat_exp", "min_tokens": 8192},
        "memory": {"low_vram": True, "head_chunks": 4, "chunk_ffn": True, "chunks": 2, "seq_threshold": 4096},
    }
    built = build_director_generation_prompt(project, shot["id"], store)
    graph = built["prompt"]
    types = [node["class_type"] for node in graph.values()]
    assert "MiniMaxH3LowVRAMAttentionT8Advanced" in types
    assert "MiniMaxH3ChunkFeedForwardT8Advanced" in types
    assert "MiniMaxH3FastH3V2SetupEXPT8" in types
    assert "MiniMaxH3FastH3V2RuntimeAuditEXPT8" in types
    assert graph["9"]["class_type"] == "SamplerCustomAdvanced"
    assert graph["1"]["inputs"]["unet_name"] == "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
    assert built["turbo_lora"] is None


def test_generation_settings_support_multiple_loras_and_total_pixel_resolution(tmp_path, monkeypatch):
    from h3_audio_t8_pkg import director_generation

    _patch_director_models(monkeypatch)
    monkeypatch.setattr(
        director_generation,
        "_pick_requested",
        lambda _folder, requested, candidates, _label: candidates[0] if requested == "auto" else requested,
    )
    store = _store(tmp_path)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A stable cinematic portrait."
    project["doc"]["generation"].update(
        lora=["style_a.safetensors", "motion_b.safetensors"],
        lora_strength=0.65,
        resolution_mp=0.4,
    )
    built = director_generation.build_director_generation_prompt(project, project["current"], store)
    lora_nodes = [node for node in built["prompt"].values() if node["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"]
    assert [node["inputs"]["lora_name"] for node in lora_nodes] == ["style_a.safetensors", "motion_b.safetensors"]
    assert all(node["inputs"]["strength_model"] == 0.65 for node in lora_nodes)
    canvas = built["report"]["shots"][0]["canvas"]
    assert canvas["width"] % 32 == canvas["height"] % 32 == 0
    assert 0.35 <= canvas["width"] * canvas["height"] / 1_000_000 <= 0.45
