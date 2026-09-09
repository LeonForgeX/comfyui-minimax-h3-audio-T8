from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.build_video_outpaint_regional_candidate_review import build_review
from tools.run_video_outpaint_regional_candidate_probe import (
    REGIONS,
    _cache_identity_manifest,
    _tree_manifest,
    _validate_first_window_checkpoint,
    _verify_reused_cache_identity,
    build_regional_candidate_prompt,
)
from test_video_outpaint_workflows import fixture


def test_regional_probe_builds_an_isolated_first_window_route_without_selection():
    prior = {"prompt": fixture()["prompt"]}
    before = deepcopy(prior)
    graph = build_regional_candidate_prompt(prior, query_chunk_rows=512)
    assert prior == before
    kinds = {node["class_type"] for node in graph.values()}
    assert {
        "MiniMaxH3VideoOutpaintGuidanceT8",
        "MiniMaxH3VideoOutpaintPrepareGuidedT8",
        "MiniMaxH3VideoOutpaintRegionalModelT8",
        "MiniMaxH3VideoOutpaintCandidateT8",
        "PreviewAny",
    } <= kinds
    assert not kinds & {
        "MiniMaxH3VideoOutpaintPrepareT8",
        "MiniMaxH3VideoOutpaintSampleT8",
        "MiniMaxH3VideoOutpaintComposeT8",
        "MiniMaxH3VideoOutpaintSelectCandidateT8",
        "MiniMaxH3VideoOutpaintContinueCandidateT8",
        "SaveVideo",
    }
    assert json.loads(graph["regional_guidance"]["inputs"]["regions_json"]) == REGIONS
    assert graph["regional_prepare"]["inputs"]["run_name"].endswith("_regional")
    assert graph["regional_prepare"]["inputs"]["resume_audio"] is True
    assert graph["regional_model"]["inputs"] == {
        "model": next(node["inputs"]["model"] for node in prior["prompt"].values()
                      if node["class_type"] == "MiniMaxH3VideoOutpaintSampleT8"),
        "prepared": ["regional_prepare", 0],
        "query_chunk_rows": 512,
    }
    candidate = graph["regional_candidate"]["inputs"]
    assert candidate["model"] == ["regional_model", 0]
    assert candidate["prepared"] == ["regional_model", 1]
    assert candidate["resume"] is False
    assert graph["candidate_identifier"]["inputs"]["source"] == ["regional_candidate", 3]


def test_regional_probe_accepts_material_specific_region_prompts():
    prior = {"prompt": fixture()["prompt"]}
    regions = [
        {"shot": "all", "region": "top", "prompt": "traditional tiled roof and leaves"},
        {"shot": "all", "region": "bottom", "prompt": "stone courtyard floor"},
    ]
    graph = build_regional_candidate_prompt(prior, regions=regions)
    assert json.loads(graph["regional_guidance"]["inputs"]["regions_json"]) == regions


def test_regional_probe_rejects_missing_or_duplicate_prior_stages():
    prior = {"prompt": fixture()["prompt"]}
    prior["prompt"] = {
        key: node for key, node in prior["prompt"].items()
        if node["class_type"] != "MiniMaxH3VideoOutpaintSampleT8"
    }
    with pytest.raises(ValueError, match="exactly one"):
        build_regional_candidate_prompt(prior)

    prior = {"prompt": fixture()["prompt"]}
    prior["prompt"]["regional_model"] = deepcopy(prior["prompt"]["12"])
    with pytest.raises(ValueError, match="reserved"):
        build_regional_candidate_prompt(prior)


def test_cache_payload_manifest_excludes_expected_lease_lock_updates(tmp_path):
    root = tmp_path / "cache"
    worker = root / "worker"
    worker.mkdir(parents=True)
    (root / "payload.bin").write_bytes(b"payload")
    (root / "manifest.lock.v2").write_bytes(b"first owner")
    (worker / "manifest.lock.v2").write_bytes(b"first worker")
    before = _tree_manifest(root)
    (root / "manifest.lock.v2").write_bytes(b"second owner")
    (worker / "manifest.lock.v2").write_bytes(b"second worker")
    assert _tree_manifest(root) == before


def _write_reused_cache_fixture(root):
    source = root / "source"
    audio = root / "audio"
    source.mkdir(parents=True)
    audio.mkdir(parents=True)
    (source / "video-a.safetensors").write_bytes(b"video")
    (source / "outpaint_source_latents.json").write_text('{"manifest":"video"}')
    (source / "source_prepared_receipt.json").write_text(json.dumps({
        "schema": "source/v1", "plan_sha256": "a" * 64,
        "cache_manifest_sha256": "b" * 64,
        "chunks_encoded_this_call": 4, "resume_from": [0, 0, 5],
    }))
    (audio / "audio-a.safetensors").write_bytes(b"audio")
    (audio / "outpaint_source_audio.json").write_text('{"manifest":"audio"}')
    (audio / "source_audio_prepared_receipt.json").write_text(json.dumps({
        "schema": "audio/v1", "plan_sha256": "a" * 64,
        "audio_manifest_sha256": "c" * 64,
        "chunks_encoded_this_call": 2, "replayed_prefix_chunks": 0, "resume_from": 0,
    }))
    return source, audio


def test_reused_cache_audit_allows_only_lock_and_receipt_progress_updates(tmp_path):
    source, audio = _write_reused_cache_fixture(tmp_path)
    before = {name: _tree_manifest(tmp_path / name) for name in ("source", "audio")}
    identity = {
        name: _cache_identity_manifest(tmp_path / name) for name in ("source", "audio")
    }
    (source / "manifest.lock.v2").write_text("new owner")
    (audio / "audio-worker").mkdir()
    (audio / "audio-worker/manifest.lock.v2").write_text("new worker")
    source_receipt = json.loads((source / "source_prepared_receipt.json").read_text())
    source_receipt.update(chunks_encoded_this_call=0, resume_from=None)
    (source / "source_prepared_receipt.json").write_text(json.dumps(source_receipt))
    audio_receipt = json.loads((audio / "source_audio_prepared_receipt.json").read_text())
    audio_receipt.update(chunks_encoded_this_call=0, replayed_prefix_chunks=0, resume_from=2)
    (audio / "source_audio_prepared_receipt.json").write_text(json.dumps(audio_receipt))
    _, after_identity = _verify_reused_cache_identity(tmp_path, before, identity)
    assert after_identity == identity


@pytest.mark.parametrize("tamper", ["payload", "manifest", "stable_receipt"])
def test_reused_cache_audit_rejects_real_identity_changes(tmp_path, tamper):
    source, _audio = _write_reused_cache_fixture(tmp_path)
    before = {name: _tree_manifest(tmp_path / name) for name in ("source", "audio")}
    identity = {
        name: _cache_identity_manifest(tmp_path / name) for name in ("source", "audio")
    }
    if tamper == "payload":
        (source / "video-a.safetensors").write_bytes(b"changed")
    elif tamper == "manifest":
        (source / "outpaint_source_latents.json").write_text('{"manifest":"changed"}')
    else:
        receipt = json.loads((source / "source_prepared_receipt.json").read_text())
        receipt["plan_sha256"] = "d" * 64
        (source / "source_prepared_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="rewrote|stable"):
        _verify_reused_cache_identity(tmp_path, before, identity)


def test_first_window_checkpoint_accepts_single_window_as_sampled():
    plan = {"shots": [{"windows": [{"render_frames": 56}]}]}
    result = _validate_first_window_checkpoint(
        {"status": "sampled", "committed": [{"window": 0}]}, plan)
    assert result == {
        "planned_windows": 1,
        "sampling_complete": True,
        "checkpoint_status": "sampled",
    }


def test_first_window_checkpoint_requires_pause_for_a_longer_plan():
    plan = {"shots": [{"windows": [{}, {}]}]}
    result = _validate_first_window_checkpoint(
        {"status": "paused", "committed": [{"window": 0}]}, plan)
    assert result["sampling_complete"] is False
    with pytest.raises(RuntimeError, match="exactly one planned window"):
        _validate_first_window_checkpoint(
            {"status": "sampled", "committed": [{"window": 0}]}, plan)


def review_pair(tmp_path, top=96, bottom=96, mode="preserve_source"):
    from tools import outpaint_probe_cases as cases
    plan = cases.plan_module().build_outpaint_plan(source_sha256="a" * 64, width=32, height=16,
        frame_count=56, source_fps="24/1", aspect="custom", top=top, bottom=bottom)
    def seal(value):
        body = {k: v for k, v in value.items() if k != "sha256"}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {**body, "sha256": digest}

    base = Image.new("RGB", (32, top + bottom + 16), (40, 50, 60))
    regional = base.copy()
    regional.paste((80, 90, 100), (0, 0, 32, top))
    regional.paste((110, 120, 130), (0, top + 16, 32, top + bottom + 16))

    def save_candidate(name, image, status):
        temporary = tmp_path / f"{name}.png"
        image.save(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        path = tmp_path / f"preview-{digest}.png"
        temporary.replace(path)
        report = tmp_path / f"{name}.json"
        cache = tmp_path / name
        cache.mkdir()
        payload = name.encode()
        window_sha = hashlib.sha256(payload).hexdigest()
        (cache / f"window-{window_sha}.safetensors").write_bytes(payload)
        identity = {k: "c" * 64 for k in ("model_sha256", "implementation_sha256", "source_cache_sha256",
            "audio_source_sha256", "conditioning_sha256")}
        identity.update(plan_sha256=plan["plan_sha256"], seed=19, steps=8,
            noise_algorithm="t8.outpaint.native_cpu_noise/v1", sampler_name="res_multistep", scheduler="simple")
        identity["model_sha256"] = ("e" if name == "region" else "c") * 64
        manifest = seal({"schema": "t8.h3.video_outpaint.window_store/v1", "status": "sampled",
            "identity": identity, "committed": [{"shot": 0, "window": 0, "sha256": window_sha, "bytes": len(payload)}]})
        manifest_path = cache / "outpaint_windows.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        preview = seal({"schema": "t8.h3.video_outpaint.candidate_first_frame/v1",
            "plan_sha256": plan["plan_sha256"], "source_sha256": "a" * 64,
            "source_mode": mode, "source_exact_before_encoding": mode == "preserve_source",
            "source_reconstructed": mode == "joint_decode", "width": image.width, "height": image.height,
            "rgb8_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "window_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "video_vae_sha256": "d" * 64, "color_settings": {"enabled": True}, "geometry_settings": {}})
        report.write_text(json.dumps({
            "status": status,
            "human_acceptance": False,
            "image_path": str(path),
            "candidate_id": name * 8,
            "new_first_window_sha256": window_sha,
            "source_sha256": "a" * 64,
            "plan_sha256": plan["plan_sha256"], "cache_root": str(cache), "preview_report": preview,
            "models": [{"path": "base.safetensors", "bytes": 100, "sha256": "f" * 64}],
            "prompt": {"candidate": {"class_type": "MiniMaxH3VideoOutpaintCandidateT8", "inputs": {
                "model": ["router" if name == "region" else "unet", 0],
                "seed": 19, "steps": 8, "color_match": True, "source_mode": mode, "geometry_align": False}},
                "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors", "weight_dtype": "default"}},
                "router": {"class_type": "MiniMaxH3VideoOutpaintRegionalModelT8", "inputs": {
                    "model": ["unet", 0], "prepared": ["prepared", 0], "query_chunk_rows": 256}}},
        }), encoding="utf-8")
        return report

    plain_report = save_candidate("plain", base, "candidate_generated_human_review_pending")
    regional_report = save_candidate(
        "region", regional, "regional_candidate_generated_human_review_pending")
    return plain_report, regional_report, plan, seal


@pytest.mark.parametrize("top,bottom", [(96, 96), (95, 97), (15, 32)])
@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode"])
def test_blind_review_hides_mapping_and_measures_only_expansion_changes(tmp_path, top, bottom, mode):
    plain_report, regional_report, plan, _ = review_pair(tmp_path, top, bottom, mode)
    result = build_review(plain_report, regional_report, tmp_path / "review", seed=7, plan=plan)
    page = Path(result["page"]).read_text(encoding="utf-8")
    assert '"method"' not in page
    assert str(plain_report) not in page and str(regional_report) not in page
    metrics = result["private_key"]["mechanical_difference_only_not_quality"]
    assert metrics["source_rectangle"]["mean_absolute_rgb_difference"] == 0
    assert metrics["top"]["mean_absolute_rgb_difference"] > 0
    assert metrics["bottom"]["mean_absolute_rgb_difference"] > 0
    assert result["public_manifest"]["source_rectangle"] == [0, top, 32, top + 16]
    assert result["public_manifest"]["source_mode"] == mode
    assert "19" in page and "8 步" in page and "0.5MP" not in page
    assert "本页只比较首帧" in page
    if mode == "joint_decode":
        assert "原片区域会经 VAE 重建" in page
    for label in ("A", "B"):
        with Image.open(tmp_path / f"review/public/{label}_top.png") as crop:
            assert crop.size == (32, top)
        with Image.open(tmp_path / f"review/public/{label}_bottom.png") as crop:
            assert crop.size == (32, bottom)


@pytest.mark.parametrize("damage", ["no_plan", "plan_geometry", "preview_seal", "preview_rgb", "preview_mode",
    "paired_mode", "steps", "color", "geometry", "manifest", "payload", "model", "vae",
    "implementation", "base_graph", "unknown_wrapper"])
def test_blind_review_rejects_misleading_pair_before_publishing(tmp_path, damage):
    plain, regional, plan, seal = review_pair(tmp_path)
    data = json.loads(regional.read_bytes())
    preview = data["preview_report"]
    inputs = data["prompt"]["candidate"]["inputs"]
    if damage == "no_plan":
        plan = None
    elif damage == "plan_geometry":
        plan["output"]["source_rect"][1] = 94
        plan = seal(plan)  # Even re-sealed geometry must match the request.
    elif damage == "preview_seal":
        preview["width"] = 33
    elif damage in {"preview_rgb", "preview_mode", "paired_mode", "vae"}:
        if damage == "preview_rgb":
            preview["rgb8_sha256"] = "0" * 64
        elif damage == "vae":
            preview["video_vae_sha256"] = "e" * 64
        else:
            preview.update(source_mode="joint_decode", source_exact_before_encoding=False, source_reconstructed=True)
            if damage == "paired_mode":
                inputs["source_mode"] = "joint_decode"
        data["preview_report"] = seal(preview)
    elif damage == "steps":
        inputs["steps"] = 20
    elif damage == "color":
        inputs["color_match"] = False
    elif damage == "geometry":
        inputs["geometry_align"] = True
    elif damage == "payload":
        (Path(data["cache_root"]) / f"window-{data['new_first_window_sha256']}.safetensors").write_bytes(b"damaged")
    elif damage == "model":
        data["models"][0]["sha256"] = "0" * 64
    elif damage == "base_graph":
        data["prompt"]["unet"]["inputs"]["weight_dtype"] = "other"
    elif damage == "unknown_wrapper":
        data["prompt"]["router"]["class_type"] = "UnknownWrapper"
    else:
        path = Path(data["cache_root"]) / "outpaint_windows.json"
        window = json.loads(path.read_bytes())
        window["identity"]["implementation_sha256"] = "e" * 64
        path.write_text(json.dumps(seal(window)))
        if damage == "implementation":
            preview["window_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            data["preview_report"] = seal(preview)
    regional.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build_review(plain, regional, tmp_path / "review", plan=plan)
    assert not (tmp_path / "review").exists()
