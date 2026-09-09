"""Run one serial, real-GPU H3 outpaint candidate with regional prompts.

The probe reuses only authenticated source/audio preparation from an older run,
creates a new regional text snapshot, samples exactly the first window, and then
stops for human review.  It never copies sampled windows or auto-selects output.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from unittest.mock import patch

from PIL import Image
import pynvml

try:
    from tools import run_video_outpaint_t8_probe as t8
    from tools.build_video_outpaint_candidate_workflows import _closure
except ModuleNotFoundError:
    import run_video_outpaint_t8_probe as t8
    from build_video_outpaint_candidate_workflows import _closure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGIONS = [
    {"shot": "all", "region": "top", "prompt": "bright blue open sky with soft natural clouds"},
    {"shot": "all", "region": "bottom", "prompt": "warm stone terrace with a few scattered green leaves"},
]
# Preserve the existing import used by older local validation tooling.
REGIONS = DEFAULT_REGIONS
IMPLEMENTATIONS = (
    "video_outpaint_guidance.py",
    "video_outpaint_regional_conditioning.py",
    "video_outpaint_regional.py",
    "video_outpaint_candidate_execution.py",
    "nodes_video_outpaint_guidance.py",
    "nodes_video_outpaint_candidates.py",
)
RUNTIME_RECEIPT_FIELDS = {
    "source_prepared_receipt.json": frozenset({"chunks_encoded_this_call", "resume_from"}),
    "source_audio_prepared_receipt.json": frozenset({
        "chunks_encoded_this_call", "replayed_prefix_chunks", "resume_from",
    }),
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tree_manifest(root):
    root = Path(root).resolve(strict=True)
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError(f"refusing linked cache assets below {root}")
    return [
        {"path": item.relative_to(root).as_posix(), "bytes": item.stat().st_size,
         "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in sorted((candidate for candidate in root.rglob("*")
                            if candidate.is_file() and candidate.name != "manifest.lock.v2"),
                           key=lambda candidate: candidate.relative_to(root).as_posix())
    ]


def _without_lock_records(records):
    return [item for item in records if not item["path"].endswith("manifest.lock.v2")]


def _cache_identity_manifest(root):
    """Fingerprint reusable payload identity while ignoring only run-progress fields."""
    root = Path(root).resolve(strict=True)
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError(f"refusing linked cache assets below {root}")
    records = []
    for item in sorted((candidate for candidate in root.rglob("*")
                        if candidate.is_file() and candidate.name != "manifest.lock.v2"),
                       key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        transient = RUNTIME_RECEIPT_FIELDS.get(item.name)
        if transient is None:
            records.append({"path": relative, "bytes": item.stat().st_size,
                            "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
            continue
        receipt = json.loads(item.read_bytes())
        missing = transient - set(receipt)
        if missing:
            raise ValueError(f"runtime receipt {relative} lacks expected fields: {sorted(missing)}")
        stable = {key: value for key, value in receipt.items() if key not in transient}
        encoded = _canonical(stable).encode()
        records.append({
            "path": relative,
            "stable_bytes": len(encoded),
            "stable_sha256": hashlib.sha256(encoded).hexdigest(),
            "ignored_runtime_fields": sorted(transient),
        })
    return records


def _verify_reused_cache_identity(cache, before, before_identity):
    """Require immutable bytes and stable receipt identity to survive preparation."""
    def immutable(records):
        return [
            item for item in _without_lock_records(records)
            if Path(item["path"]).name not in RUNTIME_RECEIPT_FIELDS
        ]
    after = {name: _tree_manifest(cache / name) for name in ("source", "audio")}
    if any(immutable(after[name]) != immutable(before[name]) for name in ("source", "audio")):
        raise RuntimeError("regional preparation rewrote authenticated source/audio payload or manifests")
    after_identity = {
        name: _cache_identity_manifest(cache / name) for name in ("source", "audio")
    }
    if after_identity != before_identity:
        raise RuntimeError("regional preparation changed stable source/audio receipt identity")
    return after, after_identity


def _validate_first_window_checkpoint(window, plan):
    planned = sum(len(shot["windows"]) for shot in plan["shots"])
    committed = window.get("committed", [])
    expected_status = "sampled" if planned == 1 else "paused"
    if planned < 1 or len(committed) != 1 or window.get("status") != expected_status:
        raise RuntimeError(
            "regional candidate did not stop normally after exactly one planned window")
    return {
        "planned_windows": planned,
        "sampling_complete": planned == 1,
        "checkpoint_status": expected_status,
    }


def build_regional_candidate_prompt(prior, query_chunk_rows=512, regions=None):
    regions = DEFAULT_REGIONS if regions is None else regions
    graph = deepcopy(prior["prompt"])
    reserved = {"regional_guidance", "regional_prepare", "regional_model",
                "regional_candidate", "candidate_identifier"}
    if set(graph) & reserved:
        raise ValueError("prior prompt uses reserved regional probe IDs")
    stages = {}
    for kind in ("Plan", "Prepare", "Sample", "Compose"):
        found = [key for key, node in graph.items()
                 if node["class_type"] == f"MiniMaxH3VideoOutpaint{kind}T8"]
        if len(found) != 1:
            raise ValueError("prior prompt needs exactly one of each outpaint stage")
        stages[kind] = found[0]
    prepare = graph[stages["Prepare"]]["inputs"]
    sample = graph[stages["Sample"]]["inputs"]
    compose = graph[stages["Compose"]]["inputs"]
    graph["regional_guidance"] = {
        "class_type": "MiniMaxH3VideoOutpaintGuidanceT8",
        "inputs": {
            "plan": [stages["Plan"], 0],
            "regions_json": _canonical(regions),
            "person_boxes_json": "[]",
        },
    }
    graph["regional_prepare"] = {
        "class_type": "MiniMaxH3VideoOutpaintPrepareGuidedT8",
        "inputs": {
            "plan": [stages["Plan"], 0],
            "guidance": ["regional_guidance", 0],
            "clip": prepare["clip"],
            "video_vae": prepare["video_vae"],
            "prompt": prepare["prompt"],
            "shot_prompts_json": prepare["shot_prompts_json"],
            "run_name": f"{prepare['run_name']}_regional",
            "audio_track": prepare["audio_track"],
            "audio_block_tokens": prepare["audio_block_tokens"],
            "resume_audio": True,
            "audio_vae": prepare["audio_vae"],
        },
    }
    graph["regional_model"] = {
        "class_type": "MiniMaxH3VideoOutpaintRegionalModelT8",
        "inputs": {
            "model": sample["model"],
            "prepared": ["regional_prepare", 0],
            "query_chunk_rows": int(query_chunk_rows),
        },
    }
    graph["regional_candidate"] = {
        "class_type": "MiniMaxH3VideoOutpaintCandidateT8",
        "inputs": {
            "model": ["regional_model", 0],
            "prepared": ["regional_model", 1],
            "video_vae": compose["video_vae"],
            "candidate_name": "regional_candidate_01",
            "seed": sample["seed"],
            "steps": sample["steps"],
            "resume": False,
            "color_match": compose["color_match"],
            **t8.cases.candidate_finish_inputs(compose),
        },
    }
    graph["candidate_identifier"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["regional_candidate", 3]},
    }
    return _closure(graph, ["candidate_identifier"])


def _validate_regional_manifest(path, expected_plan_sha, regions=None):
    regions = DEFAULT_REGIONS if regions is None else regions
    blob = path.read_bytes()
    manifest = json.loads(blob)
    digest = manifest.pop("sha256", None)
    if digest != hashlib.sha256(_canonical(manifest).encode()).hexdigest():
        raise ValueError("regional conditioning embedded hash mismatch")
    if manifest.get("schema") != "t8.h3.outpaint.regional_conditioning/v1":
        raise ValueError("unexpected regional conditioning schema")
    if manifest.get("plan_sha256") != expected_plan_sha:
        raise ValueError("regional conditioning belongs to another plan")
    if (path.parent / "outpaint_conditioning.json").exists():
        raise ValueError("plain and regional conditioning manifests coexist")
    guidance = manifest["guidance"]
    binding = manifest["binding"]
    guidance_digest = guidance.pop("guidance_sha256", None)
    if guidance_digest != hashlib.sha256(_canonical(guidance).encode()).hexdigest():
        raise ValueError("regional guidance embedded hash mismatch")
    guidance["guidance_sha256"] = guidance_digest
    binding_digest = binding.pop("binding_sha256", None)
    if binding_digest != hashlib.sha256(_canonical(binding).encode()).hexdigest():
        raise ValueError("regional binding embedded hash mismatch")
    binding["binding_sha256"] = binding_digest
    if binding.get("guidance_sha256") != guidance_digest:
        raise ValueError("regional binding and guidance hashes differ")
    if guidance["request"]["regions"] != regions:
        raise ValueError("regional guidance request differs from the probe")
    if guidance["request"]["person_boxes"]:
        raise ValueError("unexpected person boxes in this probe")
    checked_shots = []
    for shot in binding["shots"]:
        regions = shot.get("regions", [])
        if [item.get("kind") for item in regions] != ["top", "bottom"]:
            raise ValueError("regional binding does not contain exact top/bottom routes")
        last_end = 0
        checked_regions = []
        for region in regions:
            start, end = int(region["text_key_start"]), int(region["text_key_end"])
            if not 0 < start < end <= int(shot["text_len"]) or start < last_end:
                raise ValueError("regional text token spans are empty, overlapping or out of range")
            rows = region.get("spatial_rows")
            if not isinstance(rows, list) or not rows or rows != sorted(set(rows)):
                raise ValueError("regional spatial rows are empty or non-canonical")
            checked_regions.append({"kind": region["kind"], "text_key_start": start,
                                    "text_key_end": end, "spatial_row_count": len(rows)})
            last_end = end
        checked_shots.append({"shot_index": shot["shot_index"], "text_len": shot["text_len"],
                              "regions": checked_regions})
    return {
        "manifest_file_sha256": hashlib.sha256(blob).hexdigest(),
        "manifest_sha256": digest,
        "guidance_sha256": guidance_digest,
        "binding_sha256": binding_digest,
        "sampling_grid": binding["sampling_grid"],
        "shots": checked_shots,
    }


def _postflight_existing_candidate(
        report, cache, before, sha, plan, regions=None, before_identity=None):
    if before_identity is None:
        raise ValueError("pre-generation stable cache identity is required")
    after, after_identity = _verify_reused_cache_identity(cache, before, before_identity)
    report["reused_cache_after"] = after
    report["reused_cache_identity_after"] = after_identity
    report["cache_lock_files_excluded_from_payload_identity"] = True
    report["runtime_receipt_progress_fields_excluded_from_payload_identity"] = {
        name: sorted(fields) for name, fields in RUNTIME_RECEIPT_FIELDS.items()
    }
    text_root = cache / "text"
    regional_path = text_root / "outpaint_regional_conditioning.json"
    regional = _validate_regional_manifest(regional_path, plan["plan_sha256"], regions)
    text_files = sorted(item.name for item in text_root.iterdir() if item.is_file())
    if len([name for name in text_files if name.startswith("conditioning-")
            and name.endswith(".safetensors")]) != 1:
        raise RuntimeError("expected exactly one immutable regional conditioning tensor asset")
    report["regional_conditioning"] = regional
    report["regional_conditioning_files"] = text_files
    candidate_root = cache / "candidates/regional_candidate_01"
    archives = list(candidate_root.glob("candidate-*.json"))
    if len(archives) != 1:
        raise RuntimeError("expected exactly one regional candidate archive")
    saved = json.loads(archives[0].read_bytes())
    window = json.loads((candidate_root / "outpaint_windows.json").read_bytes())
    checkpoint = _validate_first_window_checkpoint(window, plan)
    image_path = candidate_root / f"preview-{saved['png_sha256']}.png"
    with Image.open(image_path) as picture:
        rgb_sha = hashlib.sha256(picture.convert("RGB").tobytes()).hexdigest()
        dimensions = list(picture.size)
    if sha(image_path).lower() != saved["png_sha256"] or rgb_sha != saved["preview_report"]["rgb8_sha256"]:
        raise RuntimeError("regional candidate PNG failed byte/RGB checks")
    report.update(
        status="regional_candidate_generated_human_review_pending",
        candidate_id=saved["sha256"],
        cache_root=str(candidate_root),
        image_path=str(image_path),
        image_dimensions=dimensions,
        preview_report=saved["preview_report"],
        committed_windows=1,
        paused=checkpoint["checkpoint_status"] == "paused",
        sampling_complete=checkpoint["sampling_complete"],
        planned_windows=checkpoint["planned_windows"],
        new_first_window_sha256=window["committed"][0]["sha256"],
    )


def finalize_existing(args):
    root = args.run_root.resolve(strict=True)
    root.relative_to(ROOT / "artifacts")
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    terminal = (report.get("execution", {}).get("terminal") or {}).get("type")
    if report.get("status") != "failed" or terminal != "execution_success":
        raise RuntimeError("existing probe is not a completed generation stopped by postflight audit")
    accepted_errors = {
        "RuntimeError: regional preparation rewrote authenticated source/audio cache",
        "RuntimeError: regional preparation rewrote authenticated source/audio payload or manifests",
    }
    if report.get("error") not in accepted_errors:
        raise RuntimeError("existing probe failed for another reason; refusing audit-only recovery")
    if not report.get("reused_cache_before"):
        raise RuntimeError("existing probe lacks the pre-generation cache fingerprints")
    run_name = report["prompt"]["regional_prepare"]["inputs"]["run_name"]
    cache = root / "output/T8_H3_Outpaint_Cache" / f"{run_name}-{report['plan_sha256'][:16]}"
    prior_root = Path(report["prior_root"]).resolve(strict=True)
    prior_root.relative_to(ROOT / "artifacts")
    prior = json.loads((prior_root / "report.json").read_bytes())
    plan = prior["case"]["plan"]
    if plan.get("plan_sha256") != report["plan_sha256"]:
        raise RuntimeError("prior plan does not match the failed regional candidate")
    prior_run_name = next(
        node["inputs"]["run_name"] for node in prior["prompt"].values()
        if node["class_type"] == "MiniMaxH3VideoOutpaintPrepareT8"
    )
    prior_cache = prior_root / "output/T8_H3_Outpaint_Cache" / (
        f"{prior_run_name}-{report['plan_sha256'][:16]}")
    prior_raw = {name: _tree_manifest(prior_cache / name) for name in ("source", "audio")}
    recorded_before = {
        name: _without_lock_records(report["reused_cache_before"][name])
        for name in ("source", "audio")
    }
    if prior_raw != recorded_before:
        raise RuntimeError("the authenticated pre-generation cache no longer matches its recorded hashes")
    before_identity = {
        name: _cache_identity_manifest(prior_cache / name) for name in ("source", "audio")
    }
    qualified = deepcopy(report)
    _postflight_existing_candidate(
        qualified, cache, report["reused_cache_before"], t8.upstream._sha256_file, plan,
        report.get("regions", DEFAULT_REGIONS), before_identity)
    qualified.pop("error", None)
    qualified["postflight_reused_existing_generation"] = True
    qualified["additional_sampling_for_postflight"] = False
    qualified["source_report"] = {
        "path": str(report_path),
        "sha256": t8.upstream._sha256_file(report_path),
        "preserved_unchanged": True,
    }
    qualified["audit_correction"] = (
        "The first audit treated expected lease-lock and run-progress receipt updates as payload "
        "changes. Learned payloads and content manifests remained byte-identical; all stable receipt "
        "identity fields remained semantically identical."
    )
    qualified_path = root / "report.requalified.json"
    t8.upstream._atomic_json(qualified_path, qualified)
    print(json.dumps({"status": qualified["status"], "report": str(qualified_path),
                      "additional_sampling": False}, ensure_ascii=False), flush=True)
    return qualified


def run(args):
    probe, sha, atomic = t8.upstream.probe, t8.upstream._sha256_file, t8.upstream._atomic_json
    prior_root, root = args.prior_root.resolve(strict=True), args.run_root.resolve()
    prior_root.relative_to(ROOT / "artifacts")
    root.relative_to(ROOT / "artifacts")
    prior_path = prior_root / "report.json"
    prior = json.loads(prior_path.read_bytes())
    t8.verify_prior_server_inactive(prior, prior_path)
    regions = [
        {"shot": "all", "region": "top", "prompt": args.top_prompt},
        {"shot": "all", "region": "bottom", "prompt": args.bottom_prompt},
    ]
    graph = build_regional_candidate_prompt(prior, args.query_chunk_rows, regions)
    case, plan = prior["case"], prior["case"]["plan"]
    source = Path(case["source_path"]).resolve(strict=True)
    source.relative_to(ROOT / "artifacts")
    run_name = graph["regional_prepare"]["inputs"]["run_name"]
    cache_name = f"{run_name}-{plan['plan_sha256'][:16]}"
    prior_run_name = next(node["inputs"]["run_name"] for node in prior["prompt"].values()
                          if node["class_type"] == "MiniMaxH3VideoOutpaintPrepareT8")
    prior_cache_name = f"{prior_run_name}-{plan['plan_sha256'][:16]}"
    prior_cache = prior_root / "output/T8_H3_Outpaint_Cache" / prior_cache_name
    if any(item.is_symlink() for item in prior_cache.rglob("*")):
        raise ValueError("refusing linked preparation assets")
    models = []
    for node in graph.values():
        directory, key = {
            "UNETLoader": ("diffusion_models", "unet_name"),
            "CLIPLoader": ("text_encoders", "clip_name"),
            "VAELoader": ("vae", "vae_name"),
        }.get(node["class_type"], (None, None))
        if directory:
            models.append(args.comfy_root / "models" / directory / node["inputs"][key])
    models = list(dict.fromkeys(path.resolve() for path in models))
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**2
    checks = {
        "new_artifact_root": not root.exists(),
        "source_hash_matches": sha(source).lower() == prior["source_sha256"].lower(),
        "gpu_headroom": free >= args.min_free_mib,
        "port_free": not probe.port_is_listening(args.host, args.port),
        "no_user_gpu_server": not probe.port_is_listening("127.0.0.1", 8188),
        "no_upstream_gpu_server": not probe.port_is_listening("127.0.0.1", 8190),
        "models_present": all(path.is_file() for path in models),
        "kj_audited": t8.cases.is_audited_kj_source(args.comfy_root / "custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py"),
        "prepared_directories_present": all((prior_cache / name).is_dir() for name in ("source", "audio")),
        "regional_implementation_present": all((ROOT / name).is_file() for name in IMPLEMENTATIONS),
    }
    report = {
        "scope": "regional_guidance_first_window_candidate_not_full_video_acceptance",
        "status": "preflight",
        "checks": checks,
        "prompt": graph,
        "prior_root": str(prior_root),
        "source_sha256": prior["source_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "gpu_initial_free_mib": free,
        "query_chunk_rows": args.query_chunk_rows,
        "regions": regions,
        "person_boxes": [],
        "human_acceptance": False,
        "automatic_selection": False,
        "old_sampled_windows_copied_or_changed": False,
        "full_video_generated": False,
        "implementation_sha256": {name: sha(ROOT / name) for name in IMPLEMENTATIONS},
    }
    if not args.confirm_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"regional candidate preflight failed: {checks}")
    root.mkdir(parents=True, exist_ok=False)
    atomic(root / "report.json", report)
    (root / "input").mkdir()
    source_node = next(node for node in graph.values() if node["class_type"] == "LoadVideo")
    shutil.copyfile(source, root / "input" / source_node["inputs"]["file"])
    old_models = {Path(model["path"]).resolve(): model for model in prior["models"]}
    report["models"] = []
    for model in models:
        actual = {"path": str(model), "bytes": model.stat().st_size, "sha256": sha(model)}
        expected = old_models.get(model.resolve())
        if expected is None or (actual["bytes"], actual["sha256"].lower()) != (
                expected["bytes"], expected["sha256"].lower()):
            raise ValueError("learned model file changed since the original prepared run")
        report["models"].append(actual)
    cache = root / "output/T8_H3_Outpaint_Cache" / cache_name
    cache.mkdir(parents=True)
    for name in ("source", "audio"):
        shutil.copytree(prior_cache / name, cache / name)
    before = {name: _tree_manifest(cache / name) for name in ("source", "audio")}
    before_identity = {
        name: _cache_identity_manifest(cache / name) for name in ("source", "audio")
    }
    report["reused_cache_before"] = before
    report["reused_cache_identity_before"] = before_identity
    if sha(root / "input" / source_node["inputs"]["file"]).lower() != prior["source_sha256"].lower():
        raise ValueError("source copy changed")
    extra = root / "draft_paths.yaml"
    atomic(extra, {"regional_candidate_test_only": {"custom_nodes": str(ROOT / "tools")}})
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        first, last = result.index("--whitelist-custom-nodes") + 1, result.index("--input-directory")
        result[first:last] = ["outpaint_candidate_extension", "ComfyUI-KJNodes"]
        result[result.index("--input-directory") + 1] = str(root / "input")
        return result + ["--extra-model-paths-config", str(extra)]

    server = probe.IsolatedServer(args, root, "regional-candidate")
    telemetry, started = None, time.monotonic()
    try:
        environment = dict(os.environ)
        environment.pop("T8_OUTPAINT_CAPTURE_RGB", None)
        if environment.get("CUDA_VISIBLE_DEVICES") == "-1":
            raise RuntimeError("parent process explicitly disables CUDA")
        with patch.object(probe, "_server_command", command), patch.dict(os.environ, environment, clear=True):
            server.start()
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
            info = json.load(response)
        required = {node["class_type"] for node in graph.values()}
        missing = required - set(info)
        if missing:
            raise RuntimeError(f"regional candidate schema missing: {missing}")
        atomic(root / "schema_snapshot.json", {key: info[key] for key in required})
        report.update(status="running", server_pid=server.process.pid)
        atomic(root / "report.json", report)
        telemetry = t8.LiveMemorySampler(server.process.pid, root / "telemetry.live.jsonl")
        telemetry.start()
        result = asyncio.run(probe.submit_prompt(
            server=f"http://{args.host}:{args.port}", prompt=graph, timeout_seconds=args.timeout))
        report["execution"] = result
        if (result.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("regional candidate execution failed; inspect execution record")
        _postflight_existing_candidate(
            report, cache, before, sha, plan, regions, before_identity)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if telemetry is not None:
                telemetry.stop()
                telemetry.write_csv(root / "telemetry.csv")
                report["memory"] = telemetry.summary()
                report["telemetry_error"] = telemetry.journal_error
        finally:
            server.stop()
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=ROOT.parents[2] / "python/python.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8198)
    parser.add_argument("--query-chunk-rows", type=int, default=512)
    parser.add_argument("--top-prompt", default=DEFAULT_REGIONS[0]["prompt"])
    parser.add_argument("--bottom-prompt", default=DEFAULT_REGIONS[1]["prompt"])
    parser.add_argument("--reserve-vram-gib", type=float, default=5.0)
    parser.add_argument("--min-free-mib", type=float, default=10000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    finalize_existing(arguments) if arguments.finalize_existing else run(arguments)
