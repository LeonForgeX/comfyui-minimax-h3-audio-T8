"""One serial learned-GPU candidate from an existing verified preparation cache.

Dry-run by default. Copies only source/audio/text caches into a new artifact
root; never consumes or overwrites the old sampled windows. No auto-selection.
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


def build_candidate_prompt(prior):
    graph = deepcopy(prior["prompt"])
    stages = {}
    for kind in ("Plan", "Prepare", "Sample", "Compose"):
        found = [key for key, node in graph.items() if node["class_type"] == f"MiniMaxH3VideoOutpaint{kind}T8"]
        if len(found) != 1:
            raise ValueError("prior prompt needs exactly one of each stage")
        stages[kind] = found[0]
    prep, sample, compose = (graph[stages[kind]]["inputs"] for kind in ("Prepare", "Sample", "Compose"))
    if set(graph) & {"reload_candidate_prepared", "generated_candidate", "candidate_identifier"}:
        raise ValueError("prior prompt uses reserved candidate probe IDs")
    graph["reload_candidate_prepared"] = {"class_type": "MiniMaxH3VideoOutpaintLoadPreparedT8", "inputs": {
        "plan": [stages["Plan"], 0], "run_name": prep["run_name"]}}
    graph["generated_candidate"] = {"class_type": "MiniMaxH3VideoOutpaintCandidateT8", "inputs": {
        "model": sample["model"], "prepared": ["reload_candidate_prepared", 0], "video_vae": compose["video_vae"],
        "candidate_name": "learned_candidate_01", "seed": sample["seed"], "steps": sample["steps"],
        "resume": False, "color_match": compose["color_match"],
        **t8.cases.candidate_finish_inputs(compose)}}
    graph["candidate_identifier"] = {"class_type": "PreviewAny", "inputs": {"source": ["generated_candidate", 3]}}
    return _closure(graph, ["candidate_identifier"])


def run(args):
    probe, sha, atomic = t8.upstream.probe, t8.upstream._sha256_file, t8.upstream._atomic_json
    prior_root, root = args.prior_root.resolve(strict=True), args.run_root.resolve()
    prior_root.relative_to(ROOT / "artifacts")
    root.relative_to(ROOT / "artifacts")
    prior_path = prior_root / "report.json"
    prior = json.loads(prior_path.read_bytes())
    t8.verify_prior_server_inactive(prior, prior_path)
    graph = build_candidate_prompt(prior)
    case = prior["case"]
    source = Path(case["source_path"]).resolve(strict=True)
    source.relative_to(ROOT / "artifacts")
    plan = case["plan"]
    run_name = graph["reload_candidate_prepared"]["inputs"]["run_name"]
    cache_name = f"{run_name}-{plan['plan_sha256'][:16]}"
    prior_cache = prior_root / "output/T8_H3_Outpaint_Cache" / cache_name
    if any(p.is_symlink() for p in prior_cache.rglob("*")):
        raise ValueError("refusing linked preparation assets")
    models = []
    for node in graph.values():
        directory, key = {"UNETLoader": ("diffusion_models", "unet_name"),
            "VAELoader": ("vae", "vae_name")}.get(node["class_type"], (None, None))
        if directory:
            models.append(args.comfy_root / "models" / directory / node["inputs"][key])
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**2
    checks = {"new_artifact_root": not root.exists(), "source_hash_matches": sha(source) == prior["source_sha256"],
        "gpu_headroom": free >= args.min_free_mib, "port_free": not probe.port_is_listening(args.host, args.port),
        "no_user_gpu_server": not probe.port_is_listening("127.0.0.1", 8188),
        "no_upstream_gpu_server": not probe.port_is_listening("127.0.0.1", 8190),
        "models_present": all(p.is_file() for p in models),
        "kj_audited": t8.cases.is_audited_kj_source(args.comfy_root / "custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py"),
        "prepared_directories_present": all((prior_cache / name).is_dir() for name in ("source", "audio", "text"))}
    report = {"scope": "learned_first_window_candidate_not_full_video_acceptance", "status": "preflight",
        "checks": checks, "prompt": graph, "prior_root": str(prior_root), "source_sha256": prior["source_sha256"],
        "plan_sha256": plan["plan_sha256"], "gpu_initial_free_mib": free, "human_acceptance": False,
        "automatic_selection": False, "old_sampled_windows_copied_or_changed": False}
    reference_window_path = prior_cache / "sampling/outpaint_windows.json"
    reference_window = json.loads(reference_window_path.read_bytes())
    report["reference_complete_window_sha256"] = [item["sha256"] for item in reference_window["committed"]]
    report["reference_complete_manifest_sha256"] = sha(reference_window_path)
    if not args.confirm_run:
        print(json.dumps(report, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"candidate preflight failed: {checks}")
    root.mkdir(parents=True, exist_ok=False)
    atomic(root / "report.json", report)
    (root / "input").mkdir()
    source_node = next(n for n in graph.values() if n["class_type"] == "LoadVideo")
    shutil.copyfile(source, root / "input" / source_node["inputs"]["file"])
    old_models = {Path(m["path"]).resolve(): m for m in prior["models"]}
    report["models"] = []
    for model in models:
        actual = {"path": str(model), "bytes": model.stat().st_size, "sha256": sha(model)}
        if actual != old_models.get(model.resolve()):
            raise ValueError("learned model file changed since the original prepared run")
        report["models"].append(actual)
    cache = root / "output/T8_H3_Outpaint_Cache" / cache_name
    cache.mkdir(parents=True)
    for name in ("source", "audio", "text"):
        shutil.copytree(prior_cache / name, cache / name)
    if sha(root / "input" / source_node["inputs"]["file"]) != prior["source_sha256"]:
        raise ValueError("source copy changed")
    extra = root / "draft_paths.yaml"
    atomic(extra, {"candidate_test_only": {"custom_nodes": str(ROOT / "tools")}})
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        first, last = result.index("--whitelist-custom-nodes") + 1, result.index("--input-directory")
        result[first:last] = ["outpaint_candidate_extension", "ComfyUI-KJNodes"]
        result[result.index("--input-directory") + 1] = str(root / "input")
        return result + ["--extra-model-paths-config", str(extra)]

    server = probe.IsolatedServer(args, root, "candidate")
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
        missing = {n["class_type"] for n in graph.values()} - set(info)
        if missing:
            raise RuntimeError(f"candidate schema missing: {missing}")
        atomic(root / "schema_snapshot.json", {k: info[k] for k in {n["class_type"] for n in graph.values()}})
        report.update(status="running", server_pid=server.process.pid)
        atomic(root / "report.json", report)
        telemetry = t8.LiveMemorySampler(server.process.pid, root / "telemetry.live.jsonl")
        telemetry.start()
        result = asyncio.run(probe.submit_prompt(server=f"http://{args.host}:{args.port}", prompt=graph,
            timeout_seconds=args.timeout))
        report["execution"] = result
        if (result.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("learned candidate execution failed; inspect execution record")
        candidate_root = cache / "candidates/learned_candidate_01"
        archives = list(candidate_root.glob("candidate-*.json"))
        if len(archives) != 1:
            raise RuntimeError("expected exactly one candidate archive")
        saved = json.loads(archives[0].read_bytes())
        window = json.loads((candidate_root / "outpaint_windows.json").read_bytes())
        if window["status"] != "paused" or len(window["committed"]) != 1:
            raise RuntimeError("candidate did not stop normally after exactly one window")
        image_path = candidate_root / f"preview-{saved['png_sha256']}.png"
        with Image.open(image_path) as picture:
            rgb_sha = hashlib.sha256(picture.convert("RGB").tobytes()).hexdigest()
            dimensions = list(picture.size)
        if sha(image_path).lower() != saved["png_sha256"] or rgb_sha != saved["preview_report"]["rgb8_sha256"]:
            raise RuntimeError("candidate PNG failed byte/RGB checks")
        report.update(status="candidate_generated_human_review_pending", candidate_id=saved["sha256"],
            cache_root=str(candidate_root), image_path=str(image_path), image_dimensions=dimensions,
            preview_report=saved["preview_report"], committed_windows=1, paused=True,
            old_first_window_sha256=reference_window["committed"][0]["sha256"],
            new_first_window_sha256=window["committed"][0]["sha256"])
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
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--reserve-vram-gib", type=float, default=5.0)
    parser.add_argument("--min-free-mib", type=float, default=10000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=1800)
    run(parser.parse_args())
