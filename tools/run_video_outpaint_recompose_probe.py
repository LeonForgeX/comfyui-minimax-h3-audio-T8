"""Recompose one completed selected outpaint cache without loading the H3 model.

The prior cache and failed/published media are read-only evidence. This probe
copies the complete cache to a new artifact root, loads only the video VAE, and
publishes a new file through the current composition and media contracts.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from unittest.mock import patch

import pynvml

try:
    from tools import run_video_outpaint_t8_probe as t8
    from tools.build_video_outpaint_candidate_workflows import _closure
except ModuleNotFoundError:
    import run_video_outpaint_t8_probe as t8
    from build_video_outpaint_candidate_workflows import _closure


ROOT = Path(__file__).resolve().parents[1]


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be an explicit lowercase SHA256 identifier")
    return value


def build_recompose_prompt(prior, *, candidate_name, selection_id, output_name="selected_test32_safe"):
    graph = deepcopy(prior["prompt"])
    by_kind = {}
    for kind in ("LoadVideo", "MiniMaxH3VideoOutpaintPlanT8", "VAELoader"):
        found = [key for key, node in graph.items() if node["class_type"] == kind]
        if len(found) != 1:
            raise ValueError(f"prior prompt needs exactly one {kind}")
        by_kind[kind] = found[0]
    reloads = [node for node in graph.values()
               if node["class_type"] == "MiniMaxH3VideoOutpaintLoadPreparedT8"]
    if len(reloads) != 1:
        raise ValueError("prior prompt needs exactly one prepared-cache reload")
    reserved = {"reload_completed_prepared", "load_completed_selection", "save_recomposed"}
    if set(graph) & reserved:
        raise ValueError("prior prompt uses reserved recompose node IDs")
    _digest(selection_id, "selection_id")
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        raise ValueError("candidate_name is required")
    plan_id = by_kind["MiniMaxH3VideoOutpaintPlanT8"]
    vae_id = by_kind["VAELoader"]
    graph["reload_completed_prepared"] = {
        "class_type": "MiniMaxH3VideoOutpaintLoadPreparedT8",
        "inputs": {"plan": [plan_id, 0], "run_name": reloads[0]["inputs"]["run_name"]},
    }
    graph["load_completed_selection"] = {
        "class_type": "MiniMaxH3VideoOutpaintLoadCompletedSelectionT8",
        "inputs": {"prepared": ["reload_completed_prepared", 0], "candidate_name": candidate_name,
                   "selection_id": selection_id},
    }
    graph["save_recomposed"] = {
        "class_type": "MiniMaxH3VideoOutpaintComposeCandidateT8",
        "inputs": {"sampled": ["load_completed_selection", 0], "video_vae": [vae_id, 0],
                   "output_name": output_name},
    }
    return _closure(graph, ["save_recomposed"])


def strict_decode_matrix(path, ffmpeg, *, attempts=20):
    rows = []
    for threads in (1, 4):
        failures = []
        for attempt in range(1, attempts + 1):
            checked = subprocess.run(
                [ffmpeg, "-v", "error", "-threads", str(threads), "-xerror", "-err_detect", "explode",
                 "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False,
            )
            detail = (checked.stderr or "").strip()
            if checked.returncode or detail:
                failures.append({"attempt": attempt, "exit": checked.returncode, "error": detail[-2000:]})
        rows.append({"threads": threads, "attempts": attempts, "failures": failures})
    if any(row["failures"] for row in rows):
        raise RuntimeError(f"fixed-thread decode matrix failed: {rows}")
    return rows


def run(args):
    probe, sha, atomic = t8.upstream.probe, t8.upstream._sha256_file, t8.upstream._atomic_json
    prior_root, root = args.prior_root.resolve(strict=True), args.run_root.resolve()
    prior_root.relative_to(ROOT / "artifacts")
    root.relative_to(ROOT / "artifacts")
    prior_path = prior_root / "report.json"
    prior = json.loads(prior_path.read_bytes())
    t8.verify_prior_server_inactive(prior, prior_path)
    if prior.get("committed_windows") != 22 or (prior.get("execution", {}).get("terminal") or {}).get("type") != "execution_success":
        raise ValueError("prior selected run did not complete all real sampling windows")
    candidate_root = Path(prior["cache_root"]).resolve(strict=True)
    candidate_root.relative_to(prior_root / "output/T8_H3_Outpaint_Cache")
    prior_cache = candidate_root.parents[1]
    manifest = candidate_root / "outpaint_windows.json"
    state = json.loads(manifest.read_bytes())
    if state.get("status") != "sampled" or len(state.get("committed", [])) != 22:
        raise ValueError("recompose needs one fully sampled 22-window candidate")
    selection_id = _digest(args.selection_id, "selection_id")
    if not (candidate_root / f"selection-{selection_id}.json").is_file():
        raise FileNotFoundError("the explicit selection archive is missing")
    graph = build_recompose_prompt(prior, candidate_name=candidate_root.name,
                                   selection_id=selection_id, output_name=args.output_name)
    source_node = next(node for node in graph.values() if node["class_type"] == "LoadVideo")
    source = prior_root / "input" / source_node["inputs"]["file"]
    vae_node = next(node for node in graph.values() if node["class_type"] == "VAELoader")
    vae_path = args.comfy_root / "models/vae" / vae_node["inputs"]["vae_name"]
    old_vae = next((item for item in prior["models"] if Path(item["path"]).resolve() == vae_path.resolve()), None)
    actual_vae = {"path": str(vae_path), "bytes": vae_path.stat().st_size, "sha256": sha(vae_path)}
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**2
    checks = {
        "new_artifact_root": not root.exists(),
        "source_unchanged": source.is_file() and sha(source) == prior["source_sha256"],
        "sampled_manifest_complete": state["status"] == "sampled" and len(state["committed"]) == 22,
        "selection_explicit": (candidate_root / f"selection-{selection_id}.json").is_file(),
        "video_vae_unchanged": old_vae == actual_vae,
        "gpu_headroom": free >= args.min_free_mib,
        "port_free": not probe.port_is_listening(args.host, args.port),
        "no_other_gpu_server": not any(probe.port_is_listening("127.0.0.1", port)
                                       for port in (8188, 8190, 8193, 8194)),
        "graph_has_no_model_or_sampler": not any(node["class_type"] in {
            "UNETLoader", "MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward",
            "MiniMaxH3VideoOutpaintContinueCandidateT8", "MiniMaxH3VideoOutpaintSampleT8",
        } for node in graph.values()),
    }
    report = {
        "scope": "completed_selected_cache_recompose_no_diffusion_sampling",
        "status": "preflight",
        "checks": checks,
        "prompt": graph,
        "prior_root": str(prior_root),
        "prior_manifest_sha256": sha(manifest),
        "selection_id": selection_id,
        "candidate_id": prior["candidate_id"],
        "selected_first_frame_rgb8_sha256": prior["selected_first_frame_rgb8_sha256"],
        "source_sha256": prior["source_sha256"],
        "plan_sha256": prior["plan_sha256"],
        "models": [actual_vae],
        "gpu_initial_free_mib": free,
        "human_acceptance": False,
    }
    if not args.confirm_run:
        print(json.dumps(report, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"recompose preflight failed: {checks}")
    root.mkdir(parents=True, exist_ok=False)
    atomic(root / "report.json", report)
    (root / "input").mkdir()
    shutil.copyfile(source, root / "input" / source.name)
    copied_cache = root / "output/T8_H3_Outpaint_Cache" / prior_cache.name
    shutil.copytree(prior_cache, copied_cache)
    copied_candidate = copied_cache / "candidates" / candidate_root.name
    if sha(copied_candidate / "outpaint_windows.json") != report["prior_manifest_sha256"]:
        raise ValueError("copied sampled manifest changed")
    runtime = [*ROOT.glob("video_outpaint*.py"), *ROOT.glob("nodes_video_outpaint*.py"), Path(__file__)]
    report["implementation_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in runtime}
    extra = root / "draft_paths.yaml"
    atomic(extra, {"recompose_test_only": {"custom_nodes": str(ROOT / "tools")}})
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        first, last = result.index("--whitelist-custom-nodes") + 1, result.index("--input-directory")
        result[first:last] = ["outpaint_candidate_extension"]
        result[result.index("--input-directory") + 1] = str(root / "input")
        return result + ["--extra-model-paths-config", str(extra)]

    server = probe.IsolatedServer(args, root, "recompose")
    telemetry, started = None, time.monotonic()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            raise RuntimeError("parent disables CUDA")
        with patch.object(probe, "_server_command", command):
            server.start()
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
            info = json.load(response)
        used = {node["class_type"] for node in graph.values()}
        if used - set(info):
            raise RuntimeError(f"missing recompose workflow nodes: {used - set(info)}")
        atomic(root / "schema_snapshot.json", {kind: info[kind] for kind in used})
        report.update(status="running", server_pid=server.process.pid, copied_cache_root=str(copied_candidate))
        atomic(root / "report.json", report)
        telemetry = t8.LiveMemorySampler(server.process.pid, root / "telemetry.live.jsonl")
        telemetry.start()
        result = asyncio.run(probe.submit_prompt(server=f"http://{args.host}:{args.port}", prompt=graph,
                                                 timeout_seconds=args.timeout))
        report["execution"] = result
        if (result.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("recompose execution failed; inspect retained evidence")
        outputs = list((root / "output/T8_H3_Outpaint").glob("*.mp4"))
        if len(outputs) != 1:
            raise RuntimeError("recompose did not publish exactly one output")
        output = outputs[0]
        sidecar = output.with_suffix(".mp4.outpaint.json")
        delivery = t8.cases.verify_delivery(sidecar, sha(output), prior["source_sha256"], prior["plan_sha256"])
        if (delivery.get("selected_first_frame_rgb8_sha256") != report["selected_first_frame_rgb8_sha256"]
                or delivery.get("selected_first_frame_verified_before_encoding") is not True
                or delivery.get("encoder_profile") != "baseline"
                or "keyint=1" not in delivery.get("encoder_x264_params", "")):
            raise RuntimeError("safe encoder or selected first-frame receipt is missing")
        streams = json.loads(subprocess.check_output(
            [args.ffprobe, "-v", "error", "-show_streams", "-of", "json", str(source)], timeout=60
        ))["streams"]
        audio_tracks = sum(stream["codec_type"] == "audio" for stream in streams)
        report["media"] = t8.cases.strict_media(output, args.ffmpeg, args.ffprobe, audio_tracks, sha)
        report["decode_matrix"] = strict_decode_matrix(output, args.ffmpeg, attempts=20)
        report["delivery"] = {"path": str(sidecar), "report": delivery}
        if any(sha(ROOT / name) != digest for name, digest in report["implementation_sha256"].items()):
            raise ValueError("implementation changed during recompose test")
        report.update(status="media_pass_human_review_pending", output_path=str(output))
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
            report["prior_manifest_unchanged"] = sha(manifest) == report["prior_manifest_sha256"]
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--output-name", default="selected_test32_safe")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=ROOT.parents[2] / "python/python.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8195)
    parser.add_argument("--reserve-vram-gib", type=float, default=3.0)
    parser.add_argument("--min-free-mib", type=float, default=6000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    run(parser.parse_args())
