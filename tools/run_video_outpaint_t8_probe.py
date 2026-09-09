"""One serial T8 draft-node GPU probe; dry-run unless --confirm-run is supplied.

Uses the already hash-bound clean reference input without rescaling. Draft nodes
are registered only in an owned temporary Comfy server. No user workflows change.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from unittest.mock import patch

import pynvml
import psutil

try:
    from tools import run_video_outpaint_upstream_probe as upstream
    from tools import outpaint_probe_cases as cases
except ModuleNotFoundError:
    import run_video_outpaint_upstream_probe as upstream
    import outpaint_probe_cases as cases


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "D1FBBF84C937AF6F853F1D1726244364EDF65C970C4C32803C72E21183C8D9B6"
KJ_SHA = "C371576B1BB31A2F518BDB4CEDA43CB10B20338F0C9D68F99ED1BE76CE06478F"


class LiveMemorySampler(upstream.MemorySampler):
    """Flush every observation so an application exit does not erase all samples."""
    def __init__(self, pid, journal):
        super().__init__(pid, 0.1)
        self.journal = Path(journal)
        self.journal_error = None

    def _run(self):
        try:
            with self.journal.open("x", encoding="utf-8", buffering=1) as handle:
                class Rows(list):
                    def append(self, row):
                        handle.write(json.dumps(row)+"\n")
                        handle.flush()
                        super().append(row)
                self.rows = Rows()
                super()._run()
        except Exception as error:
            self.journal_error = str(error)
            self._stop.set()


def build_prompt(*, resume=False, native_noise=False, case=None, head_chunks=4, ffn_chunks=4,
                 geometry_align=False, source_mode="joint_decode"):
    cases.pixel_receipt_module().validate_source_mode(source_mode)
    if not isinstance(geometry_align, bool):
        raise ValueError("geometry_align must be boolean")
    if not 1 <= head_chunks <= 56:
        raise ValueError("head_chunks must be in [1, 56]")
    if not 1 <= ffn_chunks <= 64:
        raise ValueError("ffn_chunks must be in [1, 64]")
    graph = upstream.build_prompt("clean_512")
    graph["5"]["inputs"]["head_chunks"] = head_chunks
    graph["9"]["inputs"]["chunks"] = ffn_chunks
    # Keep the reference loaders and KJ pair. The four new stages replace the
    # all-in-one upstream node. Compose is now an output node itself, so there
    # is no redundant native SaveVideo re-encode after verified publication.
    del graph["6"]
    graph["10"] = {"class_type": "MiniMaxH3VideoOutpaintPlanT8", "inputs": {
        "source_video": ["1", 0], "aspect": "custom", "left": 0, "right": 0, "top": 192, "bottom": 192,
        "anchor_x": 0.5, "anchor_y": 0.5, "generation_megapixels": 0.5, "window_frames": "90", "cut_frames_json": "[]"}}
    graph["11"] = {"class_type": "MiniMaxH3VideoOutpaintPrepareT8", "inputs": {
        "plan": ["10", 0], "clip": ["3", 0], "video_vae": ["2", 0], "audio_vae": ["7", 0],
        "prompt": "", "shot_prompts_json": "[]", "run_name": "t8_clean90", "audio_track": 0,
        "audio_block_tokens": 64, "resume_audio": resume}}
    graph["12"] = {"class_type": "MiniMaxH3VideoOutpaintSampleT8", "inputs": {
        "model": ["9", 0], "prepared": ["11", 0], "seed": 20260808, "steps": 20, "resume": resume}}
    if native_noise:
        graph["12"]["inputs"]["noise_algorithm"] = "t8.outpaint.native_cpu_noise/v1"
    graph["13"] = {"class_type": "MiniMaxH3VideoOutpaintComposeT8", "inputs": {
        "sampled": ["12", 0], "video_vae": ["2", 0], "output_name": "t8_clean90", "color_match": False,
        "source_mode": source_mode}}
    if geometry_align:
        graph["13"]["inputs"]["geometry_align"] = True
    del graph["8"]
    if case is not None:
        request = case["plan"]["request"]
        graph["1"]["inputs"]["file"] = "source.mp4"
        graph["10"]["inputs"].update({key: request[key] for key in (
            "aspect", "left", "right", "top", "bottom", "anchor_x", "anchor_y", "generation_megapixels")})
        graph["10"]["inputs"].update(window_frames=str(request["window_frames"]), cut_frames_json=json.dumps(request["cut_frames"]))
        graph["11"]["inputs"].update(prompt=case["prompt"], shot_prompts_json=json.dumps(case["shot_prompts"]), run_name="t8_case")
        graph["13"]["inputs"].update(output_name="t8_case", color_match=case["color_match"])
    return graph


def verify_prior_server_inactive(report, report_path):
    """A recycled Windows PID is not the previous server; never terminate it."""
    pid = report.get("server_pid")
    if pid is None:
        return
    try:
        process = psutil.Process(int(pid))
        created = process.create_time()
    except psutil.NoSuchProcess:
        return
    # Only a terminal report predating the current PID's creation is positive
    # evidence of PID reuse. AccessDenied/unknown/live identities still stop us.
    terminal = report.get("status") in {"failed", "media_pass_human_review_pending",
        "media_pass_memory_margin_failed", "media_pass_memory_evidence_incomplete"}
    if terminal and created > Path(report_path).stat().st_mtime + 1:
        return
    raise RuntimeError("prior server PID is still present without proof of exit/reuse; do not restart or copy its working cache")


def checked_resume_cache(prior_root, sha, source_sha=SOURCE_SHA, *, compose_only=False, expected_plan_sha=None):
    """Read-only validation; never turn a live/stale report alone into a restart."""
    prior = Path(prior_root).resolve(strict=True)
    prior.relative_to(ROOT / "artifacts")
    report = json.loads((prior / "report.json").read_text(encoding="utf-8"))
    verify_prior_server_inactive(report, prior / "report.json")
    if report.get("source_sha256") != source_sha:
        raise ValueError("resume source differs from the clean reference input")
    for path, digest in report.get("implementation_sha256", {}).items():
        if path.replace("\\", "/").startswith("tools/"):
            continue  # Probe orchestration may add recovery; runtime identities may not change.
        if compose_only and path.replace("\\", "/") in {
            "video_outpaint_compose.py", "video_outpaint_media.py", "video_outpaint_delivery.py", "video_outpaint_packet_mux.py"
        }:
            continue  # Explicit completed-latent compose retry, never a sampling resume.
        source = (ROOT / path).resolve(strict=True)
        source.relative_to(ROOT)
        if sha(source) != digest:
            raise ValueError(f"runtime changed since the prior probe: {path}")
    cache = prior / "output/T8_H3_Outpaint_Cache"
    if not cache.is_dir() or any(path.is_symlink() for path in cache.rglob("*")):
        raise ValueError("resume requires a regular existing outpaint cache, without symlinks")
    if compose_only:
        manifests = list(cache.rglob("outpaint_windows.json"))
        if len(manifests) != 1:
            raise ValueError("compose-only retry needs exactly one completed sampling manifest")
        state = json.loads(manifests[0].read_text(encoding="utf-8"))
        digest = state.pop("sha256", None)
        if digest != hashlib.sha256(cases.plan_module().canonical(state).encode()).hexdigest():
            raise ValueError("compose-only sampling manifest integrity mismatch")
        if (state.get("status") != "sampled" or not state.get("committed")
                or (expected_plan_sha and state["identity"]["plan_sha256"] != expected_plan_sha)):
            raise ValueError("compose-only retry requires fully sampled matching-plan windows")
        for record in state["committed"]:
            digest = record.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("compose-only sampling asset hash malformed")
            asset = manifests[0].parent / f"window-{digest}.safetensors"
            if not asset.is_file() or asset.stat().st_size != record["bytes"] or sha(asset).lower() != digest:
                raise ValueError("compose-only sampling asset integrity mismatch")
    return cache, report


def run(args):
    probe, sha, atomic = upstream.probe, upstream._sha256_file, upstream._atomic_json
    root = args.run_root.resolve()
    case = cases.load_case(args.case_file, args.ffprobe, sha) if args.case_file else None
    source = case["source_path"] if case else ROOT / "artifacts/outpaint-upstream-clean-p0-20260906/input/source_512x288_90f.mp4"
    source_sha = case["source_sha256"] if case else SOURCE_SHA
    if args.resume_compose and not args.resume_from:
        raise ValueError("--resume-compose requires --resume-from")
    capture_rgb = getattr(args, "capture_compose_rgb", False)
    if capture_rgb and not args.resume_compose:
        raise ValueError("--capture-compose-rgb requires --resume-compose")
    expected = ({key: case["plan"]["output"][key] for key in ("width", "height", "frames", "fps")}
                if case else {"width": 512, "height": 672, "frames": 90, "fps": "24/1"})
    kj_source = args.comfy_root / "custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py"
    graph = build_prompt(
        resume=args.resume_from is not None,
        native_noise=args.native_noise,
        case=case,
        head_chunks=args.head_chunks,
        ffn_chunks=args.ffn_chunks,
        geometry_align=getattr(args, "geometry_align", False),
        source_mode=getattr(args, "source_mode", "joint_decode"),
    )
    prior_cache, prior_report = checked_resume_cache(args.resume_from, sha, source_sha,
        compose_only=args.resume_compose, expected_plan_sha=case["plan"]["plan_sha256"] if case else None) if args.resume_from else (None, None)
    models = [args.comfy_root / "models" / directory / graph[node]["inputs"][key] for directory, node, key in (
        ("diffusion_models", "4", "unet_name"), ("text_encoders", "3", "clip_name"),
        ("vae", "2", "vae_name"), ("vae", "7", "vae_name"))]
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free/1024**2
    checks = {"source_matches_bound_input": source.is_file() and sha(source) == source_sha,
        "kj_source_matches_audited_adapter": cases.is_audited_kj_source(kj_source),
        "initial_gpu_headroom": free >= args.min_free_mib,
        "models_present": all(path.is_file() for path in models),
        "port_free": not probe.port_is_listening(args.host, args.port),
        "user_8188_not_running": not probe.port_is_listening("127.0.0.1", 8188),
        "upstream_probe_not_running": not probe.port_is_listening("127.0.0.1", 8190),
        "ffmpeg_present": bool(args.ffmpeg and args.ffprobe)}
    report = {"schema": "t8.h3.outpaint.draft_gpu_probe/v1", "status": "preflight_ready" if all(checks.values()) else "preflight_not_ready",
        "checks": checks, "gpu_initial_free_mib": free, "prompt": graph, "source_sha256": source_sha,
        "expected_output": expected,
        "kj_memory_profile": {
            "head_chunks": args.head_chunks,
            "ffn_chunks": args.ffn_chunks,
            "ffn_seq_threshold": graph["9"]["inputs"]["seq_threshold"],
        },
        "reference_differences": [
            ("Native FP32 CPU video-then-audio noise order, streamed in bounded blocks and unit-tested bit-exact on this geometry; no arbitrary hook RNG parity claim"
             if args.native_noise else "T8 coordinate noise v1 is not upstream whole-tensor RNG order despite the same seed"),
            f"{args.reserve_vram_gib}GiB reserved VRAM; the original clean reference used 2GiB; no same-memory or speed-parity claim",
            ("bounded VAE/source preparation; joint VAE reconstruction without RGB paste-back or source-boundary postprocessing"
             if graph["13"]["inputs"]["source_mode"] == "joint_decode" else
             "bounded VAE/source preparation and original RGB paste-back; boundary color enabled="
             + str(graph["13"]["inputs"]["color_match"]))],
        "human_acceptance": False, "perceptual_or_numerical_parity_claimed": False}
    if capture_rgb:
        report["diagnostic_rgb_capture"] = {"path": str(root / "diagnostic_rgb/frames.rgb"),
            "expected_bytes": expected["width"] * expected["height"] * expected["frames"] * 3,
            "scope": "explicit composition diagnostic; not a normal node feature or delivery"}
    if case:
        report["case"] = {**case, "source_path": str(source)}
        report["reference_differences"].append("Separate audio/window/material validation case, not the same-input upstream clean90 pair")
    if prior_report is not None:
        report["resume"] = {"from": str(args.resume_from.resolve()), "prior_report_sha256": sha(args.resume_from / "report.json"),
            "prior_server_process_verified_inactive": True,
            "policy": ("explicit completed-latent compose retry; only composition/media/delivery changes are permitted"
                       if args.resume_compose else "copy verified caches into a new run; restart only the unfinished sampling window, not mid-step")}
    if not args.confirm_run:
        print(json.dumps(report, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"T8 GPU probe is not ready: {checks}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "input").mkdir()
    atomic(root / "report.json", report)
    copied_source = root / "input" / graph["1"]["inputs"]["file"]
    shutil.copyfile(source, copied_source)
    if sha(copied_source) != source_sha:
        raise ValueError("copied reference source changed")
    report["models"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in models]
    if prior_report is not None:
        if report["models"] != prior_report.get("models"):
            raise ValueError("actual model files changed since the prior run")
        shutil.copytree(prior_cache, root / "output/T8_H3_Outpaint_Cache")
    report["implementation_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(
        [*ROOT.glob("video_outpaint*.py"), ROOT / "nodes_video_outpaint.py", Path(__file__),
         ROOT / "tools/outpaint_probe_cases.py", ROOT / "tools/outpaint_probe_extension/__init__.py"])}
    extra = root / "draft_paths.yaml"
    atomic(extra, {"outpaint_test_only": {"custom_nodes": str(ROOT / "tools")}})
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        result[result.index("--input-directory")+1] = str(root / "input")
        return result+["--extra-model-paths-config", str(extra)]

    args.extra_whitelist_custom_nodes = ("outpaint_probe_extension", "ComfyUI-KJNodes")
    args.lowvram = False
    server = probe.IsolatedServer(args, root, "t8")
    telemetry = None
    started = time.monotonic()
    try:
        # Only the owned test server inherits this flag. Never mutate OS/user configuration.
        environment = {**os.environ}
        environment.pop("T8_OUTPAINT_CAPTURE_RGB", None)
        if capture_rgb:
            environment["T8_OUTPAINT_CAPTURE_RGB"] = report["diagnostic_rgb_capture"]["path"]
        with patch.object(probe, "_server_command", command), patch.dict(os.environ, environment, clear=True):
            server.start()
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
            available = json.loads(response.read())
        missing = sorted({node["class_type"] for node in graph.values()}-set(available))
        if missing:
            raise RuntimeError(f"Draft graph nodes are missing before submission: {missing}")
        if any(sha(ROOT / p) != value for p, value in report["implementation_sha256"].items()):
            raise ValueError("T8 implementation changed while starting the server")
        report.update(status="running", server_pid=server.process.pid)
        atomic(root / "report.json", report)
        telemetry = LiveMemorySampler(server.process.pid, root / "telemetry.live.jsonl")
        telemetry.start()
        result = asyncio.run(probe.submit_prompt(server=f"http://{args.host}:{args.port}", prompt=graph,
                                                timeout_seconds=args.timeout))
        report["execution"] = result
        if (result.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError(f"Draft execution failed: {result.get('terminal')}")
        outputs = list((root / "output/T8_H3_Outpaint").glob("*.mp4"))
        if len(outputs) != 1:
            raise RuntimeError("T8 compose did not publish exactly one authoritative output")
        report["media"] = cases.strict_media(outputs[0], args.ffmpeg, args.ffprobe, case["audio_tracks"] if case else 0, sha)
        sidecar = outputs[0].with_suffix(".mp4.outpaint.json")
        delivery = cases.verify_delivery(sidecar, report["media"]["sha256"], source_sha,
                                         case["plan"]["plan_sha256"] if case else None,
                                         expected_plan=case["plan"] if case else None,
                                         expected_source_mode=graph["13"]["inputs"]["source_mode"])
        report["delivery"] = {"path": str(sidecar), "file_sha256": sha(sidecar), "report": delivery}
        video = next(s for s in report["media"]["streams"] if s["codec_type"] == "video")
        if (int(video["width"]), int(video["height"]), int(video["nb_read_frames"]), video["avg_frame_rate"]) != tuple(expected[k] for k in ("width", "height", "frames", "fps")):
            raise RuntimeError("T8 output geometry/timeline differs from its plan")
        if any(sha(ROOT / p) != value for p, value in report["implementation_sha256"].items()):
            raise ValueError("T8 implementation changed during execution")
        report["status"] = "media_pass_human_review_pending"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if telemetry is not None:
                telemetry.stop()
                if telemetry.journal_error:
                    raise RuntimeError(f"live telemetry journal failed: {telemetry.journal_error}")
                telemetry.write_csv(root / "telemetry.csv")
                report["memory"] = telemetry.summary()
                report["observed_512mib_margin"] = report["memory"]["min_gpu_free_mib"] >= 512
                if report["status"] == "media_pass_human_review_pending" and not report["observed_512mib_margin"]:
                    report["status"] = "media_pass_memory_margin_failed"
        except Exception as error:
            report["telemetry_error"] = str(error)
            if report["status"] == "media_pass_human_review_pending":
                report["status"] = "media_pass_memory_evidence_incomplete"
        finally:
            server.stop()
            report["elapsed_seconds"] = round(time.monotonic()-started, 3)
            atomic(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--native-noise", action="store_true", help="Explicit bounded replay of the reference CPU initial noise stream")
    parser.add_argument("--geometry-align", action="store_true", help="Enable the exact-source geometric finishing path in the actual Compose node")
    parser.add_argument("--source-mode", choices=("joint_decode", "preserve_source"), default="joint_decode",
                        help="Explicit output policy, verified in the final pixel receipt")
    parser.add_argument("--case-file", type=Path, help="Hash-bound separate material/geometry/audio case; no source mutation")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Copy an interrupted probe's verified caches into a new run; old results remain untouched")
    parser.add_argument("--resume-compose", action="store_true", help="Reuse fully sampled matching latents after composition-only changes")
    parser.add_argument("--capture-compose-rgb", action="store_true", help="Explicitly retain raw encoder input on disk for a compose-only diagnostic")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=ROOT.parents[2] / "python/python.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8191)
    parser.add_argument("--reserve-vram-gib", type=float, default=3.0)
    parser.add_argument("--head-chunks", type=int, default=4)
    parser.add_argument("--ffn-chunks", type=int, default=4)
    parser.add_argument("--min-free-mib", type=float, default=10000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    run(parser.parse_args())
