"""One strictly serial, isolated, pinned-upstream H3 probe; never alters user workflows.

Only --confirm-run starts a server or model. Output directories are single-use. The
original upstream node is loaded from the isolated reference checkout, not copied
into the user's custom_nodes root. This tool never unloads Ollama or other apps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from unittest.mock import patch

import pynvml

try:
    from tools import run_nfe_resume_real_probe as probe
    from tools.run_face_refine_window_memory_matrix import MemorySampler, _atomic_json, _sha256_file
except ModuleNotFoundError:
    import run_nfe_resume_real_probe as probe
    from run_face_refine_window_memory_matrix import MemorySampler, _atomic_json, _sha256_file


ROOT = Path(__file__).resolve().parents[1]
REVISION = "27df0ff0538896dd494e3541c5a374cf2fe00aab"
REFERENCE = ROOT / "artifacts/upstream-h3-video-outpaint-27df0ff"


def build_prompt(source_preset="clean_512"):
    graph = json.loads((REFERENCE / "example_workflows/MiniMax H3 Video Outpaint - API.json").read_text(encoding="utf-8"))
    graph["1"]["inputs"]["file"] = "source_512x288_90f.mp4" if source_preset == "clean_512" else "source_640x384_90f.mp4"
    graph["6"]["inputs"].update(temporal_window_frames="auto", minimum_source_megapixels=0.1,
                                  generation_megapixels=0.5, frame_load_cap=90)
    graph["8"]["inputs"]["filename_prefix"] = "outpaint_upstream_stock20"
    return graph


def strict_silent_media(path, ffmpeg, ffprobe):
    result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=120)
    streams = json.loads(result.stdout)["streams"]
    if any(s["codec_type"] == "audio" for s in streams):
        raise RuntimeError("Silent source unexpectedly acquired an audio stream")
    exits = {}
    for label, mapping in (("video", ["-map", "0:v:0"]), ("joint", [])):
        checked = subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(path), *mapping, "-f", "null", "-"],
                                 capture_output=True, timeout=180)
        if checked.returncode:
            raise RuntimeError(f"Strict {label} decode failed")
        exits[label] = checked.returncode
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path),
            "streams": streams, "strict_decode_exit": exits, "audio_check": "not_applicable_silent_source"}


def run(args):
    root = args.run_root.resolve()
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**2
    revision = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain"], text=True).strip()
    prompt = build_prompt(args.source_preset)
    clean = args.source_preset == "clean_512"
    expected = {"width": 512 if clean else 640, "height": 672 if clean else 832, "frames": 90, "fps": "24/1"}
    required = [
        args.comfy_root / "models/diffusion_models" / prompt["4"]["inputs"]["unet_name"],
        args.comfy_root / "models/text_encoders" / prompt["3"]["inputs"]["clip_name"],
        args.comfy_root / "models/vae" / prompt["2"]["inputs"]["vae_name"],
        args.comfy_root / "models/vae" / prompt["7"]["inputs"]["vae_name"],
    ]
    checks = {"pinned_clean_reference": revision == REVISION and not dirty,
              "port_free": not probe.port_is_listening(args.host, args.port),
              "user_8188_not_running": not probe.port_is_listening("127.0.0.1", 8188),
              "initial_headroom": free >= args.min_free_mib,
              "models_present": all(p.is_file() for p in required),
              "ffmpeg_present": bool(args.ffmpeg and args.ffprobe)}
    kj_root = args.comfy_root / "custom_nodes/ComfyUI-KJNodes"
    kj_source = kj_root / "nodes/minimax_nodes.py"
    checks["reference_kj_memory_nodes_present"] = kj_source.is_file()
    report = {"schema": "t8.h3.video_outpaint.upstream_probe/v1", "checks": checks,
              "gpu_initial_free_mib": free, "upstream_commit": revision,
              "upstream_nodes_sha256": _sha256_file(REFERENCE / "nodes.py"),
              "prompt": prompt, "status": "preflight_ready" if all(checks.values()) else "preflight_not_ready",
              "models": [{"path": str(p), "bytes": p.stat().st_size if p.exists() else None} for p in required],
              "weight_full_hashes_verified": False, "human_quality_accepted": False,
              "reference_memory_nodes": {"repository": str(kj_root),
                  "source_sha256": _sha256_file(kj_source) if kj_source.is_file() else None,
                  "nodes": ["MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward"],
                  "note": "Original reference graph uses external KJ memory patches; not a bare unpatched MODEL comparison"},
              "source_preset": args.source_preset,
              "source_preprocess": ("24fps/90 frames, isotropic 512x288 resize with no artificial padding; silent reference" if clean else
                  "24fps/90 frames, isotropic 640x360 resize then 12px top/bottom padding; historical diagnostic only"),
              "expected_output": expected,
              "actual_canvas_note": f"Native canvas is {expected['width']*expected['height']} pixels with its 0.5MP setting; not a T8 budget test"}
    if not args.confirm_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"Probe preflight not ready: {checks}")
    if root.exists():
        raise FileExistsError("Probe run_root must be new; inspect existing run before any retry")
    root.mkdir(parents=True)
    (root / "input").mkdir()
    extra_paths = root / "reference_paths.yaml"
    extra_paths.write_text(json.dumps({"outpaint_reference": {"custom_nodes": str(REFERENCE.parent)}}), encoding="utf-8")
    _atomic_json(root / "report.json", report)
    source = root / "input" / prompt["1"]["inputs"]["file"]
    subprocess.run([args.ffmpeg, "-v", "error", "-n", "-i", str(REFERENCE / "examples/big_buck_bunny_source.mp4"),
                    "-vf", ("scale=512:288:flags=lanczos,setsar=1" if clean else "scale=640:360:flags=lanczos,pad=640:384:0:12,setsar=1"), "-frames:v", "90", "-an",
                    "-c:v", "libx264", "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", str(source)],
                   check=True, timeout=120)
    report["source_sha256"] = _sha256_file(source)
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        result[result.index("--input-directory") + 1] = str(root / "input")
        result += ["--extra-model-paths-config", str(extra_paths)]
        return result

    args.extra_whitelist_custom_nodes = (REFERENCE.name, "ComfyUI-KJNodes")
    args.lowvram = False
    server = probe.IsolatedServer(args, root, "upstream")
    telemetry = None
    started = time.monotonic()
    try:
        with patch.object(probe, "_server_command", command):
            server.start()
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
            available = json.loads(response.read())
        missing = sorted({node["class_type"] for node in prompt.values()} - set(available))
        report["missing_reference_node_types"] = missing
        if missing:
            raise RuntimeError(f"Reference graph nodes are unavailable before GPU submission: {missing}")
        if _sha256_file(kj_source) != report["reference_memory_nodes"]["source_sha256"]:
            raise RuntimeError("KJ memory-node source changed while starting the isolated server")
        report.update(status="running", server_pid=server.process.pid)
        _atomic_json(root / "report.json", report)
        telemetry = MemorySampler(server.process.pid, 0.1)
        telemetry.start()
        execution = asyncio.run(probe.submit_prompt(server=f"http://{args.host}:{args.port}", prompt=prompt,
                                                    timeout_seconds=args.timeout))
        report["execution"] = execution
        if (execution.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError(f"Upstream execution failed: {execution.get('terminal')}")
        outputs = list((root / "output").rglob("outpaint_upstream_stock20*.mp4"))
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one result, found {len(outputs)}")
        report["media"] = strict_silent_media(outputs[0], args.ffmpeg, args.ffprobe)
        streams = report["media"]["streams"]
        video = next(s for s in streams if s["codec_type"] == "video")
        if (int(video["width"]), int(video["height"]), int(video["nb_read_frames"]), video["avg_frame_rate"]) != (expected["width"], expected["height"], 90, "24/1"):
            raise RuntimeError("Upstream output geometry/timeline does not match expected actual canvas")
        if any(s["codec_type"] == "audio" for s in streams):
            raise RuntimeError("Silent source unexpectedly acquired an audio stream")
        report["status"] = "mechanical_pass_not_human_acceptance"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if telemetry is not None:
                telemetry.stop()
                telemetry.write_csv(root / "telemetry.csv")
                report["memory"] = telemetry.summary()
                report["observed_512mib_margin"] = report["memory"]["min_gpu_free_mib"] >= 512
                if report["status"] == "mechanical_pass_not_human_acceptance" and not report["observed_512mib_margin"]:
                    report["status"] = "media_pass_memory_margin_failed"
        except Exception as error:
            report["telemetry_error"] = f"{type(error).__name__}: {error}"
            report["observed_512mib_margin"] = None
            if report["status"] == "mechanical_pass_not_human_acceptance":
                report["status"] = "media_pass_memory_evidence_incomplete"
        finally:
            try:
                server.stop()
            finally:
                report["elapsed_seconds"] = round(time.monotonic() - started, 3)
                _atomic_json(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--source-preset", choices=("clean_512", "padded_640"), default="clean_512")
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts/outpaint-upstream-p0-20260906")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=ROOT.parents[2] / "python/python.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--reserve-vram-gib", type=float, default=2.0)
    parser.add_argument("--min-free-mib", type=float, default=10000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
