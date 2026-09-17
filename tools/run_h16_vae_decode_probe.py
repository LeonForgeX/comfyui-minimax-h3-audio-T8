"""Decode the same saved H3 latent with one native VAE; no diffusion sampling.

Run FP16 and official INT8 in separate serial processes. The ordinary Core VAE
loader and decode/decode_tiled entry points are used. No production defaults are
changed. Float output is retained for a separate numerical comparison; preview
encoding copies the complete original audio stream without -shortest or gain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402


def write(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


@contextmanager
def resource_monitor(reader, guard, destination):
    stopped, failure = threading.Event(), []

    def observe():
        with destination.open("x", encoding="utf8") as stream:
            while not stopped.is_set():
                try:
                    sample = reader.sample()
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    reason = guard.observe(sample)
                    if reason:
                        failure.append(reason)
                        return
                except Exception as error:
                    failure.append(str(error))
                    return
                stopped.wait(.5)

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    def check():
        if failure:
            raise RuntimeError("Resource monitor: " + failure[0])
    try:
        yield check
    finally:
        stopped.set()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Resource monitor did not stop")


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def expected_shape(value):
    frames = {1: 1, 2: 5, 7: 22, 22: 73}[value.shape[2]]
    return frames, value.shape[3]*16, value.shape[4]*16, 3


def audit_preview(path, source, expected_frames, width, height):
    """Check complete decoding and copied audio packet payloads, not just a mux exit code."""
    def probe(filename, *extra):
        result = subprocess.run(["ffprobe", "-v", "error", *extra, "-of", "json", str(filename)],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    streams = probe(path, "-count_frames", "-show_streams")["streams"]
    video = next(item for item in streams if item["codec_type"] == "video")
    actual = [video["width"], video["height"], int(video["nb_read_frames"]), video["avg_frame_rate"]]
    if actual != [width, height, expected_frames, "24/1"] or video["codec_name"] != "h264":
        raise RuntimeError("Preview canvas, frame count, FPS or codec differs from decoded tensor")
    for selection in ("0:v:0", "0:a:0"):
        result = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
            "-i", str(path), "-map", selection, "-f", "null", "-"], capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            raise RuntimeError("Preview strict decode failed: " + result.stderr)
    def packets(filename):
        result = probe(filename, "-select_streams", "a:0", "-show_packets", "-show_data_hash", "sha256")
        return [item["data_hash"] for item in result["packets"]]
    original, copied = packets(source), packets(path)
    if not original or original != copied:
        raise RuntimeError("Preview changed source audio packet payloads")
    return {"strict_decode": True, "geometry": actual, "audio_packet_count": len(original),
            "source_audio_payloads_identical": True, "sha256": file_identity(path)["sha256"]}


def preview(images, source, output):
    import torch
    height, width = images.shape[1:3]
    command = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "24", "-i", "pipe:0", "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(output)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                          creationflags=flags) as process:
        try:
            for frame in images:
                process.stdin.write((frame.clamp(0, 1)*255).round().to(torch.uint8).numpy().tobytes())
            process.stdin.close()
            error = process.stderr.read().decode("utf8", errors="replace")
            if process.wait(timeout=120) or error.strip():
                raise RuntimeError("Preview encode failed: " + error)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Fresh task-owned artifact directory required")
    capture = json.loads(args.capture.read_text(encoding="utf8"))
    latent = Path(capture["files"]["video"]["path"]).resolve(strict=True)
    if not latent.is_relative_to(project / "artifacts"):
        raise ValueError("Use this checkout's captured actual sampler output")
    model = (args.core / "models/vae" / args.model).resolve(strict=True)
    model_identity, latent_identity = file_identity(model), file_identity(latent)
    if model_identity["sha256"] != args.model_sha256 or latent_identity["sha256"] != capture["files"]["video"]["sha256"]:
        raise ValueError("Model or actual sampler capture identity mismatch")
    root.mkdir(parents=True)
    result = {"status": "incomplete", "model": model_identity, "latent": latent_identity,
        "probe_source": file_identity(Path(__file__)),
        "source_video": file_identity(args.source_video), "quality_accepted": False, "diffusion_forwards": 0}
    guard = ResourceGuard()
    lease = args.core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader:
            startup = reader.sample()
            if (startup["gpu_free_bytes"] < guard.policy.startup_free_gpu_bytes or
                    startup["ram_available_bytes"] < guard.policy.startup_free_ram_bytes):
                raise RuntimeError("Insufficient startup resource headroom for isolated VAE qualification")
            sys.path.insert(0, str(args.core))
            sys.argv = [sys.argv[0], "--reserve-vram", "5"]
            import comfy.options
            comfy.options.enable_args_parsing()
            import torch
            import nodes
            import comfy_kitchen as ck
            from safetensors.torch import load_file, save_file
            torch.set_num_threads(2)
            samples = load_file(str(latent), device="cpu")["latent_tensor"]
            if list(samples.shape) != capture["files"]["video"]["tensor"]["shape"] or tensor_sha(samples) != capture["files"]["video"]["tensor"]["sha256"]:
                raise RuntimeError("Serialized samples do not match capture tensor")
            if tuple(samples.shape) != (1, 24, 22, 30, 52):
                raise ValueError("This qualification recipe requires the captured 832x480x73 tensor")
            # Start the timed telemetry contract after Python imports, immediately
            # before the monitor. Slow imports are not a missing runtime sample.
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError("Startup resource guard: " + reason)
            with resource_monitor(reader, guard, root / "resources.jsonl") as check:
                start = time.perf_counter()
                vae = nodes.VAELoader().load_vae(args.model)[0]
                result["loader_seconds"] = time.perf_counter() - start
                result["vae"] = {"class": type(vae.first_stage_model).__qualname__,
                    "dtype": str(vae.vae_dtype), "device": str(vae.device),
                    "handles_tiling": vae.handles_tiling, "torch": torch.__version__,
                    "native_tiled_note": "H3 owns semantic temporal/spatial tiling; decode_tiled delegates to decode, not a distinct arbitrary tile algorithm."}
                calls = {"attempted": 0, "completed": 0, "convrot_completed": 0}
                original = ck.int8_linear
                def measured(*values, **kwargs):
                    calls["attempted"] += 1
                    output = original(*values, **kwargs)
                    calls["completed"] += 1
                    calls["convrot_completed"] += bool(kwargs.get("convrot", False))
                    return output
                ck.int8_linear = measured
                try:
                    cases = [("cold_full", samples, False), ("hot_full", samples, False),
                        ("native_tiled_full", samples, True), ("single_frame", samples[:, :, :1], False),
                        ("first_five_frames", samples[:, :, :2], False),
                        ("temporal_22_frames", samples[:, :, :7], False),
                        ("spatial_boundary_816x464", samples[:, :, :, :29, :51], False)]
                    baseline, case_reports = None, []
                    for name, value, tiled in cases:
                        check()
                        before = dict(calls)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        with torch.inference_mode():
                            output = vae.decode_tiled(value) if tiled else vae.decode(value)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if output.ndim == 5:
                            output = output.flatten(0, 1)
                        output = output.detach().cpu().contiguous()
                        if tuple(output.shape) != expected_shape(value) or not bool(torch.isfinite(output).all()):
                            raise RuntimeError(f"Invalid decoded result {name}: {output.shape}")
                        row = {"case": name, "shape": list(output.shape), "sha256": tensor_sha(output),
                            "decode_seconds": elapsed, "cuda_max_allocated": torch.cuda.max_memory_allocated(),
                            "cuda_max_reserved": torch.cuda.max_memory_reserved(),
                            "kernel_calls": {key: calls[key]-before[key] for key in calls},
                            "minimum": float(output.min()), "maximum": float(output.max())}
                        if name == "cold_full":
                            baseline = output
                            save_file({"images": output}, str(root / "decoded-float.safetensors"))
                            result["preview_command"] = preview(output, args.source_video, root / "preview.mp4")
                            result["preview_audit"] = audit_preview(root / "preview.mp4", args.source_video,
                                output.shape[0], output.shape[2], output.shape[1])
                        elif name in ("hot_full", "native_tiled_full"):
                            row["cold_bit_equal"] = torch.equal(output, baseline)
                            row["cold_max_abs"] = float((output-baseline).abs().max())
                        case_reports.append(row)
                        write(root / (name + ".json"), row)
                        print(json.dumps(row), flush=True)
                        check()
                    result["cases"] = case_reports
                    result["quantized_dispatch"] = calls
                    if "int8" in args.model and (calls["convrot_completed"] == 0 or calls["attempted"] != calls["completed"]):
                        raise RuntimeError("Official INT8 did not complete observed ConvRot INT8 linear calls")
                    if "fp16" in args.model and calls["attempted"]:
                        raise RuntimeError("FP16 baseline unexpectedly used the observed INT8 path")
                finally:
                    ck.int8_linear = original
            if tensor_sha(samples) != capture["files"]["video"]["tensor"]["sha256"]:
                raise RuntimeError("Decode mutated the saved latent")
            for key in ("model", "latent", "source_video", "probe_source"):
                identity = result[key]
                if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
                    raise RuntimeError("Qualification input/source changed: " + key)
            result["status"] = "decode_mechanical_pass_human_pending"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["resources"] = guard.report()
        write(root / "terminal.json", result)


if __name__ == "__main__":
    main()
