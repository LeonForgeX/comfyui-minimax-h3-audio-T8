"""Replay captured RGB through an owned FFmpeg stdin pipe, without Comfy/Torch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024**2), b""):
            value.update(data)
    return value.hexdigest()


def replay(raw, output, *, pause=0):
    raw, output = Path(raw).resolve(strict=True), Path(output).resolve()
    meta = json.loads(raw.with_suffix(raw.suffix + ".json").read_text())
    if (meta["schema"] != "t8.outpaint.encoder_rgb_capture/v1" or not meta["complete_rgb_stream"]
            or meta["pixel_format"] != "rgb24" or meta["fps"] != "24/1"
            or meta["frames_written"] != meta["expected_frames"]
            or raw.stat().st_size != meta["expected_bytes"] or sha(raw) != meta["rgb_sha256"]):
        raise ValueError("raw capture identity or completeness mismatch")
    report_path = output.with_suffix(output.suffix + ".replay.json")
    if output.exists() or report_path.exists() or output.suffix != ".mp4" or pause < 0:
        raise ValueError("replay requires new MP4/report paths and nonnegative pause")
    width, height = meta["width"], meta["height"]
    frame_bytes = width * height * 3
    if frame_bytes * meta["expected_frames"] != meta["expected_bytes"]:
        raise ValueError("capture geometry does not match its byte count")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    # This diagnostic deliberately tests the exact currently audited settings;
    # never execute an arbitrary command read from a sidecar.
    args = ["-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", "24", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p",
            "-vf", "scale=in_range=full:out_range=limited:out_color_matrix=bt709", "-color_range", "tv",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-output_ts_offset", "0.000000000000", "-movflags", "+faststart"]
    if meta["provenance"]["encoder_command"][1:-1] != args:
        raise ValueError("captured encoder parameters differ from this diagnostic's audited settings")
    args[3] = "-n"
    command = [ffmpeg, *args, str(output)]
    with tempfile.TemporaryFile() as errors, raw.open("rb") as source:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), shell=False, bufsize=0)
        try:
            for index in range(meta["expected_frames"]):
                payload = memoryview(source.read(frame_bytes))
                if len(payload) != frame_bytes:
                    raise ValueError("raw file changed during replay")
                while payload:
                    written = process.stdin.write(payload)
                    if not written:
                        raise BrokenPipeError("replay encoder stopped accepting bytes")
                    payload = payload[written:]
                if pause and (index + 1) % 17 == 0:
                    time.sleep(pause)
            process.stdin.close()
            process.wait(timeout=300)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        errors.seek(0)
        message = errors.read(8000).decode("utf-8", errors="replace")
        result = {"command": command, "pause_per_17_frames": pause, "exit_code": process.returncode,
                  "stderr": message, "raw_sha256": sha(raw), "output_sha256": sha(output) if output.exists() else None,
                  "scope": "diagnostic_only; decode/visual acceptance not claimed"}
        with report_path.open("x", encoding="utf-8") as report:
            json.dump(result, report, indent=2)
        if result["raw_sha256"] != meta["rgb_sha256"] or process.returncode or message.strip():
            raise RuntimeError("replay failed or input changed; retained diagnostic output/report")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pause-per-17-frames", type=float, default=0)
    args = parser.parse_args()
    print(json.dumps(replay(args.raw, args.output, pause=args.pause_per_17_frames), indent=2))
