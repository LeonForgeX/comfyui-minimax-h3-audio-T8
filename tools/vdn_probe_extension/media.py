"""Probe-only raw IMAGE/AUDIO delivery through the existing isolated encoder."""

import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
import types

import torch


def delivery_modules():
    name = "t8_vdn_delivery_probe"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[2])]
        sys.modules[name] = package
    return (importlib.import_module(name + ".long_video_delivery"),
            importlib.import_module(name + ".h3_world_advanced"))


def save_raw_av(images, audio, output, fps=24):
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("IMAGE must be [frames,height,width,channels]")
    count, height, width, channels = images.shape
    if min(count, height, width) <= 0 or width % 2 or height % 2 or channels not in (3, 4):
        raise ValueError("probe requires nonempty even-size RGB/RGBA frames")
    if fps != 24:
        raise ValueError("probe requires 24 fps")
    output = Path(output).resolve()
    if output.suffix != ".mp4" or output.exists():
        raise ValueError("probe output must be a new MP4")
    rate = audio.get("sample_rate")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("invalid AUDIO sample rate")
    delivery, world = delivery_modules()
    audio_array, rate, audio_report = world._normalize_output_audio(
        audio, expected_samples=math.ceil(count * rate / fps))
    if audio_report["clipped_sample_values"]:
        raise ValueError("probe refuses clipped source audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb_hash = hashlib.sha256()

    def chunks():
        for frame in images:
            rgb = frame[..., :3].detach().float().cpu()
            if not torch.isfinite(rgb).all():
                raise ValueError("nonfinite IMAGE")
            raw = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
            rgb_hash.update(raw)
            yield raw

    # Private temporary directory, never a transcode of the corrupt native MP4.
    with tempfile.TemporaryDirectory(prefix=".vdn-encode-", dir=output.parent) as temporary:
        temp = Path(temporary)
        video, raw_audio, combined = temp / "video.mp4", temp / "audio.f32le", temp / "combined.mp4"
        delivery._encode_rgb_frames_isolated(video, chunks, frame_count=count, width=width,
                                            height=height, fps=fps, bit_depth=8, crf=18)
        delivery._strict_validate_mp4(video, require_audio=False)
        delivery._write_planar_audio_raw(raw_audio, audio_array)
        world._mux_h3_world_audio(video, raw_audio, combined, sample_rate=rate, duration_seconds=count / fps)
        delivery._strict_validate_mp4(combined, require_audio=True)
        # Same-volume hard link publishes only if the destination is still absent.
        os.link(combined, output)
    return {"status": "pass", "encoder_policy": delivery.ISOLATED_VIDEO_ENCODER_POLICY,
            "source_rgb8_sha256": rgb_hash.hexdigest(), "frames": count, "width": width,
            "height": height, "fps": fps, "audio": audio_report,
            "output_sha256": delivery._sha256_file(output), "output": str(output)}
