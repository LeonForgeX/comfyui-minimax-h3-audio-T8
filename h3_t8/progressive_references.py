"""Keep native reference coordinates independent of the generated canvas.

Core PackedLayout assigns each reference its own spatial/time coordinates.
The progressive boundary changes the generated video canvas only. Reference
latents, soundtrack lengths and text/vision tokens retain their values/order.
"""
from __future__ import annotations

import torch


def describe_references(refs, *, required=True):
    if not isinstance(refs, list) or (required and not refs) or len(refs) > 15:
        raise ValueError("Ref2VA requires a native reference list with 1–15 blocks")
    counts = {"image": 0, "video": 0, "audio": 0}
    report = []
    for ordinal, block in enumerate(refs, 1):
        if not isinstance(block, dict):
            raise ValueError("Ref2VA reference blocks must be dictionaries")
        kind = block.get("kind")
        if kind not in {"image", "video", "video_audio", "audio"}:
            raise ValueError("Ref2VA only accepts native image/video/audio reference blocks")
        video_kind = kind in {"video", "video_audio"}
        counts["video" if video_kind else kind] += 1
        row = {"ordinal": ordinal, "kind": kind}
        if kind != "audio":
            height, width = block.get("latent_h"), block.get("latent_w")
            if any(type(size) is not int or size < 2 or size % 2 for size in (height, width)):
                raise ValueError("Ref2VA reference canvas must use even positive native latent dimensions")
            time = block.get("latent_t") if video_kind else 1
            if video_kind and (type(time) is not int or time < 2 or (time - 2) % 5):
                raise ValueError("Ref2VA video reference must follow the native 5n+2 time grid")
            latent = block.get("latent")
            _tensor(latent, (1, 24, time, height, width), "visual reference")
            row["video_shape"] = list(latent.shape)
        elif "latent" in block:
            raise ValueError("Ref2VA audio block cannot carry a visual latent")
        if kind in {"audio", "video_audio"}:
            time = block.get("ref_audio_t")
            if type(time) is not int or time <= 0:
                raise ValueError("Ref2VA audio reference requires its actual native time length")
            latent = block.get("audio_latent")
            _tensor(latent, (1, 32, 2, time), "audio reference")
            row["audio_shape"] = list(latent.shape)
        elif block.get("audio_latent") is not None or (video_kind and block.get("ref_audio_t") != 0):
            raise ValueError("Ref2VA silent visual reference has unexpected audio metadata")
        report.append(row)
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("Ref2VA reference limits are 9 images, 3 videos and 3 standalone audios")
    return report


def _tensor(value, shape, label):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Ref2VA {label} must be finite floating point with shape {shape}")
