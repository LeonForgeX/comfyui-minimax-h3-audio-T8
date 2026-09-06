#!/usr/bin/env python3
"""Compare the same-material Face Refine run before and after the v1.1 sampler-mask fix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


SCHEMA = "t8.minimax_h3.face_refine_sampler_mask_v11_analysis.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _decode_rgb(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decoded frames: {path}")
    return np.stack(frames)


def _pcm_sha256(path: Path, ffmpeg: str) -> str:
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    token = completed.stdout.strip().split("=", 1)[-1]
    if len(token) != 64:
        raise ValueError(f"Unexpected FFmpeg PCM hash output for {path}")
    return token.upper()


def _ssim_rows(first: np.ndarray, second: np.ndarray) -> list[float]:
    if first.shape != second.shape:
        raise ValueError(f"Frame shape mismatch: {first.shape} versus {second.shape}")
    return [
        float(structural_similarity(a, b, data_range=255, channel_axis=2))
        for a, b in zip(first, second, strict=True)
    ]


def _ssim_summary(rows: list[float]) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _quadrants(frame: np.ndarray) -> list[np.ndarray]:
    height, width = frame.shape
    return [
        frame[: height // 2, : width // 2],
        frame[: height // 2, width // 2 :],
        frame[height // 2 :, : width // 2],
        frame[height // 2 :, width // 2 :],
    ]


def _sixteen_px_prominence(tile: np.ndarray) -> float:
    height, width = tile.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((tile - tile.mean()) * window)))
    fy = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    band = (radius >= 1.0 / 80.0) & (radius <= 1.0 / 6.0)
    tolerance = max(1.0 / height, 1.0 / width) * 1.25
    target = band & (
        (np.abs(np.abs(fx) - 1.0 / 16.0) <= tolerance)
        | (np.abs(np.abs(fy) - 1.0 / 16.0) <= tolerance)
    )
    background = spectrum[band & ~target]
    if not np.any(target) or background.size == 0:
        return 0.0
    return float(np.max(spectrum[target]) / max(float(np.median(background)), 1e-12))


def _grid_summary(source: np.ndarray, candidate: np.ndarray, frame_count: int) -> dict[str, Any]:
    rows = []
    for index in range(min(frame_count, len(source), len(candidate))):
        delta = candidate[index].astype(np.float32) - source[index].astype(np.float32)
        gray = delta.mean(axis=2)
        values = [_sixteen_px_prominence(tile) for tile in _quadrants(gray)]
        rows.append(
            {
                "frame": index,
                "quadrants": values,
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            }
        )
    all_values = np.asarray(
        [value for row in rows for value in row["quadrants"]], dtype=np.float64
    )
    return {
        "method": (
            "generated-minus-source grayscale; Hann-window 2D FFT per image quadrant; "
            "maximum 16px-axis component divided by median 6..80px frequency band"
        ),
        "frame_count": len(rows),
        "quadrant_observation_count": int(all_values.size),
        "median_prominence": float(np.median(all_values)),
        "mean_prominence": float(np.mean(all_values)),
        "maximum_prominence": float(np.max(all_values)),
        "per_frame": rows,
        "boundary": "diagnostic approximation of upstream issue #15981, not a perceptual score",
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "source": args.source.resolve(),
        "previous": args.previous.resolve(),
        "corrected": args.corrected.resolve(),
        "author_target": args.author_target.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    frames = {name: _decode_rgb(path) for name, path in paths.items()}
    shapes = {name: list(value.shape) for name, value in frames.items()}
    if len({tuple(value.shape) for value in frames.values()}) != 1:
        raise ValueError(f"Decoded frame stacks differ: {shapes}")

    source = frames["source"]
    previous = frames["previous"]
    corrected = frames["corrected"]
    author = frames["author_target"]
    ssim = {
        "previous_vs_corrected": _ssim_rows(previous, corrected),
        "source_vs_previous": _ssim_rows(source, previous),
        "source_vs_corrected": _ssim_rows(source, corrected),
        "author_vs_previous": _ssim_rows(author, previous),
        "author_vs_corrected": _ssim_rows(author, corrected),
    }
    ssim_summary = {
        name: {
            "full": _ssim_summary(rows),
            "first_24": _ssim_summary(rows[:24]),
        }
        for name, rows in ssim.items()
    }
    pcm = {name: _pcm_sha256(path, args.ffmpeg) for name, path in paths.items()}
    old_grid = _grid_summary(source, previous, 24)
    new_grid = _grid_summary(source, corrected, 24)
    result = {
        "schema": SCHEMA,
        "inputs": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "decoded_shape": shapes[name],
                "pcm_s16le_sha256": pcm[name],
            }
            for name, path in paths.items()
        },
        "strict_same_material": {
            "same_decoded_shape": len({tuple(value.shape) for value in frames.values()}) == 1,
            "same_source_audio_pcm": pcm["source"]
            == pcm["previous"]
            == pcm["corrected"],
            "previous_corrected_not_duplicate": _sha256(paths["previous"])
            != _sha256(paths["corrected"]),
        },
        "rgb_ssim": ssim_summary,
        "sixteen_px_grid_diagnostic": {
            "previous": old_grid,
            "corrected": new_grid,
            "median_ratio_corrected_over_previous": (
                new_grid["median_prominence"] / old_grid["median_prominence"]
            ),
            "corrected_median_lower": new_grid["median_prominence"]
            < old_grid["median_prominence"],
        },
        "quality_boundary": (
            "SSIM and FFT are diagnostics only. They cannot decide facial identity, naturalness, "
            "expression, mouth shape or temporal quality; full-video blind review is required."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--author-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = analyze(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
