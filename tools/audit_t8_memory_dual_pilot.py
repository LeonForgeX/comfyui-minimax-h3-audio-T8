"""Audit one two-segment T8-memory dual-MODEL acceptance run.

The audit reads the emitted bytes after the owned ComfyUI process has stopped.
It proves the two independent LowVRAM/ChunkFFN paths and Prompt Relay executed;
visual continuity, speech quality, and seam quality remain human judgements.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

try:
    from tools.audit_dual_model_pilot import validate_audio_stage_contract
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from audit_dual_model_pilot import validate_audio_stage_contract


ATTENTION_WRAPPER = "t8_minimax_h3_low_vram_attention_owner_v1"
FFN_WRAPPER = "t8_minimax_h3_chunk_ffn_owner_v1"
RUNTIME_TOKEN = "t8_minimax_h3_memory_tokens_v1"
LATENT_SPATIAL_STRIDE = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video_latent_geometry(
    tensor: torch.Tensor, *, width: int, height: int, label: str
) -> dict:
    """Validate H3's VAE(8x) plus DiT patch(2x) spatial token grid."""
    if width % LATENT_SPATIAL_STRIDE or height % LATENT_SPATIAL_STRIDE:
        raise ValueError(f"{label} pixel geometry is not representable on the H3 latent grid")
    observed = tuple(int(value) for value in tensor.shape[-2:])
    expected = (
        height // LATENT_SPATIAL_STRIDE,
        width // LATENT_SPATIAL_STRIDE,
    )
    if observed != expected:
        raise ValueError(f"{label} latent geometry differs from the declared recipe")
    return {"pixels": [width, height], "latent_spatial": list(observed)}


def validate_t8_memory_stage(report: dict, terminal: dict, *, nfe: int = 4) -> dict:
    memory = report.get("memory_composition")
    backend_memory = report.get("backend", {}).get("memory_composition")
    relay = report.get("prompt_relay_execution")
    expected_ffn = [int(terminal.get("memory_ffn_chunks", 0)), 4096]
    expected_heads = int(terminal.get("memory_head_chunks", 0))
    if (
        terminal.get("backend") != "t8-memory"
        or expected_heads not in (1, 4)
        or terminal.get("memory_ffn_chunks") != 2
        or not isinstance(memory, dict)
        or memory != backend_memory
        or memory.get("kind") != "t8_h3_memory"
        or memory.get("head_chunks") != expected_heads
        or memory.get("ffn_settings") != expected_ffn
        or set(memory.get("wrapper_keys", ())) != {ATTENTION_WRAPPER, FFN_WRAPPER}
        or RUNTIME_TOKEN not in set(memory.get("runtime_option_keys", ()))
        or not memory.get("source_sha256s")
        or not isinstance(relay, dict)
        or relay.get("completed_forwards") != nfe
        or int(relay.get("routed_attention_calls", 0)) <= 0
        or report.get("completed_network_forwards") != nfe
    ):
        raise ValueError(
            "Required authenticated T8 memory and Prompt Relay execution was not observed"
        )
    return {
        "head_chunks": expected_heads,
        "ffn_settings": expected_ffn,
        "network_forwards": nfe,
        "relay_routed_attention_calls": int(relay["routed_attention_calls"]),
        "source_sha256s": list(memory["source_sha256s"]),
    }


def read_stage(output_root: Path, segment: int, stage: str) -> tuple[dict, dict]:
    directory = output_root / "dual_stages" / f"segment_{segment:05d}"
    paths = list(directory.rglob(stage + "-*.json"))
    if len(paths) != 1:
        raise ValueError(
            f"Segment {segment} requires exactly one {stage} checkpoint receipt"
        )
    receipt_path = paths[0]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("contract", {}).get("segment") != segment:
        raise ValueError("Stage receipt segment identity differs from its directory")
    tensor_path = (receipt_path.parent / receipt["tensor_file"]).resolve(strict=True)
    if not tensor_path.is_relative_to(output_root) or sha256(tensor_path) != receipt.get(
        "tensor_sha256"
    ):
        raise ValueError("Stage checkpoint byte identity mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if not tensors or any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Stage checkpoint contains missing or nonfinite tensors")
    return receipt, tensors


def probe_media(media: Path, *, width: int, height: int, frames: int) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(media),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    observed = (
        int(video["width"]),
        int(video["height"]),
        int(video["nb_read_frames"]),
        Fraction(video["avg_frame_rate"]),
    )
    if observed != (width, height, frames, Fraction(24)):
        raise ValueError(f"Final media geometry/timeline differs: {observed}")
    for mapping in (("0:v:0",), ("0:a:0",), ("0:v:0", "0:a:0")):
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "4",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(media),
        ]
        for item in mapping:
            command.extend(("-map", item))
        command.extend(("-f", "null", "-"))
        decoded = subprocess.run(command, capture_output=True, text=True)
        if decoded.returncode or decoded.stderr.strip():
            raise RuntimeError("Strict decoder error: " + decoded.stderr)
    return {
        "path": str(media),
        "sha256": sha256(media),
        "frames": frames,
        "width": width,
        "height": height,
        "fps": "24/1",
        "video_seconds": video.get("duration"),
        "audio_seconds": audio.get("duration"),
        "container_seconds": payload["format"].get("duration"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new audit output path")
    torch.set_num_threads(2)
    root = args.root.resolve(strict=True)
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    if (
        not terminal.get("status", "").startswith("generation_completed")
        or terminal.get("server_stop", {}).get("owned_children_remaining")
        or terminal.get("duration") != 8
        or not terminal.get("relay")
    ):
        raise ValueError("Requires one stopped, completed 8-second Relay generation")

    chain_roots = list((root / "output" / "minimax_h3_t8_long_video").glob("*"))
    chain_roots = [path.resolve() for path in chain_roots if path.is_dir()]
    if len(chain_roots) != 1:
        raise ValueError("Acceptance run must contain exactly one output chain")
    output_root = chain_roots[0]
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    segments = manifest.get("segments", [])
    if (
        manifest.get("schema") != 2
        or len(segments) != 2
        or [item.get("index") for item in segments] != [0, 1]
        or sum(int(item.get("frame_count", 0)) for item in segments) != 192
        or segments[0].get("timeline_start_frame") != 0
        or segments[-1].get("timeline_end_frame") != 192
        or any(not item.get("strict_decode_validated") for item in segments)
    ):
        raise ValueError("Accepted manifest is not the required two-segment 8-second chain")

    width = int(terminal["width"])
    height = int(terminal["height"])
    low_width = int(terminal["low_width"])
    low_height = int(terminal["low_height"])
    segment_reports = []
    stage_hashes = []
    for segment in range(2):
        records, tensors = {}, {}
        for stage in ("low_x0", "high_input", "high_output"):
            records[stage], tensors[stage] = read_stage(output_root, segment, stage)
        low, prepared, high = (tensors[name] for name in ("low_x0", "high_input", "high_output"))
        low_geometry = validate_video_latent_geometry(
            low["samples_video"], width=low_width, height=low_height, label="Low-stage"
        )
        high_geometry = validate_video_latent_geometry(
            high["samples_video"], width=width, height=height, label="High-stage"
        )
        if tuple(prepared["samples_video"].shape) != tuple(high["samples_video"].shape):
            raise ValueError("Prepared and output high-stage video geometry differs")
        audio = validate_audio_stage_contract(records, tensors)
        first = validate_t8_memory_stage(records["low_x0"]["report"], terminal)
        second = validate_t8_memory_stage(records["high_output"]["report"], terminal)
        segment_reports.append(
            {
                "index": segment,
                "audio": audio,
                "geometry": {"low": low_geometry, "high": high_geometry},
                "passes": [first, second],
            }
        )
        stage_hashes.append(
            {name: records[name]["tensor_sha256"] for name in records}
        )

    for item in segments:
        accepted = (output_root / item["video_path"]).resolve(strict=True)
        if not accepted.is_relative_to(output_root) or sha256(accepted) != item["video_sha256"]:
            raise ValueError("Accepted segment media byte identity mismatch")
    assembled = list((output_root / "assembled").glob("*.mp4"))
    if len(assembled) != 1:
        raise ValueError("Expected exactly one final assembled MP4")
    media = probe_media(assembled[0], width=width, height=height, frames=192)
    payload = {
        "status": "two_segment_t8_memory_mechanical_audit_pass_human_pending",
        "backend": "t8-memory",
        "duration_seconds": 8,
        "segment_count": 2,
        "boundary_frame": 124,
        "stage_sha256": stage_hashes,
        "segment_reports": segment_reports,
        "media": media,
        "cuda_initialized_by_auditor": torch.cuda.is_initialized(),
        "scope": (
            "Two serial 4+4 passes per segment with authenticated T8 memory and Relay; "
            "human review is still required for continuation, sound, and the frame-124 seam."
        ),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
