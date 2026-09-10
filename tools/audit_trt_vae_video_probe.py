"""CPU-only complete RGB evidence audit, one frame at a time; no inference."""

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from trt_vae_contract import output_shape  # noqa: E402


def frame_metrics(reference, candidate):
    import torch

    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("Frame geometry differs or is empty")
    for value in (reference, candidate):
        if (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                or value.min().item() < 0 or value.max().item() > 1):
            raise ValueError("Output is not finite normalized float32 RGB")
    delta = (candidate - reference).double()
    return {"count": delta.numel(), "sum_squared_error": delta.square().sum().item(),
            "max_error": delta.abs().max().item(), "over_2_codes": int((delta.abs() > 2 / 255).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    request = json.loads((root / "request.json").read_text())
    result = json.loads((root / "result.json").read_text())
    terminal = json.loads((root / "terminal.json").read_text())
    resources = json.loads((root / "resource-summary.json").read_text())
    if (terminal["status"] != "worker_complete_result_requires_audit"
            or terminal["isolated"]["status"] != "complete"
            or terminal["isolated"]["active_after_cleanup"] != 0
            or resources["status"] != "observations_within_policy"
            or result["status"] != "complete_rgb_requires_independent_audit"):
        raise ValueError("Execution or resource evidence is incomplete")
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Source identity changed")
    for name, digest in result["files"].items():
        if Path(name).name != name or digest_file(root / name) != digest:
            raise ValueError("Output identity changed")
    with safe_open(request["latent"], framework="pt", device="cpu") as source, safe_open(
        root / "input-latent.safetensors", framework="pt", device="cpu"
    ) as saved:
        if "latent_format_version_0" not in source.keys():
            raise ValueError("Unknown serialization scale")
        z = source.get_tensor("latent_tensor").half()
        if not torch.equal(z, saved.get_tensor("normalized_latent")):
            raise ValueError("Complete normalized input differs from saved source")
    shape = output_shape(z.shape)
    if tuple(result["output_shape"]) != shape:
        raise ValueError("Declared geometry mismatch")
    # Derive actual calls from the reference source oracle, not worker counters alone.
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from tools.trt_vae_video_probe_worker import native_video_shell
    core = native_video_shell()
    if tuple(core.decode_output_shape(z.shape)) != shape:
        raise ValueError("Actual Core output oracle differs")
    _, windows = core._decode_temporal_chunks(z.shape[2])
    calls = windows * len(core.split_tiles(shape[3])[0]) * len(core.split_tiles(shape[4])[0])
    if result["native_calls"] != calls or result["trt_calls"] != calls:
        raise ValueError("Missing actual tile calls")
    frames = []
    with safe_open(root / "native-rgb.safetensors", framework="pt", device="cpu") as native, safe_open(
        root / "trt-rgb.safetensors", framework="pt", device="cpu"
    ) as candidate:
        a, b = native.get_slice("rgb"), candidate.get_slice("rgb")
        if tuple(a.get_shape()) != shape or tuple(b.get_shape()) != shape:
            raise ValueError("Saved full RGB geometry mismatch")
        for index in range(shape[2]):
            metrics = frame_metrics(a[:, :, index], b[:, :, index])
            mse = metrics["sum_squared_error"] / metrics["count"]
            frames.append({"frame": index, **metrics, "psnr_db": -10 * math.log10(mse) if mse else None})
    count = sum(row["count"] for row in frames)
    mse = sum(row["sum_squared_error"] for row in frames) / count
    report = {"status": "complete_rgb_evidence_verified_not_human_or_speed_qualification",
              "source_shape": list(z.shape), "output_shape": list(shape), "actual_calls_per_route": calls,
              "same_saved_input_bit_equal": True, "finite_normalized_all_frames": True,
              "rgb_psnr_db": -10 * math.log10(mse) if mse else None,
              "rgb_max_error": max(row["max_error"] for row in frames),
              "rgb_fraction_over_2_codes": sum(row["over_2_codes"] for row in frames) / count,
              "per_frame": frames, "cuda_initialized": torch.cuda.is_initialized(),
              "files": {name: digest_file(root / name) for name in
                        ("request.json", "result.json", "terminal.json", "resource-summary.json",
                         "input-latent.safetensors", "native-rgb.safetensors", "trt-rgb.safetensors")},
              "audit_source_sha256": digest_file(__file__),
              "limits": "Only this complete source, no audio validation yet, no human or benchmark pass; metrics do not judge native source quality"}
    if report["cuda_initialized"]:
        raise RuntimeError("Independent CPU audit initialized CUDA")
    write_new_json(root / "independent-video-audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("per_frame", "files")}, indent=2))


if __name__ == "__main__":
    main()
