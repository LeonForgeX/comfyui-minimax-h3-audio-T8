"""Independent CPU reread of saved inputs/outputs; never re-executes the engine."""

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    output = root / "independent-audit.json"
    if output.exists():
        raise FileExistsError(output)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    request = json.loads((root / "request.json").read_text())
    terminal = json.loads((root / "terminal.json").read_text())
    resources = json.loads((root / "resource-summary.json").read_text())
    if (terminal["status"] != "worker_complete_result_requires_audit"
            or terminal["isolated"]["status"] != "complete"
            or terminal["isolated"]["active_after_cleanup"] != 0
            or resources["status"] != "observations_within_policy"):
        raise ValueError("Worker or resource qualification failed")
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Probe source changed")
    with safe_open(request["latent"], framework="pt", device="cpu") as file:
        if "latent_format_version_0" not in file.keys():
            raise ValueError("Unknown saved-latent scaling")
        latent = file.get_slice("latent_tensor")[0:1, :, 0:7, 0:16, 0:16].half()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["minimax_h3_video_vae"])
    mean = torch.tensor(metadata["latents_mean"], dtype=torch.float16).view(1, 24, 1, 1, 1)
    std = torch.tensor(metadata["latents_std"], dtype=torch.float16).view(1, 24, 1, 1, 1)
    expected = latent * std + mean
    actual_input = load_file(root / "input-tile.safetensors", device="cpu")["raw_tile"]
    if not torch.equal(expected, actual_input):
        raise ValueError("Saved engine input is not the declared native latent crop")
    native = load_file(root / "native.safetensors", device="cpu")["raw_rgb"]
    candidate = load_file(root / "trt.safetensors", device="cpu")["raw_rgb"]
    if native.shape != candidate.shape or tuple(native.shape) != (1, 3, 28, 256, 256):
        raise ValueError("Output geometry mismatch")
    if not bool(torch.isfinite(native).all() and torch.isfinite(candidate).all()):
        raise ValueError("Nonfinite output")
    native, candidate = native.float(), candidate.float()
    raw_diff = candidate - native
    raw_rmse = raw_diff.double().square().mean().sqrt().item()
    mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1, 1)
    std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1, 1)
    native_rgb = (native * std + mean).clamp(0, 1)
    candidate_rgb = (candidate * std + mean).clamp(0, 1)
    delta = candidate_rgb - native_rgb
    rgb_mse = delta.double().square().mean().item()
    report = {"status": "single_tile_evidence_verified_not_visual_qualification", "cuda_initialized": torch.cuda.is_initialized(),
              "same_saved_input_bit_equal": True, "finite": True, "raw_rmse": raw_rmse,
              "raw_max_absolute_error": raw_diff.abs().max().item(),
              "relative_raw_rmse": raw_rmse / native.double().square().mean().sqrt().item(),
              "rgb_psnr_db": -10 * math.log10(rgb_mse) if rgb_mse else None,
              "rgb_max_absolute_error": delta.abs().max().item(),
              "rgb_fraction_over_2_codes": (delta.abs() > 2 / 255).float().mean().item(),
              "files": {name: digest_file(root / name) for name in
                        ("input-tile.safetensors", "native.safetensors", "trt.safetensors", "request.json", "result.json")},
              "limitation": "256px raw tile only; not full temporal assembly, speed, audio, long-video or human acceptance"}
    if report["cuda_initialized"]:
        raise RuntimeError("CPU audit initialized CUDA")
    write_new_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
