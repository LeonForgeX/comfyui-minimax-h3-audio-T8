"""Compare actual same-latent decoded floats; numerical evidence is not human acceptance."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h16_vae_decode_probe import tensor_sha  # noqa: E402


def metrics(a, b):
    import torch
    if a.shape != b.shape or a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError("Expected matching F,H,W,3 tensors")
    rows = []
    for index, (first, second) in enumerate(zip(a, b)):
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("Non-finite decoded image")
        delta = second.float() - first.float()
        mse = float(delta.square().mean())
        rows.append(dict(frame=index, mae=float(delta.abs().mean()), mse=mse,
                         max_abs=float(delta.abs().max()),
                         mean_rgb_delta=delta.mean((0, 1)).tolist()))
    mse = sum(row["mse"] for row in rows) / len(rows)
    return dict(frames=rows, mse=mse, mae=sum(row["mae"] for row in rows) / len(rows),
                max_abs=max(row["max_abs"] for row in rows),
                psnr_db=None if mse == 0 else -10 * math.log10(mse),
                identical=mse == 0, range_for_psnr=1.0)


def compare(first, second):
    from safetensors.torch import load_file
    reports, tensors = [], []
    for root in (first, second):
        report = json.loads((root / "terminal.json").read_text(encoding="utf8"))
        if report["status"] != "decode_mechanical_pass_human_pending":
            raise ValueError("Both complete decodes must have passed mechanical checks")
        images = load_file(str(root / "decoded-float.safetensors"), device="cpu")["images"]
        cold = next(row for row in report["cases"] if row["case"] == "cold_full")
        if tensor_sha(images) != cold["sha256"] or list(images.shape) != cold["shape"]:
            raise ValueError("Decoded float evidence changed")
        reports.append(report)
        tensors.append(images)
    if reports[0]["latent"]["sha256"] != reports[1]["latent"]["sha256"]:
        raise ValueError("Different sampler latent: not a VAE-only comparison")
    if reports[0]["source_video"]["sha256"] != reports[1]["source_video"]["sha256"]:
        raise ValueError("Source audio/video differs")
    return dict(status="same_latent_numerical_comparison", quality_accepted=False,
                diffusion_forwards=0, latent_sha256=reports[0]["latent"]["sha256"],
                models=[report["model"] for report in reports],
                measurement=[{"loader_seconds": report["loader_seconds"],
                              "cases": report["cases"], "resources": report["resources"]}
                             for report in reports],
                float_difference=metrics(*tensors),
                limitations="Single saved latent and machine. PSNR and timing do not prove human quality or end-to-end generation speed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    result = compare(args.fp16, args.int8)
    if torch.cuda.is_initialized():
        raise RuntimeError("Comparison must not initialize CUDA")
    result["cuda_initialized"] = False
    with args.report.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result["float_difference"].items() if key != "frames"}))


if __name__ == "__main__":
    main()
