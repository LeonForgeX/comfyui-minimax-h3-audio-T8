"""Independently inspect saved complete encoder tiles, original pixels and all moments."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def tensor_metrics(reference, candidate):
    import torch
    if reference.shape != candidate.shape or not reference.numel():
        raise ValueError("Encoder tensor shape mismatch or empty")
    if any(not value.is_floating_point() or not bool(torch.isfinite(value).all()) for value in (reference, candidate)):
        raise ValueError("Encoder evidence must be finite floating point")
    delta = candidate.double() - reference.double()
    rmse = delta.square().mean().sqrt().item()
    rms = reference.double().square().mean().sqrt().item()
    return {"shape": list(reference.shape), "rmse": rmse, "reference_rms": rms,
            "relative_rmse": rmse / rms if rms else None,
            "max_error": delta.abs().max().item(),
            "exact_tensor_equal": torch.equal(reference, candidate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    def read(name):
        return json.loads((root / name).read_text())
    request, result, terminal = read("request.json"), read("result.json"), read("terminal.json")
    complete = request.get("scope", "tile") == "full_0p5mp"
    if request.get("scope", "tile") not in ("tile", "full_0p5mp"):
        raise ValueError("Unknown audit scope")
    width, height, frame_count = (1024, 512, 73) if complete else (256, 256, 17)
    if (result["status"] != ("encoder_full_requires_independent_audit" if complete else "encoder_tiles_require_independent_audit")
            or terminal["status"] != "worker_complete_result_requires_audit"
            or terminal["isolated"]["status"] != "complete" or terminal["isolated"]["active_after_cleanup"] != 0
            or read("resource-summary.json")["status"] != "observations_within_policy"):
        raise ValueError("Encoder execution/cleanup/resource evidence incomplete")
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Encoder source changed")
    native_file = "native-latents.safetensors" if complete else "native-moments.safetensors"
    trt_file = "trt-latents.safetensors" if complete else "trt-moments.safetensors"
    expected_files = {"source-rgb8.safetensors", "source-crops.json", native_file,
                      "normalized-input.safetensors", trt_file}
    if set(result["files"]) != expected_files:
        raise ValueError("Missing encoder evidence files")
    for name, digest in result["files"].items():
        if digest_file(root / name) != digest:
            raise ValueError("Encoder evidence file changed")
    import av
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax.vae import IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD
    if complete:
        # Derive tile calls from the actual Core oracle, independently of counters.
        from tools.trt_vae_encoder_probe_worker import native_encoder_shell
        oracle = native_encoder_shell()
        spatial = len(oracle.split_tiles(height)[0]) * len(oracle.split_tiles(width)[0])
        expected_calls = {"image": spatial, "video": spatial * ((frame_count + oracle.clip_length - 1) // oracle.clip_length)}
        if result["native_calls"] != expected_calls or result["trt_calls"] != expected_calls:
            raise ValueError("Incomplete full encoder tile execution")
        calls = sum(expected_calls.values())
    else:
        calls = 2
    if result["actual_trt_calls"] != calls:
        raise ValueError("Actual encoder calls mismatch")
    pixels = load_file(str(root / "source-rgb8.safetensors"))
    normalized = load_file(str(root / "normalized-input.safetensors"))
    native = load_file(str(root / native_file))
    candidate = load_file(str(root / trt_file))
    crops = read("source-crops.json")
    def verify_pixels(array, kind, frame_index):
        geometry = crops[kind]
        h, w = array.shape[:2]
        x, y, cw, ch = geometry["xywh"]
        if geometry["source_wh"] != [w, h] or [x, y, cw, ch] != [(w-width)//2, (h-height)//2, width, height]:
            raise ValueError("Unexpected source crop")
        actual = pixels[kind][0, :, frame_index].permute(1, 2, 0).numpy()
        if not np.array_equal(array[y:y+ch, x:x+cw], actual):
            raise ValueError("Saved RGB pixels differ from real source")
    with Image.open(request["image"]) as source:
        verify_pixels(np.asarray(ImageOps.exif_transpose(source).convert("RGB")), "image", 0)
    seen = 0
    with av.open(request["video"]) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index == frame_count:
                break
            verify_pixels(frame.to_ndarray(format="rgb24"), "video", index)
            if str(frame.pts * frame.time_base) != crops["video_pts"][index]:
                raise ValueError("Source video clock mismatch")
            seen += 1
    if seen != frame_count:
        raise ValueError("Incomplete original video")
    for name, count in (("image", 1), ("video", frame_count)):
        if pixels[name].dtype != torch.uint8 or tuple(pixels[name].shape) != (1, 3, count, height, width):
            raise ValueError("RGB8 shape mismatch")
        raw = (pixels[name].float() / 255 * 2 - 1).half()
        mean = raw.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        std = raw.new_tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
        expected = ((raw + 1) * .5 - mean) / std
        if not torch.equal(expected, normalized[name]):
            raise ValueError("Encoder normalization differs from native half operations")
    if complete:
        if set(native) != {"image", "video"} or set(candidate) != {"image", "video"}:
            raise ValueError("Full encoder missing an input route")
        for name, tokens in (("image", 1), ("video", 22)):
            for values in (native, candidate):
                if values[name].dtype != torch.float32 or tuple(values[name].shape) != (1, 24, tokens, 32, 64):
                    raise ValueError("Full normalized latent shape/dtype mismatch")
        comparisons = {name: (native[name], candidate[name]) for name in native}
    else:
        for name, count in (("image", 1), ("image_padded17", 5), ("video", 5)):
            if native[name].dtype != torch.float16 or tuple(native[name].shape) != (1, 48, count, 16, 16):
                raise ValueError("Native moments geometry/dtype mismatch")
        for value in candidate.values():
            if value.dtype != torch.float16 or tuple(value.shape) != (1, 48, 5, 16, 16):
                raise ValueError("TRT moments geometry/dtype mismatch")
        comparisons = {
            "native_single_vs_native_padded_first": (native["image"], native["image_padded17"][:, :, :1]),
            "native_single_vs_trt_padded_first": (native["image"], candidate["image"][:, :, :1]),
            "native_padded17_vs_trt_padded17": (native["image_padded17"], candidate["image"]),
            "native_video17_vs_trt_video17": (native["video"], candidate["video"]),
        }
    latent_mean = torch.tensor(LATENTS_MEAN).view(1, 24, 1, 1, 1)
    latent_std = torch.tensor(LATENTS_STD).view(1, 24, 1, 1, 1)
    def latent(value):
        return (value[:, :24].float() - latent_mean) / latent_std
    rows = ({key: {"normalized_latent": tensor_metrics(a, b)} for key, (a, b) in comparisons.items()}
            if complete else {key: {"raw_moments": tensor_metrics(a, b), "normalized_latent": tensor_metrics(latent(a), latent(b))}
                              for key, (a, b) in comparisons.items()})
    report = {"status": ("full_0p5mp_encoder_evidence_audited_not_downstream_quality_qualification" if complete
                         else "real_encoder_tile_evidence_audited_not_full_encoder_or_quality_qualification"),
              "source_pixels_exact": True, "pixel_normalization_exact": True,
              "actual_trt_calls": calls, "comparisons": rows, "cuda_initialized": torch.cuda.is_initialized(),
              "auditor_sha256": digest_file(__file__),
              "files": {name: digest_file(root / name) for name in expected_files | {"request.json", "result.json", "terminal.json", "resource-summary.json"}},
              "limits": ("Complete0.5MP image and73-frame short video encoding, not downstream sampler/human acceptance, warm speed or long-video qualification."
                         if complete else "Exact 256px source tiles only. Numerical measurements are not human acceptance, full spatial/temporal encoding, sampler-quality or speed qualification.")}
    if report["cuda_initialized"]:
        raise RuntimeError("CPU encoder audit initialized CUDA")
    write_new_json(root / "independent-encoder-audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
