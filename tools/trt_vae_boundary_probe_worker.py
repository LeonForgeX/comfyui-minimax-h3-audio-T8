"""Native versus real flex-engine decoder on bounded actual latent regions."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def make_cases(image, video):
    if tuple(image.shape) != (1,24,1,32,64) or tuple(video.shape) != (1,24,22,32,64):
        raise ValueError("Boundary probe requires actual0.5MP image and saved73-frame video latents")
    cases = {"image_full": image, "image_small": image[..., :8, :13], "image_min": image[..., :1, :1]}
    for t in (2,5,6,7,8,11,12):
        cases[f"video_t{t}"] = video[:, :, :t, :16, :16]
    for name, h, w in (("small",8,13),("min",1,1),("subtile",15,16),("seam",17,19)):
        cases["video_"+name] = video[:, :, :7, :h, :w]
    return {k: v.half().contiguous() for k,v in cases.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf8"))
    if not request.get("boundary_probe") or request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the explicit owned boundary controller")
    for path, sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError("Boundary probe source changed: " + Path(path).name)
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from tools.trt_vae_video_probe_worker import native_video_shell
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("t8_trt_boundary_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_type = importlib.import_module(package.__name__+".trt_vae_backend").ScopedBackend
    decode_raw = importlib.import_module(package.__name__+".trt_vae_decode").decode_raw_latent
    contract = importlib.import_module(package.__name__+".trt_vae_contract")
    with safe_open(request["latent"], framework="pt", device="cpu") as f:
        if "latent_format_version_0" not in f.keys():
            raise ValueError("Unknown saved video latent format")
        video = f.get_tensor("latent_tensor")
    image = load_file(request["image_latent"], device="cpu")["native_latent"]
    cases = make_cases(image, video)
    save_file(cases, str(root / "inputs.safetensors"))
    core = native_video_shell()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as f:
        state = {k:f.get_tensor(k) for k in f.keys() if k.startswith(("decoder.","post_quant_conv."))}
    state.update(latents_mean=core.latents_mean, latents_std=core.latents_std)
    core.load_state_dict(state, strict=True, assign=True)
    del state
    core.decoder = core.decoder.to(device="cuda:0", dtype=torch.float16).eval()
    core.post_quant_conv = core.post_quant_conv.to(device="cuda:0", dtype=torch.float16).eval()
    native, native_calls, count = {}, {}, 0
    def hook(module, inputs):
        nonlocal count
        count += 1
    handle = core.decoder.register_forward_pre_hook(hook)
    mean = core.latents_mean.half().view(1,24,1,1,1)
    std = core.latents_std.half().view(1,24,1,1,1)
    with torch.inference_mode():
        for name, z in cases.items():
            print("T8_BOUNDARY_NATIVE " + name, flush=True)
            count = 0
            native[name] = core.decode(z.cuda()).cpu()
            native_calls[name] = count
            if tuple(native[name].shape) != contract.output_shape(tuple(z.shape)):
                raise ValueError("Actual native output differs from temporal/spatial contract")
    handle.remove()
    del core
    gc.collect()
    torch.cuda.empty_cache()
    save_file(native, str(root / "native-rgb.safetensors"))
    def check():
        if (root / "cancel.request").exists():
            raise InterruptedError("Boundary test cancelled")
    backend = backend_type({"decoder": request["bundle"]}, request["runtime_site"], check=check,
                           serial_lease=lambda: SerialProbeLease(Path(tempfile.gettempdir())/"T8-TRT-VAE-boundary-worker.lock"))
    shapes = sorted({shape for z in cases.values() for shape in contract.required_tile_shapes(tuple(z.shape))})
    candidate, calls, actual_shapes = {}, {}, {}
    with backend("decoder", shapes) as decode:
        for name, z in cases.items():
            print("T8_BOUNDARY_TRT " + name, flush=True)
            count = 0
            seen = []
            def tile(value):
                nonlocal count
                count += 1
                seen.append(list(value.shape))
                return decode(value)
            candidate[name] = decode_raw(z*std+mean, tile, max_output_bytes=request["output_budget_bytes"], check=check)
            calls[name], actual_shapes[name] = count, seen
            if candidate[name].shape != native[name].shape or not bool(torch.isfinite(candidate[name]).all()):
                raise ValueError("TRT boundary output shape/finiteness failure")
    if calls != native_calls:
        raise ValueError("Native and TRT actual tile counts differ")
    save_file(candidate, str(root / "trt-rgb.safetensors"))
    report = {"status": "real_boundary_comparisons_require_independent_audit", "native_calls": native_calls,
              "trt_calls": calls, "actual_trt_shapes": actual_shapes, "lease": backend.last_report,
              "files": {name:digest_file(root/name) for name in ("inputs.safetensors","native-rgb.safetensors","trt-rgb.safetensors")},
              "scope": "One genuine encoded still plus exact cropped/truncated regions of saved short generated latent. T1, min/small/seam spatial and temporal tail boundaries; not new generation or long-video/human acceptance."}
    write_new_json(root / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
