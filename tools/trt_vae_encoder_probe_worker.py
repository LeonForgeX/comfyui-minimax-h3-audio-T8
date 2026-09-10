"""Actual Core encoder vs pinned TRT, preserving raw input and moments evidence."""

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import time
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, loaded_windows_libraries, validate_request, write_new_json  # noqa: E402


def native_encoder_shell():
    import torch
    from comfy.ldm.minimax.vae import IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE

    with torch.device("meta"):
        core = MiniMaxH3VideoVAE()
    del core.decoder, core.post_quant_conv
    core.latents_mean, core.latents_std = torch.tensor(LATENTS_MEAN), torch.tensor(LATENTS_STD)
    core.pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    core.pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    return core


def source_tiles(image_path, video_path, *, width=256, height=256, frame_count=17):
    import av
    import numpy as np
    from PIL import Image, ImageOps
    import torch

    if (width, height, frame_count) not in ((256, 256, 17), (1024, 512, 73)):
        raise ValueError("Unknown bounded encoder probe geometry")

    def crop(array):
        h, w = array.shape[:2]
        if h < height or w < width:
            raise ValueError("Real source must contain requested region without scaling")
        y, x = (h - height) // 2, (w - width) // 2
        return array[y:y + height, x:x + width].copy(), {"source_wh": [w, h], "xywh": [x, y, width, height]}

    with Image.open(image_path) as source:
        still, image_crop = crop(np.asarray(ImageOps.exif_transpose(source).convert("RGB")))
    frames, pts, video_crop = [], [], None
    with av.open(str(video_path)) as source:
        for frame in source.decode(video=0):
            tile, geometry = crop(frame.to_ndarray(format="rgb24"))
            if frames and geometry != video_crop:
                raise ValueError("Video geometry changed")
            video_crop = geometry
            frames.append(tile)
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Source frame has no timestamp")
            pts.append(str(frame.pts * frame.time_base))
            if len(frames) == frame_count:
                break
    if len(frames) != frame_count:
        raise ValueError(f"Need {frame_count} genuine consecutive frames, no repeated-frame substitute")
    def tensor(array):
        return torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return {"image": tensor(still[None]), "video": tensor(np.stack(frames))}, {
        "image": image_crop, "video": video_crop, "video_pts": pts,
        "processing": f"Exact centered {width}x{height} RGB8 regions; EXIF orientation applied to image; no resizing/sharpening"}


def probe_geometry(request):
    scope = request.get("scope", "tile")
    if scope == "tile":
        return False, 256, 256, 17
    if scope == "full_0p5mp":
        return True, 1024, 512, 73
    raise ValueError("Unknown encoder probe scope")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    complete, width, height, frames = probe_geometry(request)
    root = args.request.parent
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Encoder probe source identity changed")
    bundle = Path(request["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest["source_request"].get("kind") != "encoder"
            or validate_request(manifest["source_request"]) != manifest["request_sha256"]
            or digest_file(bundle / "model.engine") != manifest["engine_sha256"]):
        raise ValueError("Encoder engine identity mismatch")
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tiles, crops = source_tiles(request["image"], request["video"], width=width, height=height, frame_count=frames)
    save_file(tiles, str(root / "source-rgb8.safetensors"))
    write_new_json(root / "source-crops.json", crops)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Single-GPU encoder probe required")
    torch.cuda.set_device(0)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() != manifest["gpu_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("Encoder GPU UUID mismatch")
    core = native_encoder_shell()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as state_file:
        state = {k: state_file.get_tensor(k) for k in state_file.keys() if k.startswith(("encoder.", "quant_conv."))}
    state.update(latents_mean=core.latents_mean, latents_std=core.latents_std)
    core.load_state_dict(state, strict=True, assign=True)
    del state
    core.encoder = core.encoder.to(device="cuda:0", dtype=torch.float16).eval()
    core.quant_conv = core.quant_conv.to(device="cuda:0", dtype=torch.float16).eval()
    if any(value.is_meta for value in (*core.parameters(), *core.buffers())):
        raise RuntimeError("Native encoder has unmaterialized tensors")
    native, normalized, raw_inputs, timings, native_calls = {}, {}, {}, {}, {}
    normalization = {"latents_mean": core.latents_mean.tolist(), "latents_std": core.latents_std.tolist()}
    padded = None
    count = 0
    def count_native(module, inputs):
        nonlocal count
        count += 1
    hook = core.encoder.register_forward_pre_hook(count_native)
    with torch.inference_mode():
        for name, tile in tiles.items():
            # Match Core's half input in [-1,1] and its own pixel normalization.
            raw = (tile.float() / 255 * 2 - 1).half().cuda()
            if complete:
                raw_inputs[name] = raw.cpu()
            value = core._normalize_pixels(raw)
            normalized[name] = value.cpu()
            torch.cuda.synchronize()
            start = time.perf_counter()
            count = 0
            native[name] = (core.encode(raw) if complete else core._encode_moments(value)).cpu()
            torch.cuda.synchronize()
            timings["native_" + name] = time.perf_counter() - start
            native_calls[name] = count
            if name == "image" and not complete:
                padded = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, 16))
                native["image_padded17"] = core._encode_moments(padded).cpu()
            del raw, value
    hook.remove()
    if any(not bool(torch.isfinite(value).all()) for value in native.values()):
        raise ValueError("Nonfinite native encoder moments")
    native_file = "native-latents.safetensors" if complete else "native-moments.safetensors"
    trt_file = "trt-latents.safetensors" if complete else "trt-moments.safetensors"
    save_file(native, str(root / native_file))
    save_file(normalized, str(root / "normalized-input.safetensors"))
    del core, native, tiles, padded
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("T8_TRT_ENCODER native_complete_unloaded", flush=True)
    sys.path.insert(0, request["runtime_site"])
    import tensorrt as trt
    if trt.__version__ != manifest["runtime_version"]:
        raise RuntimeError("Encoder runtime version changed")
    mapped = loaded_windows_libraries()
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll"):
        if not Path(mapped[name]).is_relative_to(Path(request["runtime_site"]).resolve()):
            raise RuntimeError("TRT DLL outside isolated runtime")
    package = types.ModuleType("t8_encoder_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    runner_type = importlib.import_module(package.__name__ + ".trt_vae_engine").TileEngineRunner
    if complete:
        encode_rgb = importlib.import_module(package.__name__ + ".trt_vae_encode").encode_rgb
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((bundle / "model.engine").read_bytes())
    if engine is None:
        raise RuntimeError("Encoder deserialize failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Encoder context creation failed")
    outputs, trt_calls = {}, {}
    with runner_type(engine, context, trt, device="cuda:0", kind="encoder") as runner:
        for name, value in (raw_inputs if complete else normalized).items():
            if name == "image" and not complete:
                value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, 16))
            start = time.perf_counter()
            prior_calls = runner.calls
            outputs[name] = (encode_rgb(value, runner, **normalization, max_output_bytes=64*1024**2,
                                        single_frame_mode="causal_zero17_first") if complete else runner(value).cpu())
            torch.cuda.synchronize()
            timings["trt_" + name] = time.perf_counter() - start
            trt_calls[name] = runner.calls - prior_calls
        calls = runner.calls
    save_file(outputs, str(root / trt_file))
    del context, engine, runtime
    gc.collect()
    torch.cuda.empty_cache()
    report = {"status": "encoder_full_requires_independent_audit" if complete else "encoder_tiles_require_independent_audit", "actual_trt_calls": calls,
              "native_calls": native_calls, "trt_calls": trt_calls,
              "timings_first_only_not_benchmark": timings, "loaded_libraries": mapped,
              "files": {name: digest_file(root / name) for name in ("source-rgb8.safetensors", "source-crops.json",
                        native_file, "normalized-input.safetensors", trt_file)},
              "scope": ("Complete1024x512 image and73 genuine video frames; full Core/native and independentTRT spatial/temporal encoding, no downstream sampler or warm performance qualification"
                        if complete else "Real centered256px image and17-frame video tiles only; no complete image/video spatial assembly, sampler quality, long-video or warm performance qualification")}
    write_new_json(root / "result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
