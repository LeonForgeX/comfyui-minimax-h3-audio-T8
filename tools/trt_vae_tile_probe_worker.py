"""One actual saved-latent tile: sequential native FP16 then TRT, no sampling."""

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
from trt_vae_build import digest_file, loaded_windows_libraries, validate_request, write_new_json  # noqa: E402


def native_decoder_shell():
    """Meta weights but real deterministic non-persistent RoPE buffer."""
    import torch
    from comfy.ldm.minimax.vae import RotaryEmbeddingND, ViT3DDecoder

    with torch.device("meta"):
        decoder = ViT3DDecoder()
        conv = torch.nn.Conv3d(24, 24, 1)
    # inv_freq is non-persistent: load_state_dict(assign=True) cannot populate it.
    # Reconstruct via the actual Core class, matching this pinned H3 architecture.
    decoder.pos_embed = RotaryEmbeddingND(48, rotary_base=100.0, n_dim=3)
    return decoder, conv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Probe source changed")
    bundle = Path(request["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (validate_request(manifest["source_request"]) != manifest["request_sha256"]
            or digest_file(bundle / "model.engine") != manifest["engine_sha256"]):
        raise ValueError("Engine bundle identity mismatch")
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True  # No implicit Comfy model management allocation.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Single-GPU probe required")
    torch.cuda.set_device(0)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() != manifest["gpu_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("Engine GPU identity mismatch")
    with safe_open(request["latent"], framework="pt", device="cpu") as file:
        if "latent_format_version_0" not in file.keys():
            raise ValueError("Unqualified latent serialization scale")
        shape = file.get_slice("latent_tensor").get_shape()
        z = file.get_slice("latent_tensor")[0:1, :, 0:7, 0:16, 0:16].half()
    if tuple(z.shape) != (1, 24, 7, 16, 16):
        raise ValueError("Saved latent cannot supply the standard tile")
    with safe_open(request["native_vae"], framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["minimax_h3_video_vae"])
        decoder_state = {key.removeprefix("decoder."): file.get_tensor(key)
                         for key in file.keys() if key.startswith("decoder.")}
        conv_state = {key.removeprefix("post_quant_conv."): file.get_tensor(key)
                      for key in file.keys() if key.startswith("post_quant_conv.")}
    mean = torch.tensor(metadata["latents_mean"], dtype=torch.float16).view(1, 24, 1, 1, 1)
    std = torch.tensor(metadata["latents_std"], dtype=torch.float16).view(1, 24, 1, 1, 1)
    raw = z * std + mean
    save_file({"raw_tile": raw}, str(root / "input-tile.safetensors"))
    print("T8_TRT_TILE_PHASE native_loading", flush=True)
    decoder, conv = native_decoder_shell()
    decoder.load_state_dict(decoder_state, strict=True, assign=True)
    conv.load_state_dict(conv_state, strict=True, assign=True)
    del decoder_state, conv_state
    if any(t.is_meta for module in (decoder, conv) for t in (*module.parameters(), *module.buffers())):
        raise RuntimeError("Unmaterialized native VAE tensor before device transfer")
    decoder, conv = decoder.to(device="cuda:0", dtype=torch.float16).eval(), conv.to(device="cuda:0", dtype=torch.float16).eval()
    print("T8_TRT_TILE_PHASE native_forward", flush=True)
    with torch.inference_mode():
        value = raw.to("cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        native = decoder(conv(value))
        torch.cuda.synchronize()
        native_seconds = time.perf_counter() - started
        native = native.cpu()
    del decoder, conv, value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    save_file({"raw_rgb": native}, str(root / "native.safetensors"))
    print("T8_TRT_TILE_PHASE trt_loading", flush=True)
    sys.path.insert(0, request["runtime_site"])
    import tensorrt as trt
    if trt.__version__ != manifest["runtime_version"]:
        raise RuntimeError("TRT version mismatch")
    mapped = loaded_windows_libraries()
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll"):
        if not Path(mapped[name]).is_relative_to(Path(request["runtime_site"]).resolve()):
            raise RuntimeError("TRT library identity mismatch")
    package = types.ModuleType("t8_trt_probe")
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    runner_type = importlib.import_module("t8_trt_probe.trt_vae_engine").TileEngineRunner
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((bundle / "model.engine").read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create engine context")
    with runner_type(engine, context, trt, device="cuda:0") as runner:
        print("T8_TRT_TILE_PHASE trt_forward", flush=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        candidate = runner(raw)
        torch.cuda.synchronize()
        trt_seconds = time.perf_counter() - started
        candidate = candidate.cpu()
        actual_calls = runner.calls
    del context, engine, runtime
    gc.collect()
    torch.cuda.empty_cache()
    save_file({"raw_rgb": candidate}, str(root / "trt.safetensors"))
    diff = candidate.float() - native.float()
    mse = diff.square().mean().item()
    rms = native.float().square().mean().sqrt().item()
    result = {"scope": "single real saved-latent 256px tile, not complete video/0.5MP/human or speed qualification",
              "source_latent_shape": shape, "tile_shape": list(raw.shape),
              "native_shape": list(native.shape), "trt_shape": list(candidate.shape), "actual_engine_calls": actual_calls,
              "native_finite": bool(torch.isfinite(native).all()), "trt_finite": bool(torch.isfinite(candidate).all()),
              "raw_rmse": mse ** .5, "relative_raw_rmse": (mse ** .5 / rms) if rms else None,
              "raw_max_absolute_error": diff.abs().max().item(),
              "native_first_call_seconds": native_seconds, "trt_first_call_including_copy_seconds": trt_seconds,
              "timing_warning": "Unequal transfer scope, first calls, no warmup: not a speedup measurement",
              "loaded_libraries": mapped, "engine_sha256": manifest["engine_sha256"]}
    result["functional_pass"] = result["native_finite"] and result["trt_finite"] and candidate.shape == native.shape
    write_new_json(root / "result.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["functional_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
