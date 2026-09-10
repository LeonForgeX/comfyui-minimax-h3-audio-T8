"""Complete saved-latent decode: actual Core reference then independent TRT assembly."""

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
from trt_vae_contract import output_bytes, output_shape, required_tile_shapes, temporal_plan, tile_axis  # noqa: E402


def native_video_shell():
    """Actual Core constructor/methods, without allocating unused encoder weights."""
    import torch
    from comfy.ldm.minimax.vae import (
        IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD,
        MiniMaxH3VideoVAE, RotaryEmbeddingND,
    )

    with torch.device("meta"):
        core = MiniMaxH3VideoVAE()
    del core.encoder, core.quant_conv
    core.decoder.pos_embed = RotaryEmbeddingND(48, rotary_base=100.0, n_dim=3)
    core.latents_mean = torch.tensor(LATENTS_MEAN)
    core.latents_std = torch.tensor(LATENTS_STD)
    core.pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    core.pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    return core


def check_geometry(shape, budget):
    from trt_vae_build import INPUT_SHAPE

    if output_bytes(shape) > budget:
        raise ValueError("Complete video exceeds explicit CPU output budget")
    if shape[0] != 1:
        raise ValueError("Initial complete reference probe requires batch 1")
    if set(required_tile_shapes(shape)) != {INPUT_SHAPE}:
        raise ValueError("This engine cannot decode the required real tile shapes")
    plan = temporal_plan(shape[2])
    expected_calls = plan.windows * len(tile_axis(shape[3] * 16).starts) * len(tile_axis(shape[4] * 16).starts)
    return output_shape(shape), expected_calls


def warmup_count(request):
    count = request.get("warmup_complete_decodes", 0)
    if type(count) is not int or count not in (0, 1):
        raise ValueError("Warmup requires explicit zero or one complete decode")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    warmups = warmup_count(request)
    root = args.request.parent
    for path, digest in request["sources"].items():
        if digest_file(path) != digest:
            raise ValueError("Video probe source changed")
    bundle = Path(request["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (validate_request(manifest["source_request"]) != manifest["request_sha256"]
            or digest_file(bundle / "model.engine") != manifest["engine_sha256"]):
        raise ValueError("Engine bundle identity mismatch")
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True  # No implicit Comfy model-management GPU allocation.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(request["latent"], framework="pt", device="cpu") as file:
        if "latent_format_version_0" not in file.keys():
            raise ValueError("Unknown SaveLatent serialization scale")
        z = file.get_tensor("latent_tensor").half()
    shape, expected_calls = check_geometry(tuple(z.shape), request["output_budget_bytes"])
    if not bool(torch.isfinite(z).all()):
        raise ValueError("Nonfinite source latent")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Single-GPU probe required")
    torch.cuda.set_device(0)
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() != manifest["gpu_uuid"].removeprefix("GPU-").lower():
        raise RuntimeError("Engine GPU identity mismatch")
    native_load_started = time.perf_counter()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as file:
        metadata = json.loads(file.metadata()["minimax_h3_video_vae"])
        state = {key: file.get_tensor(key) for key in file.keys()
                 if key.startswith(("decoder.", "post_quant_conv."))}
    core = native_video_shell()
    for key in ("latents_mean", "latents_std"):
        if not torch.equal(getattr(core, key), torch.tensor(metadata[key])):
            raise ValueError("Native Core and checkpoint latent normalization differ")
        state[key] = getattr(core, key)
    core.load_state_dict(state, strict=True, assign=True)
    del state
    if any(t.is_meta for t in (*core.parameters(), *core.buffers())):
        raise RuntimeError("Unmaterialized native reference tensor")
    # Preserve float32 normalization constants; only inference modules become half.
    core.decoder = core.decoder.to(device="cuda:0", dtype=torch.float16).eval()
    core.post_quant_conv = core.post_quant_conv.to(device="cuda:0", dtype=torch.float16).eval()
    if tuple(core.decode_output_shape(z.shape)) != shape:
        raise ValueError("Actual Core output geometry differs")
    torch.cuda.synchronize()
    native_load_seconds = time.perf_counter() - native_load_started
    save_file({"normalized_latent": z}, str(root / "input-latent.safetensors"))
    events = []

    def phase(name):
        events.append({"phase": name, "monotonic": time.perf_counter()})
        print("T8_TRT_VIDEO_PHASE " + name, flush=True)

    native_calls = 0

    def count_native(module, inputs):
        nonlocal native_calls
        native_calls += 1

    hook = core.decoder.register_forward_pre_hook(count_native)
    with torch.inference_mode():
        native_warmup_seconds = []
        for _ in range(warmups):
            phase("native_complete_warmup")
            torch.cuda.synchronize()
            started = time.perf_counter()
            warm = core.decode(z.to("cuda:0"), output_buffer=torch.empty(shape, dtype=torch.float32, device="cpu"))
            torch.cuda.synchronize()
            native_warmup_seconds.append(time.perf_counter() - started)
            if not bool(torch.isfinite(warm).all()):
                raise ValueError("Nonfinite native warmup")
            del warm
        native_warmup_calls = native_calls
        if native_warmup_calls != warmups * expected_calls:
            raise ValueError("Native warmup tile accounting mismatch")
        native_calls = 0
        phase("native_complete_decode")
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = z.to("cuda:0")
        native = core.decode(value, output_buffer=torch.empty(shape, dtype=torch.float32, device="cpu"))
        torch.cuda.synchronize()
        native_seconds = time.perf_counter() - start
    hook.remove()
    del core, value
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if tuple(native.shape) != shape or not bool(torch.isfinite(native).all()):
        raise ValueError("Invalid native complete output")
    save_file({"rgb": native.contiguous()}, str(root / "native-rgb.safetensors"))
    del native
    gc.collect()
    phase("trt_loading_after_native_unload")
    trt_load_started = time.perf_counter()
    sys.path.insert(0, request["runtime_site"])
    import tensorrt as trt
    if trt.__version__ != manifest["runtime_version"]:
        raise RuntimeError("TRT runtime mismatch")
    mapped = loaded_windows_libraries()
    for name in ("nvinfer_10.dll", "nvonnxparser_10.dll"):
        if not Path(mapped[name]).is_relative_to(Path(request["runtime_site"]).resolve()):
            raise RuntimeError("TRT DLL outside isolated runtime")
    package = types.ModuleType("t8_trt_video_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    runner_type = importlib.import_module(package.__name__ + ".trt_vae_engine").TileEngineRunner
    decode = importlib.import_module(package.__name__ + ".trt_vae_decode").decode_raw_latent
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine((bundle / "model.engine").read_bytes())
    if engine is None:
        raise RuntimeError("Engine deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Engine context creation failed")

    def cancel_check():
        if (root / "cancel.request").exists():
            raise InterruptedError("Video probe cancelled")

    with runner_type(engine, context, trt, device="cuda:0") as runner, torch.inference_mode():
        torch.cuda.synchronize()
        trt_load_seconds = time.perf_counter() - trt_load_started
        trt_warmup_seconds = []
        for _ in range(warmups):
            phase("trt_complete_warmup")
            torch.cuda.synchronize()
            started = time.perf_counter()
            warm_value = z.to("cuda:0")
            warm_mean = warm_value.new_tensor(metadata["latents_mean"]).view(1, 24, 1, 1, 1)
            warm_std = warm_value.new_tensor(metadata["latents_std"]).view(1, 24, 1, 1, 1)
            warm = decode(warm_value * warm_std + warm_mean, runner,
                          max_output_bytes=request["output_budget_bytes"], check=cancel_check)
            torch.cuda.synchronize()
            trt_warmup_seconds.append(time.perf_counter() - started)
            if not bool(torch.isfinite(warm).all()):
                raise ValueError("Nonfinite TRT warmup")
            del warm, warm_value, warm_mean, warm_std
        trt_warmup_calls = runner.calls
        if trt_warmup_calls != warmups * expected_calls:
            raise ValueError("TRT warmup tile accounting mismatch")
        phase("trt_complete_decode")
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = z.to("cuda:0")
        mean = value.new_tensor(metadata["latents_mean"]).view(1, 24, 1, 1, 1)
        std = value.new_tensor(metadata["latents_std"]).view(1, 24, 1, 1, 1)
        candidate = decode(value * std + mean, runner, max_output_bytes=request["output_budget_bytes"], check=cancel_check)
        torch.cuda.synchronize()
        trt_seconds = time.perf_counter() - start
        trt_calls = runner.calls - trt_warmup_calls
    del context, engine, runtime, value, mean, std
    gc.collect()
    torch.cuda.empty_cache()
    if tuple(candidate.shape) != shape or not bool(torch.isfinite(candidate).all()):
        raise ValueError("Invalid TRT complete output")
    if native_calls != expected_calls or trt_calls != expected_calls:
        raise ValueError("Actual tile execution count differs from complete plan")
    save_file({"rgb": candidate.contiguous()}, str(root / "trt-rgb.safetensors"))
    phase("complete_rgb_saved")
    result = {"status": "complete_rgb_requires_independent_audit", "source_latent_shape": list(z.shape),
              "output_shape": list(shape), "expected_tile_calls": expected_calls,
              "native_calls": native_calls, "trt_calls": trt_calls,
              "warmup_complete_decodes_per_route": warmups,
              "native_warmup_calls": native_warmup_calls, "trt_warmup_calls": trt_warmup_calls,
              "native_warmup_seconds": native_warmup_seconds, "trt_warmup_seconds": trt_warmup_seconds,
              "native_weight_materialization_seconds": native_load_seconds,
              "trt_runtime_engine_loading_seconds": trt_load_seconds,
              "native_measured_complete_decode_seconds": native_seconds,
              "trt_measured_complete_decode_seconds": trt_seconds,
              "native_first_complete_decode_seconds": native_seconds if not warmups else None,
              "trt_first_complete_decode_seconds": trt_seconds if not warmups else None,
              "timing_scope": "CPU half latent to CPU float RGB, normalization and GPU transfer included, loading/saving excluded",
              "timing_limit": "One measured decode per route; warmups separately recorded. Repeated alternating runs and independent benchmark audit still required.",
              "native_reference": "Actual installed Core MiniMaxH3VideoVAE.decode, PyTorch attention, FP16 decoder; no encoder allocation",
              "scope": "Complete saved latent only; source-route/media/quality qualification require independent audits. No long-video qualification claim.",
              "loaded_libraries": mapped, "events": events,
              "files": {name: digest_file(root / name) for name in
                        ("input-latent.safetensors", "native-rgb.safetensors", "trt-rgb.safetensors")}}
    write_new_json(root / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
