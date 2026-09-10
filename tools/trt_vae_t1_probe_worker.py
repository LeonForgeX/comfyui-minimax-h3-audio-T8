"""Real T1 moments plus whole0.5MP image through native/scoped actual VAE APIs."""
import argparse
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf8"))
    root = args.request.parent
    if request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the explicit owned T1 probe controller")
    for path, sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError("T1 probe source changed: " + Path(path).name)
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("t8_trt_t1_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_type = importlib.import_module(package.__name__ + ".trt_vae_backend").ScopedBackend
    interface_type = importlib.import_module(package.__name__ + ".trt_vae_interface").H3VAEInterface
    image = load_file(request["source_rgb8"], device="cpu")["image"]
    reference = load_file(request["reference_tile"], device="cpu")
    if image.dtype != torch.uint8 or tuple(image.shape) != (1, 3, 1, 512, 1024):
        raise ValueError("Expected actual saved0.5MP single image")
    pixels = image[0].movedim(0, -1).float() / 255
    print("T8_T1_PHASE actual_native_vae_image_encode", flush=True)
    state, metadata = comfy.utils.load_torch_file(request["native_vae"], return_metadata=True)
    native = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cuda:0"), dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    count = 0
    def count_native(module, inputs):
        nonlocal count
        count += 1
    hook = native.first_stage_model.encoder.register_forward_pre_hook(count_native)
    timings, natives, calls = [], [], []
    # Initial model-load call plus three hot complete-image calls, never sampling.
    with torch.inference_mode():
        for _ in range(4):
            count = 0
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = native.encode(pixels).cpu()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            calls.append(count)
            natives.append(result)
    hook.remove()
    if calls != [15]*4 or any(not torch.equal(natives[0], v) for v in natives[1:]):
        raise ValueError("Native image did not execute15tiles deterministically each time")
    # Only this owned process's models; no server or other application's state.
    mm.unload_all_models()
    native.first_stage_model.to(device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    def check():
        if (root / "cancel.request").exists():
            raise InterruptedError("T1 probe cancelled")
    backend = backend_type({"encoder": request["bundle"]}, request["runtime_site"], check=check,
                           serial_lease=lambda: SerialProbeLease(Path(tempfile.gettempdir()) / "T8-TRT-VAE-t1-worker.lock"))
    print("T8_T1_PHASE real_single_tile_no_temporal_padding", flush=True)
    with backend("encoder", [(1, 3, 1, 256, 256)]) as encode:
        moments = encode(reference["normalized_pixels"])
    tile_report = copy.deepcopy(backend.last_report)
    if tuple(moments.shape) != (1, 48, 1, 16, 16):
        raise ValueError("Actual T1 engine output differs from trace")
    vae = interface_type(native, backend, encode_backend="trt", max_output_bytes=64*1024**2, check=check)
    outputs, leases, wall = [], [], []
    print("T8_T1_PHASE actual_interface_complete_image_three_separate_leases", flush=True)
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs.append(vae.encode(pixels))
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - start)
        leases.append(copy.deepcopy(backend.last_report))
    for result in [*outputs, *natives]:
        if result.device.type != "cpu" or result.dtype != torch.float32 or tuple(result.shape) != (1,24,1,32,64) or not bool(torch.isfinite(result).all()):
            raise ValueError("Image VAE API latent shape/dtype/finiteness failure")
    if any(not torch.equal(outputs[0], value) for value in outputs[1:]):
        raise ValueError("T1 image encoding not deterministic")
    for lease in leases:
        if lease["calls"] != 15 or lease["status"] != "complete" or lease["free_before_deserialize"] - lease["free_after_release"] > 512*1024**2:
            raise ValueError("T1 full image lease calls/cleanup failed")
    save_file({"native_latent": natives[0], "trt_latent": outputs[0], "trt_moments": moments}, str(root / "outputs.safetensors"))
    report = {"status": "actual_t1_full_image_requires_independent_audit", "tile_lease": tile_report,
              "complete_leases": leases, "native_calls": calls,
              "native_first_call_seconds_including_model_load": timings[0],
              "native_hot_complete_image_seconds": timings[1:], "trt_scoped_complete_image_seconds": wall,
              "native_class": type(native).__module__ + "." + type(native).__name__,
              "interface_report": vae.last_report, "outputs_sha256": digest_file(root / "outputs.safetensors"),
              "limits": "One actual0.5MP still; each TRT API call includes runtime hash/engine load/release, native hot keeps model loaded. Not directly equivalent caching policies; no sampler/human quality, video or long-video claim."}
    write_new_json(root / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
