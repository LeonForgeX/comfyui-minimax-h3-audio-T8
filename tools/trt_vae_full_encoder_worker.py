"""Actual native/scoped Core full73-frame video encode and profile routing."""
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
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf8"))
    if request.get("video_interface_probe") is not True or request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the owned video encoder controller")
    for path, sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError("Video encoder source changed: " + Path(path).name)
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
    package = types.ModuleType("t8_trt_video_encoder_probe")
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_module = importlib.import_module(package.__name__+".trt_vae_backend")
    interface_type = importlib.import_module(package.__name__+".trt_vae_interface").H3VAEInterface
    video = load_file(request["source_rgb8"], device="cpu")["video"]
    if video.dtype != torch.uint8 or tuple(video.shape) != (1,3,73,512,1024):
        raise ValueError("Use the actual saved complete0.5MP/73-frame RGB source")
    pixels = video[0].movedim(0,-1).float()/255
    bundles = [request["t1_bundle"], request["bundle"]]
    selected = {}
    for name, shape in (("image",(1,3,1,256,256)), ("video",(1,3,17,256,256))):
        path, manifest = backend_module.select_bundle(bundles,"encoder",[shape])
        selected[name] = {"bundle": str(path), "engine_sha256": manifest["engine_sha256"]}
    if Path(selected["image"]["bundle"]) != Path(request["t1_bundle"]) or Path(selected["video"]["bundle"]) != Path(request["bundle"]):
        raise ValueError("Actual bundle selection did not preserve T1/T17 separation")
    state, metadata = comfy.utils.load_torch_file(request["native_vae"], return_metadata=True)
    native = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cuda:0"), dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    calls = 0
    def count(module, inputs):
        nonlocal calls
        calls += 1
    hook = native.first_stage_model.encoder.register_forward_pre_hook(count)
    print("T8_VIDEO_ENCODER_PHASE actual_core_encode73frames", flush=True)
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter()
        reference = native.encode(pixels).cpu()
        torch.cuda.synchronize()
        native_seconds = time.perf_counter()-start
    hook.remove()
    if calls != 75:
        raise ValueError("Actual native video encode did not execute75tiles")
    mm.unload_all_models()
    native.first_stage_model.to(device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    def check():
        if (root / "cancel.request").exists():
            raise InterruptedError("Video encoder cancelled")
    backend = backend_module.ScopedBackend({"encoder": bundles}, request["runtime_site"], check=check,
                serial_lease=lambda: SerialProbeLease(Path(tempfile.gettempdir())/"T8-TRT-VAE-video-encoder-worker.lock"))
    vae = interface_type(native, backend, encode_backend="trt", max_output_bytes=64*1024**2, check=check)
    print("T8_VIDEO_ENCODER_PHASE actual_scoped_encode73frames", flush=True)
    start = time.perf_counter()
    candidate = vae.encode(pixels)
    trt_seconds = time.perf_counter()-start
    lease = copy.deepcopy(backend.last_report)
    if (lease["status"] != "complete" or lease["calls"] != 75 or lease["engine_sha256"] != selected["video"]["engine_sha256"]
            or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2):
        raise ValueError("Actual75tile selected profile execution/release failed")
    for value in (reference, candidate):
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != (1,24,22,32,64) or not bool(torch.isfinite(value).all()):
            raise ValueError("Full video latent API output invalid or time cropped")
    save_file({"native_latent": reference, "trt_latent": candidate}, str(root / "outputs.safetensors"))
    report = {"status": "actual_full_video_encoder_interface_requires_independent_audit", "selected_profiles": selected,
              "native_calls": calls, "lease": lease, "native_model_class": type(native).__module__+"."+type(native).__name__,
              "native_cold_encode_seconds": native_seconds, "trt_scoped_encode_seconds": trt_seconds,
              "outputs_sha256": digest_file(root/"outputs.safetensors"), "interface_report": vae.last_report,
              "limits": "One actual73-frame video API operation per route; native model loading vs TRT hash/load/release policies differ. Not repeated speed benchmark, new sampling/human/long-video acceptance. T1 actual encode not repeated; only profile selection rechecked."}
    write_new_json(root/"result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
