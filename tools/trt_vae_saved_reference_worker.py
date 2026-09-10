"""Actual scoped whole short decode; reuse identity-bound native RGB on CPU."""
import argparse
import copy
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
from tools.trt_vae_saved_reference import bind_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf8"))
    if request.get("saved_reference_probe") is not True or request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the owned saved-reference controller")
    for path, sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError("Source changed: " + Path(path).name)
    bound = bind_reference(request["reference_rgb"], request["latent"], request["native_vae"])
    if bound != request["reference_sources"]:
        raise ValueError("Native reference binding changed")
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(2)
    package = types.ModuleType("t8_trt_saved_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_type = importlib.import_module(package.__name__ + ".trt_vae_backend").ScopedBackend
    interface_type = importlib.import_module(package.__name__ + ".trt_vae_interface").H3VAEInterface
    z = load_file(request["latent"], device="cpu")["latent_tensor"]
    print("T8_SAVED_REFERENCE_PHASE native_vae_metadata_on_cpu", flush=True)
    state, metadata = comfy.utils.load_torch_file(request["native_vae"], return_metadata=True)
    native = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cuda:0"), dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    if any(t.device.type != "cpu" for t in (*native.first_stage_model.parameters(), *native.first_stage_model.buffers())):
        raise ValueError("Native VAE unexpectedly occupies GPU before scoped TRT load")
    def check():
        if (root / "cancel.request").exists():
            raise InterruptedError("Saved-reference decode cancelled")
    backend = backend_type({"decoder": request["bundle"]}, request["runtime_site"], check=check,
                           serial_lease=lambda: SerialProbeLease(Path(tempfile.gettempdir())/"T8-TRT-VAE-saved-reference-worker.lock"))
    vae = interface_type(native, backend, max_output_bytes=request["output_budget_bytes"], check=check)
    print("T8_SAVED_REFERENCE_PHASE complete73_frame_scoped_decode", flush=True)
    start = time.perf_counter()
    rgb = vae.decode(z)
    seconds = time.perf_counter() - start
    if tuple(rgb.shape) != (1,73,512,1024,3) or rgb.dtype != torch.float32 or rgb.device.type != "cpu" or not bool(torch.isfinite(rgb).all()):
        raise ValueError("Complete standard VAE RGB contract failed")
    lease = copy.deepcopy(backend.last_report)
    if lease["status"] != "complete" or lease["calls"] != 60 or lease["free_before_deserialize"] - lease["free_after_release"] > 512*1024**2:
        raise ValueError("Incomplete60-tile execution or release")
    start = time.perf_counter()
    save_file({"rgb": rgb.movedim(-1,1).contiguous()}, str(root / "trt-rgb.safetensors"))
    save_seconds = time.perf_counter() - start
    report = {"status": "saved_native_reference_complete_decode_requires_audit", "lease": lease,
              "standard_output_shape": list(rgb.shape), "native_model_class": type(native).__module__+"."+type(native).__name__,
              "complete_scoped_decode_seconds": seconds, "rgb_save_seconds": save_seconds,
              "interface_report": vae.last_report, "rgb_sha256": digest_file(root / "trt-rgb.safetensors"),
              "limits": "Actual scoped complete0.5MP/73-frame decode. Same native RGB reused, not rerun. One call including runtime verification/load/release, no comparative speed claim, sampling, audio or human acceptance. No long-video test."}
    write_new_json(root / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
