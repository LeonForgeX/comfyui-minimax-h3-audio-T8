"""Real Core VAE wrapping, scoped TRT loading/cleanup and complete short decode.

Called only by run_trt_vae_video_probe's OS-lease/Job/resource-guard controller.
No sampler, browser, installation or native decode rerun.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf8"))
    root = args.request.parent
    if not request.get("interface_probe") or request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the explicit owned interface probe controller")
    for path, sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError(f"Interface probe source changed: {Path(path).name}")
    sys.path.insert(0, request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    torch.set_num_threads(2)
    package = types.ModuleType("t8_trt_interface_probe")
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_type = importlib.import_module(package.__name__ + ".trt_vae_backend").ScopedBackend
    interface_type = importlib.import_module(package.__name__ + ".trt_vae_interface").H3VAEInterface
    with safe_open(request["latent"], framework="pt", device="cpu") as saved:
        if "latent_format_version_0" not in saved.keys():
            raise ValueError("Unexpected latent serialization scale")
        z = saved.get_tensor("latent_tensor")
    if tuple(z.shape) != (1, 24, 22, 32, 64):
        raise ValueError("This lifecycle probe is restricted to saved 0.5MP/73-frame short latent")
    print("T8_TRT_INTERFACE_PHASE loading_original_core_vae_on_cpu", flush=True)
    state, metadata = comfy.utils.load_torch_file(request["native_vae"], return_metadata=True)
    native = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cuda:0"), dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    if any(t.device.type != "cpu" for t in (*native.first_stage_model.parameters(), *native.first_stage_model.buffers())):
        raise ValueError("Original VAE unexpectedly materialized on GPU before TRT admission")
    def check():
        if (root / "cancel.request").exists():
            raise InterruptedError("Interface probe cancelled by user/controller")
    backend = backend_type({"decoder": request["bundle"]}, request["runtime_site"], check=check,
                           # Controller owns the shared TRT/FI locks for this Job.
                           serial_lease=lambda: SerialProbeLease(Path(tempfile.gettempdir()) / "T8-TRT-VAE-interface-worker.lock"))
    mean = native.first_stage_model.latents_mean.half().view(1, 24, 1, 1, 1)
    std = native.first_stage_model.latents_std.half().view(1, 24, 1, 1, 1)
    raw_tile = z[:, :, :7, :16, :16].half() * std + mean
    print("T8_TRT_INTERFACE_PHASE controlled_cancel_after_two_tiles", flush=True)
    try:
        with backend("decoder", [(1, 24, 7, 16, 16)]) as decode:
            for _ in range(2):
                result = decode(raw_tile)
                if result.device.type != "cpu" or not bool(torch.isfinite(result).all()):
                    raise ValueError("Tile did not return finite CPU data")
                del result
            raise InterruptedError("intentional_lifecycle_probe_cancel")
    except InterruptedError as error:
        if str(error) != "intentional_lifecycle_probe_cancel":
            raise
    cancelled = copy.deepcopy(backend.last_report)
    if cancelled["status"] != "failed" or cancelled["free_before_deserialize"] - cancelled["free_after_release"] > 512 * 1024**2:
        raise ValueError("Cancelled engine/context did not return observed memory within tolerance")
    # A distinct explicit second operation, not a hidden retry after failure.
    print("T8_TRT_INTERFACE_PHASE explicit_complete_short_decode", flush=True)
    vae = interface_type(native, backend, max_output_bytes=request["output_budget_bytes"], check=check)
    rgb = vae.decode(z)
    if tuple(rgb.shape) != (1, 73, 512, 1024, 3) or rgb.device.type != "cpu" or rgb.dtype != torch.float32 or not bool(torch.isfinite(rgb).all()):
        raise ValueError("Standard VAE output layout/dtype/values differ from Core contract")
    complete = copy.deepcopy(backend.last_report)
    if complete["status"] != "complete" or complete["calls"] != 60:
        raise ValueError("Complete interface did not execute all60 decoder tiles")
    if complete["free_before_deserialize"] - complete["free_after_release"] > 512 * 1024**2:
        raise ValueError("Complete operation did not release observed GPU memory within tolerance")
    candidate = rgb.movedim(-1, 1).contiguous()
    reference = load_file(request["reference_rgb"], device="cpu")["rgb"]
    if candidate.shape != reference.shape:
        raise ValueError("Saved same-latent reference has different dimensions")
    error = candidate - reference
    report = {"status": "real_scoped_interface_complete_requires_independent_audit", "cancelled_lease": cancelled,
              "complete_lease": complete, "vae_interface": vae.last_report,
              "standard_output_shape": list(rgb.shape), "candidate_shape": list(candidate.shape),
              "max_abs_vs_previous_trt": float(error.abs().max()), "mse_vs_previous_trt": float(error.square().mean()),
              "native_model_class": type(native).__module__ + "." + type(native).__name__,
              "native_stage_class": type(native.first_stage_model).__name__,
              "scope": "Actual Core VAE wrapped, two-tile deliberate cancellation then distinct full73-frame decode. No native rerun, sampling, long-video, public-node, encoder or human acceptance."}
    save_file({"rgb": candidate}, str(root / "interface-rgb.safetensors"))
    report["files"] = {"interface-rgb.safetensors": digest_file(root / "interface-rgb.safetensors")}
    write_new_json(root / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
