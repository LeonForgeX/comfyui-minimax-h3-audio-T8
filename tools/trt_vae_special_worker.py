"""Actual39-frame Core/scoped/bounded outpaint decode; never sample a new video."""
import argparse
import copy
from contextlib import contextmanager
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    root = args.request.parent
    request = json.loads(args.request.read_text(encoding="utf8"))
    if request.get("special_interface_probe") is not True or request.get("serial_leases_held_by_controller") is not True:
        raise ValueError("Use the owned special-interface controller")
    for path,sha in request["sources"].items():
        if digest_file(path) != sha:
            raise ValueError("Special interface source changed: "+Path(path).name)
    sys.path.insert(0,request["core_root"])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("t8_trt_special_probe")
    package.__path__ = [str(PROJECT / "h3_t8"), str(PROJECT)]
    sys.modules[package.__name__] = package
    backend_type = importlib.import_module(package.__name__+".trt_vae_backend").ScopedBackend
    interface_type = importlib.import_module(package.__name__+".trt_vae_interface").H3VAEInterface
    build_plan = importlib.import_module(package.__name__+".video_outpaint_plan").build_outpaint_plan
    bounded_decode = importlib.import_module(package.__name__+".video_outpaint_decode").iter_decode_outpaint_shot
    with safe_open(request["latent"],framework="pt",device="cpu") as f:
        if "latent_format_version_0" not in f.keys() or f.get_slice("latent_tensor").get_shape() != [1,24,22,32,64]:
            raise ValueError("Use the actual saved0.5MP73-frame latent")
        z = f.get_tensor("latent_tensor")[:,:,:12].contiguous()
    save_file({"normalized_latent":z},str(root/"input-latent.safetensors"))
    state,metadata = comfy.utils.load_torch_file(request["native_vae"],return_metadata=True)
    native = comfy.sd.VAE(sd=state,metadata=metadata,device=torch.device("cuda:0"),dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    count = 0
    def hook(module,inputs):
        nonlocal count
        count += 1
    handle = native.first_stage_model.decoder.register_forward_pre_hook(hook)
    print("T8_SPECIAL_PHASE actual_core_standard_decode39frames",flush=True)
    with torch.inference_mode():
        reference = native.decode(z).cpu()
    handle.remove()
    if count != 30 or tuple(reference.shape) != (1,39,512,1024,3):
        raise ValueError("Actual native full prefix output/tile contract differs")
    buffers = {name:getattr(native.first_stage_model,name).detach().cpu().clone() for name in
               ("pixel_mean","pixel_std","latents_mean","latents_std")}
    save_file(buffers,str(root/"actual-core-buffers.safetensors"))
    mm.unload_all_models()
    native.first_stage_model.to(device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    def check():
        if (root/"cancel.request").exists():
            raise InterruptedError("Special interface probe cancelled")
    backend = backend_type({"decoder":request["bundle"]},request["runtime_site"],check=check,
              serial_lease=lambda:SerialProbeLease(Path(tempfile.gettempdir())/"T8-TRT-VAE-special-worker.lock"))
    leases = []
    @contextmanager
    def leased(kind,shapes):
        try:
            with backend(kind,shapes) as call:
                yield call
        finally:
            leases.append(copy.deepcopy(backend.last_report))
    vae = interface_type(native,leased,max_output_bytes=request["output_budget_bytes"],check=check)
    print("T8_SPECIAL_PHASE trt_standard_decode39frames",flush=True)
    standard = vae.decode(z)
    plan = build_plan(source_sha256=digest_file(request["latent"]),width=1024,height=512,frame_count=39,
                      aspect="source",generation_megapixels=0,window_frames=39)
    write_new_json(root/"plan.json",plan)
    reads = []
    def read(a,b):
        reads.append([a,b])
        return z[:,:,a:b]
    print("T8_SPECIAL_PHASE actual_bounded_outpaint_decode39frames",flush=True)
    chunks = list(bounded_decode(vae,read,plan,interrupt_check=check))
    bounded = torch.cat([frames for _,frames,_ in chunks]).unsqueeze(0)
    if [v["calls"] for v in leases] != [30,15,15] or reads != [[0,7],[5,12]]:
        raise ValueError("Standard/bounded did not execute real native30/15/15tile windows")
    for lease in leases:
        if lease["status"] != "complete" or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2:
            raise ValueError("Special interface lease cleanup failed")
    values = {"native":reference,"trt_standard":standard,"trt_bounded":bounded}
    for v in values.values():
        if tuple(v.shape) != (1,39,512,1024,3) or v.dtype != torch.float32 or v.device.type != "cpu" or not bool(torch.isfinite(v).all()):
            raise ValueError("Actual special interface RGB invalid")
    save_file({k:v.movedim(-1,1).contiguous() for k,v in values.items()},str(root/"outputs.safetensors"))
    report = {"status":"actual_short_special_interfaces_require_independent_audit","native_calls":count,"leases":leases,
              "reads":reads,"native_model_class":type(native).__module__+"."+type(native).__name__,
              "standard_bounded_bit_equal":torch.equal(standard,bounded),
              "chunks":[{"start":start,"frames":len(frames),"report":detail} for start,frames,detail in chunks],
              "buffer_dtypes":{k:str(v.dtype) for k,v in buffers.items()},
              "files":{n:digest_file(root/n) for n in ("input-latent.safetensors","actual-core-buffers.safetensors","outputs.safetensors","plan.json")},
              "limits":"True Core standard decode vs TRT standard and existing bounded outpaint decoder on saved39-frame prefix. No new sampling/extension, FaceRefine, audio, human or long-video acceptance."}
    write_new_json(root/"result.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
