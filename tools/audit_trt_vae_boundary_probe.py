"""Independent CPU audit of all saved native/TRT spatial and temporal cases."""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, validate_request, write_new_json, DECODER_FLEX_MODEL_SHA  # noqa: E402


def audit(root):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root / name).read_text(encoding="utf8"))
    request, result = read("request.json"), read("result.json")
    job, guard = read("terminal.json")["isolated"], read("resource-summary.json")
    if job["status"] != "complete" or job["exit_code"] or job["active_after_cleanup"] or not job["job_assigned_before_task"]:
        raise ValueError("Boundary Job did not finish cleanly")
    if guard["status"] != "observations_within_policy":
        raise ValueError("Boundary guard failed")
    for path, sha in request["sources"].items():
        source = Path(path)
        if source.suffix == ".py" and source.is_relative_to(PROJECT):
            source = root / "source-snapshot" / (sha + "-" + source.name)
        if digest_file(source) != sha:
            raise ValueError("Boundary source/snapshot changed")
    bundle = Path(request["bundle"])
    manifest = json.loads((bundle/"manifest.json").read_text(encoding="utf8"))
    build = manifest["source_request"]
    if build["model_sha256"] != DECODER_FLEX_MODEL_SHA or validate_request(build) != manifest["request_sha256"] or digest_file(bundle/"model.engine") != manifest["engine_sha256"]:
        raise ValueError("Different flex engine")
    if digest_file(request["native_vae"]) != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Unexpected native VAE")
    for name, sha in result["files"].items():
        if name not in ("inputs.safetensors","native-rgb.safetensors","trt-rgb.safetensors") or digest_file(root/name) != sha:
            raise ValueError("Boundary saved evidence changed")
    inputs = load_file(str(root/"inputs.safetensors"))
    native = load_file(str(root/"native-rgb.safetensors"))
    trt = load_file(str(root/"trt-rgb.safetensors"))
    with safe_open(request["latent"], framework="pt", device="cpu") as saved:
        if "latent_format_version_0" not in saved.keys():
            raise ValueError("Unknown saved latent convention")
        video = saved.get_tensor("latent_tensor")
    image_path = Path(request["image_latent"])
    image_audit = json.loads((image_path.parent/"independent-t1-audit.json").read_text(encoding="utf8"))
    if image_audit["status"] != "t1_actual_engine_and_full_image_api_bound_not_human_qualified" or digest_file(image_path) != image_audit["outputs_sha256"]:
        raise ValueError("Image latent is not the audited genuine image encoding")
    image = load_file(str(image_path))["native_latent"]
    spec = {"image_full": (image,1,32,64,1), "image_small":(image,1,8,13,1), "image_min":(image,1,1,1,1),
            "video_small":(video,7,8,13,22), "video_min":(video,7,1,1,22),
            "video_subtile":(video,7,15,16,22), "video_seam":(video,7,17,19,22)}
    for t, frames in ((2,5),(5,17),(6,18),(7,22),(8,26),(11,35),(12,39)):
        spec[f"video_t{t}"] = (video,t,16,16,frames)
    if set(inputs) != set(spec) or set(native) != set(spec) or set(trt) != set(spec):
        raise ValueError("Missing or unexpected boundary case")
    rows = []
    for name, (source,t,h,w,frames) in spec.items():
        if not torch.equal(inputs[name], source[:,:,:t,:h,:w].half()):
            raise ValueError("Boundary source is not the declared exact latent region")
        expected = (1,3,frames,h*16,w*16)
        for output in (native[name],trt[name]):
            if tuple(output.shape) != expected or output.dtype != torch.float32 or not bool(torch.isfinite(output).all()) or float(output.min()) < 0 or float(output.max()) > 1:
                raise ValueError("Boundary RGB size/dtype/range/finiteness failed")
        if result["native_calls"][name] != result["trt_calls"][name] or len(result["actual_trt_shapes"][name]) != result["trt_calls"][name]:
            raise ValueError("Native/TRT actual tile calls differ")
        for shape in result["actual_trt_shapes"][name]:
            if shape[:2] != [1,24] or shape[2] != (1 if t == 1 else 7) or not 1 <= shape[3] <= 16 or not 1 <= shape[4] <= 16:
                raise ValueError("Boundary tiles were padded to an incompatible profile")
        error = trt[name].double()-native[name].double()
        mse = float(error.square().mean())
        rows.append({"case":name,"frames":frames,"shape":list(expected),"calls":result["trt_calls"][name],
                     "mse":mse,"psnr_db":-10*math.log10(mse) if mse else None,"max_abs":float(error.abs().max())})
    lease = result["lease"]
    if lease["status"] != "complete" or lease["calls"] != sum(r["calls"] for r in rows) or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2:
        raise ValueError("Boundary lease calls/cleanup failed")
    if torch.cuda.is_initialized():
        raise ValueError("Independent auditor must stay CPU-only")
    return {"status":"native_and_flex_all14_boundary_cases_bound_not_human_qualified", "rows":rows,
            "resource_summary":guard,"engine_sha256":manifest["engine_sha256"],"cuda_initialized":False,
            "limits":"Actual same-latent geometry/finiteRGB/numerical/Job audit. Cropped and temporal-prefix cases are boundary fixtures, not independent generated videos or long-video qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    write_new_json(args.run_dir/"independent-boundary-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
