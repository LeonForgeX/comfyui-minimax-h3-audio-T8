"""CPU evidence audit of real T1 encoder and full actual VAE image API."""
import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, validate_request, write_new_json, ENCODER_T1_MODEL_SHA  # noqa: E402


def metrics(reference, candidate):
    import torch
    if reference.shape != candidate.shape or not reference.is_floating_point() or not candidate.is_floating_point():
        raise ValueError("Metrics require matching floating tensors")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("Nonfinite encoder evidence")
    reference, candidate = reference.double(), candidate.double()
    error = candidate - reference
    signal = reference.square().mean()
    if float(signal) == 0:
        raise ValueError("Zero reference energy")
    return {"relative_rmse_percent": 100*float((error.square().mean()/signal).sqrt()),
            "max_abs": float(error.abs().max()), "rmse": float(error.square().mean().sqrt())}


def audit(root):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    request = json.loads((root / "request.json").read_text(encoding="utf8"))
    report = json.loads((root / "result.json").read_text(encoding="utf8"))
    job = json.loads((root / "terminal.json").read_text(encoding="utf8"))["isolated"]
    guard = json.loads((root / "resource-summary.json").read_text(encoding="utf8"))
    if job["status"] != "complete" or job["exit_code"] or job["active_after_cleanup"] or not job["job_assigned_before_task"]:
        raise ValueError("T1 Job did not finish cleanly")
    if guard["status"] != "observations_within_policy":
        raise ValueError("T1 resource guard failed")
    for path, sha in request["sources"].items():
        source = Path(path)
        if source.suffix == ".py":
            source = root / "source-snapshot" / (sha + "-" + source.name)
        if digest_file(source) != sha:
            raise ValueError("T1 frozen source changed")
    if digest_file(request["native_vae"]) != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Native VAE differs")
    bundle = Path(request["bundle"])
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf8"))
    build = manifest["source_request"]
    if validate_request(build) != manifest["request_sha256"] or build["model_sha256"] != ENCODER_T1_MODEL_SHA:
        raise ValueError("Different T1 build graph")
    if digest_file(bundle / "model.engine") != manifest["engine_sha256"] or manifest["output_shape"] != [1,48,1,16,16]:
        raise ValueError("Different T1 engine/output")
    reference_path = Path(request["reference_tile"])
    export = json.loads((reference_path.parent / "independent-export-audit.json").read_text(encoding="utf8"))
    export_manifest = json.loads((reference_path.parent / "manifest.json").read_text(encoding="utf8"))
    if export["status"] != "t1_export_weights_and_actual_reference_bound_not_trt_qualified" or export["onnx_sha256"] != build["model_sha256"] or digest_file(reference_path) != export_manifest["reference_sha256"]:
        raise ValueError("Independent export/reference identity differs")
    reference = load_file(str(reference_path))
    image = load_file(request["source_rgb8"])["image"]
    raw = (image[...,128:384,384:640].float()/255*2-1).half()
    mean = raw.new_tensor((.485,.456,.406)).view(1,3,1,1,1)
    std = raw.new_tensor((.229,.224,.225)).view(1,3,1,1,1)
    if not torch.equal(((raw+1)*.5-mean)/std, reference["normalized_pixels"]):
        raise ValueError("Whole image does not contain the declared T1 reference tile")
    output_path = root / "outputs.safetensors"
    if digest_file(output_path) != report["outputs_sha256"]:
        raise ValueError("T1 outputs changed")
    outputs = load_file(str(output_path))
    for name in ("native_latent", "trt_latent"):
        if outputs[name].dtype != torch.float32 or tuple(outputs[name].shape) != (1,24,1,32,64):
            raise ValueError("Full image latent shape/dtype differs")
    if tuple(outputs["trt_moments"].shape) != (1,48,1,16,16) or outputs["trt_moments"].dtype != torch.float16:
        raise ValueError("T1 real moments shape/dtype differs")
    if report["native_class"] != "comfy.sd.VAE" or report["native_calls"] != [15]*4:
        raise ValueError("Native API/call evidence differs")
    leases = [report["tile_lease"], *report["complete_leases"]]
    if len(leases) != 4:
        raise ValueError("Missing T1 leases")
    for index, lease in enumerate(leases):
        if lease["status"] != "complete" or lease["calls"] != (1 if index == 0 else 15) or lease["engine_sha256"] != manifest["engine_sha256"]:
            raise ValueError("T1 actual engine calls differ")
        if lease["free_before_deserialize"] - lease["free_after_release"] > 512*1024**2:
            raise ValueError("T1 lease did not release memory")
    native_hot, trt_scoped = report["native_hot_complete_image_seconds"], report["trt_scoped_complete_image_seconds"]
    if len(native_hot) != 3 or len(trt_scoped) != 3 or any(v <= 0 for v in native_hot + trt_scoped):
        raise ValueError("Missing timing observations")
    if torch.cuda.is_initialized():
        raise ValueError("Independent audit must be CPU-only")
    return {"status": "t1_actual_engine_and_full_image_api_bound_not_human_qualified",
            "tile_moments": metrics(reference["native_moments"], outputs["trt_moments"]),
            "full_image_latent": metrics(outputs["native_latent"], outputs["trt_latent"]),
            "native_hot_median_seconds": statistics.median(native_hot),
            "trt_scoped_median_seconds": statistics.median(trt_scoped),
            "engine_sha256": manifest["engine_sha256"], "outputs_sha256": report["outputs_sha256"],
            "resource_summary": guard, "cuda_initialized": False,
            "limits": "Actual single still only. Native hot vs TRT per-call hashing/load/release have different caching policies; no H3 end-to-end acceleration or downstream visual quality claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    write_new_json(args.run_dir / "independent-t1-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
