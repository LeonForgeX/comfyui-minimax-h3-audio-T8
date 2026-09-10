"""Independent CPU whole-frame audit against a provenance-bound native decode."""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, validate_request, validate_quantized_manifest, write_new_json  # noqa: E402
from tools.trt_vae_saved_reference import bind_reference  # noqa: E402
from tools.audit_trt_vae_video_probe import frame_metrics  # noqa: E402


def audit(root):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    request = json.loads((root / "request.json").read_text(encoding="utf8"))
    result = json.loads((root / "result.json").read_text(encoding="utf8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf8"))
    resources = json.loads((root / "resource-summary.json").read_text(encoding="utf8"))
    job = terminal["isolated"]
    if (terminal["status"] != "worker_complete_result_requires_audit" or job["status"] != "complete"
            or job["exit_code"] != 0 or job["active_after_cleanup"] != 0 or not job["job_assigned_before_task"]
            or resources["status"] != "observations_within_policy"):
        raise ValueError("Job or active resource observation failed")
    frozen = 0
    for path, sha in request["sources"].items():
        source = Path(path)
        if source.suffix == ".py" and source.is_relative_to(PROJECT):
            source = root / "source-snapshot" / (sha+"-"+source.name)
            frozen += 1
        if digest_file(source) != sha:
            raise ValueError("Snapshot or asset identity changed: " + source.name)
    if bind_reference(request["reference_rgb"], request["latent"], request["native_vae"]) != request["reference_sources"]:
        raise ValueError("Reused native reference provenance differs")
    manifest = json.loads((Path(request["bundle"]) / "manifest.json").read_text(encoding="utf8"))
    if validate_request(manifest["source_request"]) != manifest["request_sha256"]:
        raise ValueError("Engine request identity differs")
    validate_quantized_manifest(manifest, manifest["source_request"])
    if digest_file(Path(request["bundle"])/"model.engine") != manifest["engine_sha256"]:
        raise ValueError("Actual engine differs from compiled engine")
    lease = result["lease"]
    if (lease["status"] != "complete" or lease["calls"] != 60
            or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2
            or result["standard_output_shape"] != [1,73,512,1024,3]
            or result["native_model_class"] != "comfy.sd.VAE"
            or digest_file(root / "trt-rgb.safetensors") != result["rgb_sha256"]):
        raise ValueError("Actual Core interface, complete execution or release evidence differs")
    rows = []
    with safe_open(request["reference_rgb"], framework="pt", device="cpu") as native, safe_open(
        root / "trt-rgb.safetensors", framework="pt", device="cpu"
    ) as candidate:
        a, b = native.get_slice("rgb"), candidate.get_slice("rgb")
        if a.get_shape() != [1,3,73,512,1024] or a.get_shape() != b.get_shape():
            raise ValueError("Full saved RGB dimensions differ")
        for index in range(73):
            metrics = frame_metrics(a[:,:,index], b[:,:,index])
            mse = metrics["sum_squared_error"] / metrics["count"]
            rows.append({"frame": index, **metrics, "psnr_db": -10*math.log10(mse) if mse else None})
    count = sum(row["count"] for row in rows)
    mse = sum(row["sum_squared_error"] for row in rows)/count
    if torch.cuda.is_initialized():
        raise RuntimeError("Independent CPU audit initialized CUDA")
    return {"status": "complete_short_scoped_decode_against_bound_native_not_human_qualified",
            "engine_sha256": manifest["engine_sha256"], "precision": manifest["source_request"]["precision"],
            "source_snapshots": frozen, "actual_tile_calls": 60, "frames": 73,
            "rgb_psnr_db": -10*math.log10(mse) if mse else None,
            "rgb_max_error": max(row["max_error"] for row in rows),
            "rgb_fraction_over_2_codes": sum(row["over_2_codes"] for row in rows)/count,
            "per_frame": rows, "cuda_initialized": False,
            "files": {name:digest_file(root/name) for name in ("request.json", "result.json", "terminal.json", "resource-summary.json", "trt-rgb.safetensors")},
            "reference_sources": request["reference_sources"], "resource_summary": resources,
            "limits": "Native whole RGB reused with same latent/VAE provenance; complete actual candidate decode. No new sampling/native timing/audio/human or long-video validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir / "independent-saved-reference-audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_frame", "files", "reference_sources")}, indent=2))


if __name__ == "__main__":
    main()
