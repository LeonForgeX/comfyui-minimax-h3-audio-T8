"""CPU audit of actual Core/scoped/bounded decode, including model buffer dtype."""
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file, write_new_json, validate_request  # noqa: E402
from tools.audit_trt_vae_video_probe import frame_metrics  # noqa: E402


def audit(root):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    request = json.loads((root/"request.json").read_text(encoding="utf8"))
    result = json.loads((root/"result.json").read_text(encoding="utf8"))
    job = json.loads((root/"terminal.json").read_text(encoding="utf8"))["isolated"]
    guard = json.loads((root/"resource-summary.json").read_text(encoding="utf8"))
    if (job["status"] != "complete" or job["exit_code"] != 0 or job["active_after_cleanup"] != 0
            or not job["job_assigned_before_task"] or guard["status"] != "observations_within_policy"):
        raise ValueError("Special probe Job/resource evidence incomplete")
    for path,sha in request["sources"].items():
        p = Path(path)
        if p.suffix == ".py" and p.is_relative_to(PROJECT):
            p = root/"source-snapshot"/(sha+"-"+p.name)
        if digest_file(p) != sha:
            raise ValueError("Special interface source identity changed")
    for name,sha in result["files"].items():
        if Path(name).name != name or digest_file(root/name) != sha:
            raise ValueError("Special output identity changed")
    manifest = json.loads((Path(request["bundle"])/"manifest.json").read_text(encoding="utf8"))
    if (validate_request(manifest["source_request"]) != manifest["request_sha256"]
            or digest_file(Path(request["bundle"])/"model.engine") != manifest["engine_sha256"]):
        raise ValueError("Actual compiled decoder engine identity differs")
    with safe_open(request["latent"],framework="pt",device="cpu") as original:
        if "latent_format_version_0" not in original.keys():
            raise ValueError("Unknown source latent serialization")
        source = original.get_tensor("latent_tensor")[:,:,:12]
    if not torch.equal(source,load_file(str(root/"input-latent.safetensors"))["normalized_latent"]):
        raise ValueError("Actual39-frame prefix input changed")
    if (result["native_calls"] != 30 or result["reads"] != [[0,7],[5,12]]
            or [v["calls"] for v in result["leases"]] != [30,15,15]
            or result["native_model_class"] != "comfy.sd.VAE"):
        raise ValueError("Actual Core/bounded trajectory or tile accounting differs")
    for lease in result["leases"]:
        if (lease["status"] != "complete" or lease["engine_sha256"] != manifest["engine_sha256"]
                or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2):
            raise ValueError("Special decoder lease identity/release failed")
    if [(c["start"],c["frames"]) for c in result["chunks"]] != [(0,17),(17,17),(34,5)]:
        raise ValueError("Bounded shot did not deliver39contiguous frames")
    if any(not c["report"]["raw_blend_before_clamp"] or c["report"]["padding_delivered"] for c in result["chunks"]):
        raise ValueError("Bounded output included padding or post-clamp blending")
    buffers = load_file(str(root/"actual-core-buffers.safetensors"))
    for name,value in buffers.items():
        if str(value.dtype) != result["buffer_dtypes"][name] or not bool(torch.isfinite(value).all()):
            raise ValueError("Actual native model normalization receipt differs")
    rows = []
    with safe_open(root/"outputs.safetensors",framework="pt",device="cpu") as f:
        tensors = [f.get_slice(name) for name in ("native","trt_standard","trt_bounded")]
        if any(v.get_shape() != [1,3,39,512,1024] for v in tensors):
            raise ValueError("Saved complete prefix RGB has wrong shape")
        for index in range(39):
            native,standard,bounded = [v[:,:,index] for v in tensors]
            if not torch.equal(standard,bounded):
                raise ValueError("TRT standard and real bounded outpaint decoder differ")
            rows.append({"frame":index,**frame_metrics(native,standard)})
    count = sum(r["count"] for r in rows)
    mse = sum(r["sum_squared_error"] for r in rows)/count
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit initialized CUDA")
    return {"status":"actual_core_and_trt_standard_bounded_short_decode_verified_not_human",
            "frames":39,"native_calls":30,"trt_lease_calls":[30,15,15],
            "all_standard_bounded_rgb_bit_equal":True,"native_buffer_dtypes":result["buffer_dtypes"],
            "rgb_psnr_db":-10*math.log10(mse) if mse else None,"rgb_max_error":max(r["max_error"] for r in rows),
            "rgb_fraction_over_2_codes":sum(r["over_2_codes"] for r in rows)/count,
            "per_frame":rows,"resource_summary":guard,"cuda_initialized":False,
            "files":{n:digest_file(root/n) for n in ("request.json","result.json","terminal.json","resource-summary.json","input-latent.safetensors","actual-core-buffers.safetensors","outputs.safetensors","plan.json")},
            "limits":"Actual short39-frame Core/scoped/bounded decoder interfaces and full-pixel equality between TRT routes. No new outpaint generation, FaceRefine, compositing/pixel-anchor, human or long-video qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir/"independent-special-audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("per_frame","files")},indent=2))


if __name__ == "__main__":
    main()
