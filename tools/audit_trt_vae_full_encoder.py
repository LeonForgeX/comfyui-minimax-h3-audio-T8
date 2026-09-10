"""Independent CPU audit: actual complete video encode through Core/scoped APIs."""
import argparse
import ast
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, validate_request, write_new_json, ENCODER_MODEL_SHA, ENCODER_T1_MODEL_SHA  # noqa: E402
from tools.audit_trt_vae_t1_probe import metrics  # noqa: E402


def normalization_residual(previous, candidate, mean, std):
    """Two measured routes used FP32 shell vs FP16 Core normalization buffers.

    Undo/reapply that known affine operation only. Bound its FP32 arithmetic
    roundoff, not a perceptual tolerance; any remaining tile/seam error fails.
    """
    import torch
    mean, std = [torch.tensor(v,dtype=torch.float32).view(1,24,1,1,1) for v in (mean,std)]
    expected = (previous*std+mean-mean.half().float())/std.half().float()
    residual = (candidate-expected).abs()
    bound = 4*torch.finfo(torch.float32).eps*(1+expected.abs())
    if not bool(torch.isfinite(residual).all()) or not bool((residual <= bound).all()):
        raise ValueError("Encoder difference is not explained by known normalization-buffer precision")
    return {"old_trt_bit_equal":torch.equal(previous,candidate), "normalization_affine_residual_max":float(residual.max()),
            "roundoff_bound":"4*float32_epsilon*(1+abs(expected)) per element",
            "all_elements_within_affine_roundoff":True,
            "explanation":"Earlier isolated shell retained FP32 latent normalization constants; actual Core model uses FP16 buffers. This check changes no output or quality threshold."}


def audit(root):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    request = json.loads((root/"request.json").read_text(encoding="utf8"))
    result = json.loads((root/"result.json").read_text(encoding="utf8"))
    job = json.loads((root/"terminal.json").read_text(encoding="utf8"))["isolated"]
    guard = json.loads((root/"resource-summary.json").read_text(encoding="utf8"))
    if (job["status"] != "complete" or job["exit_code"] != 0 or job["active_after_cleanup"] != 0
            or not job["job_assigned_before_task"] or guard["status"] != "observations_within_policy"):
        raise ValueError("Actual worker Job/resources failed")
    for path, sha in request["sources"].items():
        p = Path(path)
        if p.suffix == ".py":
            p = root/"source-snapshot"/(sha+"-"+p.name)
        if digest_file(p) != sha:
            raise ValueError("Frozen source/asset identity changed")
    if digest_file(request["native_vae"]) != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Native VAE identity differs")
    for name, key, sha in (("video","bundle",ENCODER_MODEL_SHA), ("image","t1_bundle",ENCODER_T1_MODEL_SHA)):
        path = Path(request[key])
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf8"))
        if (validate_request(manifest["source_request"]) != manifest["request_sha256"]
                or manifest["source_request"]["model_sha256"] != sha
                or digest_file(path/"model.engine") != manifest["engine_sha256"]
                or Path(result["selected_profiles"][name]["bundle"]).resolve() != path.resolve()
                or result["selected_profiles"][name]["engine_sha256"] != manifest["engine_sha256"]):
            raise ValueError("Encoder profile routing/identity differs")
    previous_root = Path(request["source_rgb8"]).parent
    previous = json.loads((previous_root/"independent-encoder-audit.json").read_text(encoding="utf8"))
    if previous["status"] != "full_0p5mp_encoder_evidence_audited_not_downstream_quality_qualification" or not previous["source_pixels_exact"]:
        raise ValueError("Saved RGB lacks prior source provenance")
    for name in ("source-rgb8.safetensors", "native-latents.safetensors", "trt-latents.safetensors"):
        if digest_file(previous_root/name) != previous["files"][name]:
            raise ValueError("Previous full encoder evidence changed")
    output = root/"outputs.safetensors"
    if digest_file(output) != result["outputs_sha256"]:
        raise ValueError("Actual output identity changed")
    values = load_file(str(output),device="cpu")
    for v in values.values():
        if v.dtype != torch.float32 or tuple(v.shape) != (1,24,22,32,64) or not bool(torch.isfinite(v).all()):
            raise ValueError("Full73-frame latent dimensions or values invalid")
    lease = result["lease"]
    if (lease["status"] != "complete" or lease["calls"] != 75 or result["native_calls"] != 75
            or result["native_model_class"] != "comfy.sd.VAE" or result["interface_report"]["operation"] != "trt_encode"
            or lease["engine_sha256"] != result["selected_profiles"]["video"]["engine_sha256"]
            or lease["free_before_deserialize"]-lease["free_after_release"] > 512*1024**2):
        raise ValueError("Actual API complete75tile execution or release failed")
    old_trt = load_file(str(previous_root/"trt-latents.safetensors"),device="cpu")["video"]
    old_native = load_file(str(previous_root/"native-latents.safetensors"),device="cpu")["video"]
    core_source = str((Path(request["core_root"])/"comfy/ldm/minimax/vae.py").resolve())
    frozen_core = root/"source-snapshot"/(request["sources"][core_source]+"-vae.py")
    constants = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(frozen_core.read_text()).body
                 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                 and node.targets[0].id in ("LATENTS_MEAN","LATENTS_STD")}
    affine = normalization_residual(old_trt,values["trt_latent"],constants["LATENTS_MEAN"],constants["LATENTS_STD"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU audit initialized CUDA")
    return {"status": "actual_complete_video_encoder_interface_and_profile_selection_bound_not_human_qualified",
            "actual_calls_per_route":75,"actual_frames":73,"trt_equal_previous_full_encode":torch.equal(old_trt,values["trt_latent"]),
            "normalization_precision_comparison":affine,
            "latent_metrics":metrics(values["native_latent"],values["trt_latent"]),
            "current_native_vs_previous":metrics(old_native,values["native_latent"]),
            "resource_summary":guard,"cuda_initialized":False,
            "files":{n:digest_file(root/n) for n in ("request.json","result.json","terminal.json","resource-summary.json","outputs.safetensors")},
            "limits":"Full video actual Core/scoped APIs, true source RGB and75calls each. T1 only profile selection checked here; prior actual T1 test retained. No new sampler or human/long/performance acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir/"independent-full-encoder-audit.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
