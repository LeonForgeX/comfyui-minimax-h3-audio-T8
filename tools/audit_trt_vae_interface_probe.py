"""CPU independent binding/whole-RGB and lifecycle receipt audit; no GPU work."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def audit(root):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    root = Path(root).resolve(strict=True)
    request = json.loads((root / "request.json").read_text(encoding="utf8"))
    report = json.loads((root / "result.json").read_text(encoding="utf8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf8"))
    resources = json.loads((root / "resource-summary.json").read_text(encoding="utf8"))
    job = terminal["isolated"]
    if job["status"] != "complete" or job["exit_code"] != 0 or job["active_after_cleanup"] != 0 or not job["job_assigned_before_task"]:
        raise ValueError("Worker termination/Job cleanup is not proven")
    if resources["status"] != "observations_within_policy":
        raise ValueError("Resource guard failed")
    frozen = 0
    for path, sha in request["sources"].items():
        source = Path(path)
        if source.suffix == ".py" and source.is_relative_to(PROJECT):
            source = root / "source-snapshot" / (sha + "-" + source.name)
            frozen += 1
        if digest_file(source) != sha:
            raise ValueError("Probe source/snapshot identity mismatch")
    previous_root = Path(request["reference_rgb"]).parent
    previous = json.loads((previous_root / "request.json").read_text(encoding="utf8"))
    previous_result = json.loads((previous_root / "result.json").read_text(encoding="utf8"))
    previous_audit = json.loads((previous_root / "independent-video-audit.json").read_text(encoding="utf8"))
    if not previous_audit:
        raise ValueError("Missing independent previous decode audit")
    for key in ("latent", "native_vae", "bundle"):
        if Path(previous[key]).resolve() != Path(request[key]).resolve():
            raise ValueError("Reference does not belong to the same latent/VAE/engine")
    for key in ("latent", "native_vae"):
        path = request[key]
        if request["sources"][path] != previous["sources"][path]:
            raise ValueError("Same-named source has different content")
    saved_rgb = Path(request["reference_rgb"])
    if digest_file(saved_rgb) != previous_result["files"][saved_rgb.name]:
        raise ValueError("Earlier TRT reference identity differs")
    with safe_open(request["latent"], framework="pt", device="cpu") as file:
        z = file.get_tensor("latent_tensor").half()
    previous_input = load_file(str(previous_root / "input-latent.safetensors"))["normalized_latent"]
    if not torch.equal(z, previous_input):
        raise ValueError("Original source latent does not match previous exact input")
    candidate_path = root / "interface-rgb.safetensors"
    if digest_file(candidate_path) != report["files"][candidate_path.name]:
        raise ValueError("Candidate RGB identity differs")
    candidate = load_file(str(candidate_path))["rgb"]
    reference = load_file(str(saved_rgb))["rgb"]
    if candidate.shape != (1, 3, 73, 512, 1024) or not torch.equal(candidate, reference) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("Whole73-frame interface RGB is not bit-identical to saved TRT reference")
    if report["standard_output_shape"] != [1, 73, 512, 1024, 3] or report["native_model_class"] != "comfy.sd.VAE":
        raise ValueError("Actual Core wrapper/layout evidence differs")
    for key, status in (("cancelled_lease", "failed"), ("complete_lease", "complete")):
        lease = report[key]
        if lease["status"] != status or lease["free_before_deserialize"] - lease["free_after_release"] > 512 * 1024**2:
            raise ValueError("Lifecycle release evidence differs")
        if lease["context_device_bytes"] != 125194240:
            raise ValueError("Pinned decoder context allocation differs")
    if report["complete_lease"]["calls"] != 60 or report["cancelled_lease"]["error"] != "intentional_lifecycle_probe_cancel":
        raise ValueError("Complete/cancelled execution receipt differs")
    return {"status": "actual_core_scoped_decode_and_cleanup_bound",
            "source_snapshots": frozen, "complete_frames": 73, "complete_calls": 60,
            "rgb_bit_identical_to_previous_trt": True, "previous_run": str(previous_root),
            "rgb_sha256": digest_file(candidate_path), "resource_summary": resources,
            "scope": "CPU independent completeRGB/source/Job/resource/lifecycle audit. Reuse prior media because pixels are identical; no new sampler/audio/human/encoder/public-node qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir / "independent-interface-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
