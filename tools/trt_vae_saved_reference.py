"""Identity-bound reuse of an already audited native short-video decode."""
import json
from pathlib import Path

from trt_vae_build import digest_file


def bind_reference(reference, latent, native_vae):
    import torch
    from safetensors import safe_open
    reference = Path(reference).resolve(strict=True)
    if reference.name != "native-rgb.safetensors":
        raise ValueError("Reuse the actual native RGB, not a TRT output relabelled as native")
    root = reference.parent
    audit = json.loads((root / "independent-video-audit.json").read_text(encoding="utf8"))
    if (audit.get("status") != "complete_rgb_evidence_verified_not_human_or_speed_qualification"
            or audit.get("finite_normalized_all_frames") is not True or audit.get("cuda_initialized") is not False):
        raise ValueError("Previous native decode lacks independent whole-frame evidence")
    files = ("request.json", "result.json", "terminal.json", "resource-summary.json",
             "input-latent.safetensors", "native-rgb.safetensors")
    for name in files:
        if digest_file(root / name) != audit["files"][name]:
            raise ValueError("Previous reference receipt or payload changed: " + name)
    previous = json.loads((root / "request.json").read_text(encoding="utf8"))
    for key, path in (("latent", latent), ("native_vae", native_vae)):
        if Path(previous[key]).resolve() != Path(path).resolve() or digest_file(path) != previous["sources"][previous[key]]:
            raise ValueError("Reused reference belongs to a different latent or native VAE")
    job = json.loads((root / "terminal.json").read_text(encoding="utf8"))["isolated"]
    if job["status"] != "complete" or job["exit_code"] != 0 or job["active_after_cleanup"] != 0 or not job["job_assigned_before_task"]:
        raise ValueError("Previous native decode did not finish in its owned Job")
    with safe_open(latent, framework="pt", device="cpu") as source, safe_open(
        root / "input-latent.safetensors", framework="pt", device="cpu"
    ) as saved, safe_open(reference, framework="pt", device="cpu") as rgb:
        if "latent_format_version_0" not in source.keys():
            raise ValueError("Unknown original latent serialization scale")
        z = source.get_tensor("latent_tensor")
        if tuple(z.shape) != (1,24,22,32,64) or not torch.equal(z.half(), saved.get_tensor("normalized_latent")):
            raise ValueError("Only the same exact complete0.5MP/73-frame short latent may be reused")
        if rgb.get_slice("rgb").get_shape() != [1,3,73,512,1024]:
            raise ValueError("Native reference dimensions differ")
    paths = [root / name for name in (*files, "independent-video-audit.json")]
    return {str(path): digest_file(path) for path in paths}
