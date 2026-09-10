"""Research-only replay of audited reference encodings, not a runtime VAE loader.

The fixed sampler receives identical source pixels and either actual saved native
or TRT reference latent. This isolates downstream quality from runtime residency
and never claims an encode-time or complete TRT pipeline benchmark.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_identity(root, *, allowed_root=RESEARCH):
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError("Encoder evidence must remain in the research directory")
    audit_path = root / "independent-encoder-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    if (audit.get("status") != "full_0p5mp_encoder_evidence_audited_not_downstream_quality_qualification"
            or audit.get("source_pixels_exact") is not True
            or audit.get("pixel_normalization_exact") is not True
            or audit.get("actual_trt_calls") != 90):
        raise ValueError("A completed independent full encoder audit is required")
    required = {"request.json", "result.json", "source-rgb8.safetensors", "native-latents.safetensors", "trt-latents.safetensors"}
    if not required.issubset(audit.get("files", {})):
        raise ValueError("Incomplete encoder evidence identities")
    for name, sha in audit["files"].items():
        path = (root / name).resolve(strict=True)
        if path.parent != root or digest(path) != sha:
            raise ValueError("Encoder evidence file changed or escaped: " + name)
    return {"root": str(root), "audit_sha256": digest(audit_path), "files": audit["files"]}


class SavedReferenceEncoder:
    """Single-use, exact-input delegate. Decode always remains the original VAE."""
    def __init__(self, vae, image, latent, identity, backend, reference_kind="image"):
        import torch
        if reference_kind not in ("image", "video"):
            raise ValueError("Unknown reference kind")
        frames, tokens = (1, 1) if reference_kind == "image" else (73, 22)
        if backend not in ("native", "trt"):
            raise ValueError("Unknown saved encoder backend")
        if image.shape != (frames, 512, 1024, 3) or image.dtype != torch.float32 or image.device.type != "cpu":
            raise ValueError("Expected a CPU float32 complete0.5MP IMAGE")
        if latent.shape != (1, 24, tokens, 32, 64) or latent.dtype != torch.float32 or latent.device.type != "cpu":
            raise ValueError("Expected the CPU float32 normalized reference latent")
        if not bool(torch.isfinite(image).all() and torch.isfinite(latent).all()) or not bool(((image >= 0) & (image <= 1)).all()):
            raise ValueError("Nonfinite or out-of-range reference evidence")
        self.vae, self.expected_image, self.latent = vae, image.clone(), latent.clone()
        self.identity, self.backend, self.calls = deepcopy(identity), backend, 0
        self.reference_kind = reference_kind

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, image):
        import torch
        actual = image.detach().cpu()
        if self.calls != 0 or actual.dtype != self.expected_image.dtype or not torch.equal(actual, self.expected_image):
            raise ValueError("Saved encoding replay requires exactly one unchanged reference image")
        self.calls += 1
        return self.latent.clone()

    def report(self):
        if self.calls != 1:
            raise ValueError("Reference encoding has not been consumed exactly once")
        report = {"status": "audited_reference_encoding_consumed", "backend": self.backend,
                "actual_encode_calls": self.calls, "identity": self.identity,
                "scope": "Saved real encoder result replay; native final decode; not a TRT runtime or encode benchmark"}
        if self.reference_kind == "video":
            report.update(reference_kind="video", reference_frames=73, reference_latent_tokens=22)
        return report


def load_reference(vae, root, backend, expected_audit_sha256, *, allowed_root=RESEARCH, reference_kind="image"):
    from safetensors import safe_open
    identity = evidence_identity(root, allowed_root=allowed_root)
    if identity["audit_sha256"] != expected_audit_sha256:
        raise ValueError("Encoder audit identity changed since preflight")
    if backend not in ("native", "trt"):
        raise ValueError("Unknown saved encoder backend")
    if reference_kind not in ("image", "video"):
        raise ValueError("Unknown reference kind")
    root = Path(identity["root"])
    with safe_open(root / "source-rgb8.safetensors", framework="pt", device="cpu") as saved:
        source = saved.get_tensor(reference_kind)
    import torch
    if source.shape != (1, 3, 1 if reference_kind == "image" else 73, 512, 1024) or source.dtype != torch.uint8:
        raise ValueError("Source must be the audited complete0.5MP RGB8 reference")
    image = source[0].movedim(0, -1).float() / 255
    with safe_open(root / (backend + "-latents.safetensors"), framework="pt", device="cpu") as saved:
        latent = saved.get_tensor(reference_kind)
    delegate = SavedReferenceEncoder(vae, image, latent, identity, backend, reference_kind)
    return delegate, image


def condition_recipe(graph, root, backend, audit_sha256, reference_kind="image"):
    if reference_kind not in ("image", "video"):
        raise ValueError("Unknown reference kind")
    output = deepcopy(graph)
    if (backend not in ("native", "trt") or output["6"]["inputs"].get("task_type") != "I2VA"
            or output["10"]["class_type"] != "T8ProgressiveNativeBaseline"
            or any(key in output for key in ("107", "108", "109"))):
        raise ValueError("Saved encoder comparison requires the fixed I2VA native8 graph")
    output.pop("20")
    output["107"] = {"class_type": "T8TRTSavedReferenceEncoder", "inputs": {
        "video_vae": ["1", 0], "evidence_root": str(root), "backend": backend, "audit_sha256": audit_sha256}}
    output["6"]["inputs"].update(video_vae=["107", 0], first_frame=["107", 1])
    output["108"] = {"class_type": "T8TRTReferenceConsumedAudit", "inputs": {
        "video_vae": ["107", 0], "positive": ["6", 0]}}
    output["100"]["inputs"]["positive"] = ["108", 0]
    output["109"] = {"class_type": "PreviewAny", "inputs": {"source": ["108", 1]}}
    output["18"]["inputs"]["filename_prefix"] = "MiniMaxH3/TRTEncoderCompare/" + backend + "_UNREVIEWED"
    if reference_kind == "video":
        output["107"]["inputs"]["reference_kind"] = "video"
        output["6"]["inputs"].pop("first_frame")
        output["6"]["inputs"].update(task_type="auto", **{"ref_videos.ref_video_0": ["107", 1]})
        output["90"]["inputs"]["value"] = (
            "Use <Video 1> as the visual reference. One continuous cinematic close-up of the same woman "
            "in the same room, matching her face, clothing and warm lighting. She gently turns her head "
            "and blinks. Stable camera and coherent motion. Quiet clean room ambience, no speech, music, "
            "cuts, subtitles, hiss, static or distortion."
        )
        output["18"]["inputs"]["filename_prefix"] = "MiniMaxH3/TRTVideoEncoderCompare/" + backend + "_UNREVIEWED"
    return output
