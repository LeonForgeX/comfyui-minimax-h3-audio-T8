"""Research-only, lossless AV latent capture before the unchanged native decode."""

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"


def tensor_identity(value):
    import torch

    cpu = value.detach().cpu().contiguous()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}


def capture_parts(video, audio, output_root, *, allowed_root=RESEARCH):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = Path(output_root).resolve(strict=True)
    if not root.is_relative_to(Path(allowed_root).resolve()) or root.name != "output":
        raise ValueError("Latent capture is restricted to an isolated research output directory")
    if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
        raise ValueError("Expected complete H3 video/audio latent parts")
    for value in (video, audio):
        if not value.is_floating_point() or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError("Only finite, nonempty floating-point latents can be captured")
    target = root / "trt-latent-evidence"
    target.mkdir(exist_ok=False)
    sources = {"video": tensor_identity(video), "audio": tensor_identity(audio)}
    save_file({"latent_tensor": video.detach().cpu().contiguous(), "latent_format_version_0": torch.tensor([])},
              str(target / "video.latent"), metadata={"scope": "complete sampled video latent; normalized; serialization scale1"})
    save_file({"audio_latent": audio.detach().cpu().contiguous()}, str(target / "audio.safetensors"))
    files = {}
    for kind, name, key in (("video", "video.latent", "latent_tensor"), ("audio", "audio.safetensors", "audio_latent")):
        with safe_open(target / name, framework="pt", device="cpu") as saved:
            if tensor_identity(saved.get_tensor(key)) != sources[kind]:
                raise ValueError("Saved latent differs from actual sampler output")
        files[kind] = {"path": str(target / name), "sha256": hashlib.sha256((target / name).read_bytes()).hexdigest(),
                       "tensor": sources[kind]}
    report = {"status": "actual_sampler_output_captured_bit_exact", "files": files,
              "scope": "Before unchanged native decode; no new noise, scaling, resizing or regeneration",
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (target / "capture.json").open("x", encoding="utf8") as file:
        json.dump(report, file, indent=2)
    return report


def capture_recipe(graph):
    from copy import deepcopy

    output = deepcopy(graph)
    if "105" in output or "106" in output or output["11"]["class_type"] != "T8ProgressiveTimedDecode":
        raise ValueError("Unexpected isolated decode graph or capture node collision")
    output["105"] = {"class_type": "T8TRTLatentCapture", "inputs": {"av_latent": output["11"]["inputs"]["av_latent"]}}
    output["11"]["inputs"]["av_latent"] = ["105", 0]
    output["106"] = {"class_type": "PreviewAny", "inputs": {"source": ["105", 1]}}
    return output
