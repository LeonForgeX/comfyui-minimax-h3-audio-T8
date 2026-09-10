"""Freeze existing pilot recipes, installed model bytes and source identity.

This command never starts ComfyUI, initializes Torch CUDA, or queues inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402
from vdn_probe_environment import verify_core_source  # noqa: E402


MODEL_INPUTS = {
    "VAELoader": ("vae", "vae_name"), "CLIPLoader": ("text_encoders", "clip_name"),
    "UNETLoader": ("diffusion_models", "unet_name"),
    "MiniMaxH3LoRACompatibilityLoaderT8Advanced": ("loras", "lora_name"),
    "MiniMaxH3ProgressiveSamplerEXPT8": ("latent_upscale_models", "upscaler_model"),
}


def comparison_common_graph(prompt):
    common = {key: deepcopy(value) for key, value in prompt.items() if key not in {"8", "9", "10", "13", "21"}}
    common["18"]["inputs"].pop("filename_prefix")
    return common


def recipe_assets(prompt, runtime):
    runtime = Path(runtime).resolve()
    assets = set()
    for node in prompt.values():
        kind, inputs = node["class_type"], node["inputs"]
        if kind in MODEL_INPUTS:
            folder, field = MODEL_INPUTS[kind]
            root = runtime / "models" / folder
            path = (root / inputs[field]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Model path leaves its declared installed folder")
            assets.add(path)
        elif kind == "LoadImage":
            root = runtime / "input"
            path = (root / inputs["image"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Reference image leaves input directory")
            assets.add(path)
    return assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    runtime = project.parents[1]
    research = project / "artifacts/acceleration-research-20260909"
    output = args.output.resolve()
    if not output.is_relative_to(research) or output.exists():
        raise ValueError("Use a new output JSON within this research directory")
    with SerialProbeLease(research / "serial-gpu.lock"):
        provenance = verify_core_source(runtime)
        folder = research / "pilot-api-drafts"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        prompts, recipes, assets = {}, {}, set()
        for case in manifest["cases"]:
            path = folder / case["file"]
            identity = file_identity(path)
            if identity["sha256"] != case["sha256"]:
                raise ValueError("Pilot recipe changed after validation")
            prompt = json.loads(path.read_text(encoding="utf-8"))
            prompts[(case["task"], case["route"])] = prompt
            recipes[case["file"]] = identity
            assets.update(recipe_assets(prompt, runtime))
        for task in ("T2VA", "I2VA"):
            native = comparison_common_graph(prompts[task, "native8"])
            progressive = comparison_common_graph(prompts[task, "progressive6plus2"])
            if native != progressive:
                raise ValueError(f"{task} baseline and candidate differ beyond the sampler route")
        identities = []
        for path in sorted(assets):
            print(f"Read-only hashing {path.name} ({path.stat().st_size / 1024**3:.2f} GiB)", flush=True)
            identities.append(file_identity(path))
        selected_sources = [*project.glob("*.py"), Path(__file__), project / "tools/progressive_probe_control.py",
                            project / "tools/vdn_probe_environment.py"]
        sources = {str(path.relative_to(project)): file_identity(path)["sha256"] for path in sorted(set(selected_sources))}
        with NvmlResourceReader() as reader:
            resources = reader.sample()
        guard = ResourceGuard()
        guard.observe(resources, startup=True)
        payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "identities_frozen_no_inference",
                   "core": provenance, "recipes": recipes, "installed_assets": identities, "project_sources_sha256": sources,
                   "resource_snapshot": resources, "startup_guard": guard.report(),
                   "sampler_comparison_only": True, "gpu_started": False, "queue_submitted": False,
                   "limitations": ["File hashes bind installed assets, not proof of actual loaded tensors.",
                       "Source manifest covers root Python plus named control tools; imports must be checked in the actual server.",
                       "This read warms filesystem cache; cold means process-cold, not disk-cold.",
                       "Runtime stage tracing, continuous guard and actual server execution still required."]}
        # Recheck the short recipes and Core provenance after the potentially long scans.
        if verify_core_source(runtime) != provenance:
            raise RuntimeError("Core changed during preparation")
        for identity in recipes.values():
            if hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest() != identity["sha256"]:
                raise RuntimeError("Recipe changed during preparation")
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"status": payload["status"], "assets": len(identities), "sources": len(sources),
                          "startup_guard": guard.report(), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
