"""Run project CPU contracts in a fresh process against one complete Core tree.

Use an extracted Git archive or a separate checkout; never changes installed Core
or Python packages. Dependency versions and imported module paths are evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
MODULES = ("comfy.ldm.minimax.model", "comfy.model_patcher", "comfy.model_prefetch",
           "comfy_api.latest._io", "folder_paths")
FOCUSED = (
    "test_h3_core_compat.py", "test_vdn_h3_advanced.py", "test_vdn_sdpa_backend.py",
    "test_vdn_attention_compat.py", "test_vdn_two_pass.py", "test_vdn_core_probe.py",
    "test_multikeyframe_advanced.py", "test_preflight_and_registration.py",
    "test_fast_h3_vsa_advanced.py", "test_h3_world_advanced.py", "test_sla_precision_v2_advanced.py",
    "test_h3_attention_ownership.py", "test_conditioning.py",
    "test_enhance_a_video_advanced.py", "test_activation_chunk_advanced.py", "test_speed_advanced.py",
    "test_vdn_native_branch_probe.py", "test_vdn_probe_media.py", "test_external_compatibility_advanced.py",
    "test_sampling.py", "test_sampling_multirate_exp.py", "test_learned_latent_upscale_advanced.py",
    "test_face_refine_sampler_mask_advanced.py", "test_dlss_nr_advanced.py", "test_h3_av_delivery.py",
    "test_vdn_two_pass_workflow_export.py", "test_h3_compile_lifecycle.py", "test_vdn_probe_environment.py",
    "test_prompt_relay_core_compat.py", "test_prompt_relay_advanced.py", "test_prompt_relay_long_video_advanced.py",
    "test_h3_block_cache_compat.py", "test_block_cache_core_compat.py",
    "test_attention_hooks_advanced.py", "test_attention_hooks_core_contract.py",
    "test_vdn_workflow_shared_controls.py",
)


def project_source_hashes():
    paths = list(PROJECT.glob("*.py"))
    for directory in ("tests", "tools", "h3_t8"):
        paths.extend((PROJECT / directory).rglob("*.py"))
    paths.sort()
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    core = args.core_root.resolve()
    output = args.output.resolve()
    if not (core / "comfy/ldm/minimax/model.py").is_file():
        raise FileNotFoundError("the selected Core tree has no native H3 model")
    output.mkdir(parents=True, exist_ok=False)
    # Explicitly remove the runner's tools directory to avoid a module alias
    # accidentally shadowing Core. No fallback to the installed Comfy checkout.
    sys.path[:] = [str(core), str(PROJECT), *[p for p in sys.path if p not in (str(PROJECT / "tools"), str(core), str(PROJECT))]]
    # Worker subprocesses must inherit the same Core, not an empty PYTHONPATH
    # or the user's installed checkout. Parent-only sys.path is insufficient.
    os.environ["PYTHONPATH"] = os.pathsep.join((str(core), str(PROJECT)))
    os.chdir(core)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import pytest

    versions = {}
    for package in ("torch", "comfy-kitchen", "comfy-aimdo", "comfyui-frontend-package", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "core_root": str(core),
              "source_revision": args.source_revision, "python": sys.version,
              "dependencies": versions, "dependency_scope": "installed host dependencies, not historical package pins",
              "cpu_only": True, "full_suite": args.full}
    report["project_sources_at_start"] = project_source_hashes()
    tests = [str(PROJECT / "tests")] if args.full else [str(PROJECT / "tests" / name) for name in FOCUSED]
    code = int(pytest.main([*tests, "-q", f"--junitxml={output / 'pytest.xml'}"]))
    paths = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in MODULES}
    origins_match = all(Path(path).is_relative_to(core) for path in paths.values())
    stable_sources = report["project_sources_at_start"] == project_source_hashes()
    report.update(pytest_exit_code=code, imported_core_modules=paths, imported_core_origins_match=origins_match,
                  project_sources_unchanged=stable_sources,
                  status="pass" if code == 0 and origins_match and stable_sources else "failed")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "project_sources_at_start"}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
