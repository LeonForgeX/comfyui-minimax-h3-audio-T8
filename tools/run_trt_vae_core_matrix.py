"""Serial CPU TRT VAE contracts against real, already-frozen old/current Core.

No packages, Core files, GPU initialization or frontend changes. Records module
origins and relevant source hashes; historical dependencies are not emulated.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402

TESTS = ["test_trt_vae_contract.py", "test_trt_vae_encode.py", "test_trt_vae_interface.py",
         "test_trt_vae_backend.py", "test_trt_vae_engine.py", "test_trt_vae_build.py", "test_trt_vae_flex.py",
         "test_trt_vae_w4.py"]
MODULES = ["comfy.sd", "comfy.model_management", "comfy.ldm.minimax.vae", "comfy.ops", "comfy_api.latest._io"]


def worker(core, output, revision):
    sys.path[:] = [str(core), str(PROJECT), *[p for p in sys.path if p not in (str(PROJECT / "tools"), str(core), str(PROJECT))]]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONPATH"] = os.pathsep.join((str(core), str(PROJECT)))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import pytest
    torch.set_num_threads(2)
    sources = [Path(__file__), *PROJECT.glob("trt_vae_*.py"), *[PROJECT / "tests" / name for name in TESTS]]
    sources += [core / (name.replace(".", "/") + ".py") for name in MODULES]
    before = {str(path): digest_file(path) for path in sources}
    code = pytest.main([*[str(PROJECT / "tests" / name) for name in TESTS], "-q", "--junitxml=" + str(output / "pytest.xml")])
    origins = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in MODULES}
    unchanged = all(digest_file(path) == sha for path, sha in before.items())
    valid = code == 0 and unchanged and not torch.cuda.is_initialized() and all(Path(p).is_relative_to(core) for p in origins.values())
    report = {"status": "pass" if valid else "failed", "core_root": str(core), "source_revision": revision,
              "core_origins": origins, "sources": before, "sources_unchanged": unchanged,
              "pytest_code": int(code), "cuda_initialized": torch.cuda.is_initialized(),
              "torch": str(torch.__version__), "tests": TESTS,
              "scope": "Actual selected Core imports, CPU tensor/orchestration/lifecycle contracts. Fake tile/engine kernels, host current dependencies; not historical GPU or package pins."}
    write_new_json(output / "report.json", report)
    return 0 if valid else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-core", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(RESEARCH) or output.exists():
        raise ValueError("New research output directory required")
    output.mkdir(parents=True, exist_ok=False)
    if args.worker_core:
        return worker(args.worker_core.resolve(strict=True), output, args.revision)
    rows = []
    for label in ("current", "cf10c5c", "e7051b0", "86aedfd", "563b98e"):
        if label == "current":
            core = PROJECT.parents[1]
            revision = subprocess.check_output(["git", "-C", str(core), "rev-parse", "HEAD"], text=True).strip()
        else:
            previous = json.loads((PROJECT / "artifacts/core-contract-matrix-20260908" / (label + "-expanded-v15") / "report.json").read_text(encoding="utf8"))
            core, revision = Path(previous["core_root"]), previous["source_revision"]
        print("START TRT CPU Core " + label, flush=True)
        with (output / (label + ".log")).open("x", encoding="utf8") as log:
            result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__)), "--output", str(output / label),
                                     "--worker-core", str(core), "--revision", revision], cwd=PROJECT,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), stdout=log, stderr=subprocess.STDOUT, timeout=300)
        rows.append({"core": label, "exit_code": result.returncode, "report": str(output / label / "report.json")})
        print("END TRT CPU Core " + label + " exit=" + str(result.returncode), flush=True)
    write_new_json(output / "summary.json", {"status": "pass" if all(r["exit_code"] == 0 for r in rows) else "failed", "runs": rows})
    return 0 if all(r["exit_code"] == 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
