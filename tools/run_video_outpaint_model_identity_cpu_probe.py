"""Load the supplied real diffusion checkpoint on CPU and validate state binding.

No diffusion forward, GPU allocation, model conversion or model-file write.
Checks RAM headroom before loading; only a small JSON report is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types


def run(args):
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.sd
    import psutil
    import torch

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("outpaint_model_identity_cpu")
    pkg.__path__ = [str(root)]
    sys.modules[pkg.__name__] = pkg
    module = importlib.import_module(pkg.__name__+".video_outpaint_identity")
    atomic = importlib.import_module(pkg.__name__+".long_video_delivery")._atomic_write_bytes
    source = args.model.resolve(strict=True)
    free = psutil.virtual_memory().available
    required = 2*source.stat().st_size+8*1024**3
    if free < required:
        raise RuntimeError(f"CPU RAM preflight needs {required} bytes, has {free}")
    args.run_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": "t8.h3.outpaint.real_model_identity_cpu/v1", "status": "running",
              "model": str(source), "model_bytes": source.stat().st_size, "initial_free_ram_bytes": free,
              "device": "cpu", "diffusion_forward_run": False, "cuda_tested": False,
              "identity_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}

    def save():
        atomic(args.run_root / "report.json", json.dumps(report, indent=2).encode())

    save()
    try:
        initial = source.stat()
        begin = time.monotonic()
        model = comfy.sd.load_diffusion_model(str(source), disable_dynamic=True)
        report["load_seconds"] = time.monotonic()-begin
        if args.kj_memory:
            patches = importlib.import_module(pkg.__name__+".video_outpaint_model_patches")
            kj_path = root.parent / "ComfyUI-KJNodes/nodes/minimax_nodes.py"
            patches.kj_source_contract(kj_path.read_bytes())
            spec = importlib.util.spec_from_file_location("outpaint_cpu_probe_kj", kj_path)
            kj = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = kj
            spec.loader.exec_module(kj)
            model = kj.MiniMaxLowVRAMAttention.execute(model, 4).result[0]
            model = kj.MiniMaxChunkFeedForward.execute(model, 4, 4096).result[0]
            report["kj_object_patch_count"] = len(model.object_patches)
        print("CPU checkpoint loaded; computing actual state identity", flush=True)
        begin = time.monotonic()
        report["identity"] = module.native_stock_model_identity(model)
        report["identity_seconds"] = time.monotonic()-begin
        print(json.dumps(report["identity"]), flush=True)
        if module.native_stock_model_identity(model) != report["identity"]:
            raise ValueError("unchanged loaded model fingerprint is unstable")
        after = source.stat()
        if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("source checkpoint changed during the read-only probe")
        report["status"] = "passed_real_loaded_stock_model_identity_cpu_only"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kj-memory", action="store_true", help="Validate the pinned reference KJ attention/FFN pair on the real loaded CPU model")
    run(parser.parse_args())
