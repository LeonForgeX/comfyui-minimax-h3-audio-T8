"""Explicit single-frame export in an owned, guarded, serial GPU worker."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT / "artifacts/acceleration-research-20260909"
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from dlss_fi_backend.process import run_isolated, IsolatedTaskError  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-vae", "source-rgb8", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root.exists() or not root.is_relative_to(RESEARCH) or shutil.disk_usage(RESEARCH).free < 3 * 1024**3:
        raise ValueError("Need new research output directory and3GiB disk")
    if digest_file(args.native_vae) != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Unexpected native VAE identity")
    worker = Path(__file__).with_name("trt_vae_t1_export_worker.py")
    paths = [Path(__file__), worker, args.native_vae, args.source_rgb8, PROJECT / "tools/trt_vae_encoder_probe_worker.py",
             PROJECT / "trt_vae_build.py", PROJECT / "dlss_fi_backend/process.py", PROJECT / "dlss_fi_backend/resources.py"]
    paths += [PROJECT.parents[1] / name for name in ("comfy/ldm/minimax/vae.py", "comfy/ops.py", "comfy/model_management.py")]
    sources = {str(path.resolve()): digest_file(path) for path in paths}
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    for path, sha in sources.items():
        if Path(path).suffix == ".py":
            shutil.copyfile(path, snapshot / (sha + "-" + Path(path).name))
    terminal = {"status": "not_started"}
    try:
        with ExitStack() as stack:
            for path in (RESEARCH / "serial-gpu.lock", Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock", Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(path))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(GuardPolicy(minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8*1024**3))
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            write_new_json(root / "preflight.json", {"snapshot": first, "reason": reason})
            if reason:
                raise RuntimeError(reason)
            request = {"native_vae": str(args.native_vae.resolve()), "source_rgb8": str(args.source_rgb8.resolve()),
                       "sources": sources, "gpu_uuid": first["gpu_uuid"]}
            write_new_json(root / "request.json", request)
            if not args.execute:
                terminal = {"status": "preflight_only"}
                return
            cancel = threading.Event()
            last = time.perf_counter()
            with (root / "resources.jsonl").open("x", encoding="utf8") as log:
                log.write(json.dumps(first) + "\n")
                def check():
                    nonlocal last
                    if (root / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("T1 export cancelled")
                    if time.perf_counter() - last < .25:
                        return
                    row = reader.sample()
                    last = time.perf_counter()
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if error := guard.observe(row):
                        raise RuntimeError(error)
                try:
                    receipt = run_isolated(worker, ["--request", str(root / "request.json")], timeout=600, cancel=cancel, check=check)
                    terminal = {"status": "export_worker_complete_not_trt_qualified", "isolated": receipt}
                finally:
                    write_new_json(root / "resource-summary.json", guard.report())
    except IsolatedTaskError as error:
        terminal = {"status": "worker_failed", "isolated": error.receipt}
        raise
    except BaseException as error:
        terminal = {"status": "controller_failed", "error": str(error)}
        raise
    finally:
        write_new_json(root / "terminal.json", terminal)
    print(json.dumps(terminal, indent=2))


if __name__ == "__main__":
    main()
