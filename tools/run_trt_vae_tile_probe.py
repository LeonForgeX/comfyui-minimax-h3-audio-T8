"""Explicit serial GPU tile check, same owned process/guard infrastructure as compilation."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dlss_fi_backend.process import IsolatedTaskError, run_isolated  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "runtime-site", "latent", "native-vae", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    research = PROJECT / "artifacts/acceleration-research-20260909"
    run = args.run_dir.resolve()
    if not run.is_relative_to(research) or run.exists():
        raise ValueError("New research run directory required")
    sources = [args.bundle / "manifest.json", args.bundle / "model.engine", args.latent, args.native_vae,
               PROJECT / "trt_vae_engine.py", PROJECT / "trt_vae_build.py", Path(__file__),
               Path(__file__).with_name("trt_vae_tile_probe_worker.py"), PROJECT.parent.parent / "comfy/ldm/minimax/vae.py"]
    sources += [p for p in args.runtime_site.rglob("*") if p.is_file() and p.suffix in (".py", ".pyd", ".dll")]
    request = {"bundle": str(args.bundle.resolve()), "runtime_site": str(args.runtime_site.resolve()),
               "latent": str(args.latent.resolve()), "native_vae": str(args.native_vae.resolve()),
               "core_root": str(PROJECT.parent.parent), "sources": {str(p.resolve()): digest_file(p) for p in sources}}
    run.mkdir(parents=True, exist_ok=False)
    write_new_json(run / "request.json", request)
    policy = GuardPolicy(minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8 * 1024**3)
    terminal = {"status": "not_started"}
    try:
        with ExitStack() as stack:
            for lock in (research / "serial-gpu.lock", Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock",
                         Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(lock))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(policy)
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            write_new_json(run / "preflight.json", {"snapshot": first, "reason": reason})
            if reason:
                raise RuntimeError(reason)
            cancel = threading.Event()
            last = time.perf_counter()
            with (run / "resources.jsonl").open("x", encoding="utf8") as log:
                log.write(json.dumps(first) + "\n")

                def check():
                    nonlocal last
                    if (run / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("Probe cancelled")
                    now = time.perf_counter()
                    if now - last < .25:
                        return
                    row = reader.sample()
                    last = now
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if reason := guard.observe(row):
                        raise RuntimeError(reason)

                try:
                    receipt = run_isolated(Path(__file__).with_name("trt_vae_tile_probe_worker.py"),
                                           ["--request", str(run / "request.json")], timeout=600, cancel=cancel, check=check)
                    terminal = {"status": "worker_complete_result_requires_audit", "isolated": receipt}
                finally:
                    write_new_json(run / "resource-summary.json", guard.report())
    except IsolatedTaskError as error:
        terminal = {"status": "worker_failed", "isolated": error.receipt}
        raise
    except BaseException as error:
        terminal = {"status": "controller_failed", "error": str(error)}
        raise
    finally:
        write_new_json(run / "terminal.json", terminal)
    print(json.dumps(terminal, indent=2))


if __name__ == "__main__":
    main()
