"""Serial owned encoder qualification using real image and short video tiles."""

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
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from dlss_fi_backend.process import IsolatedTaskError, run_isolated  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy, NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "runtime-site", "image", "video", "native-vae", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scope", choices=("tile", "full_0p5mp"), default="tile")
    args = parser.parse_args()
    research = PROJECT / "artifacts/acceleration-research-20260909"
    root = args.run_dir.resolve()
    if not root.is_relative_to(research) or root.exists():
        raise ValueError("New research run directory required")
    core = PROJECT.parent.parent
    sources = [args.bundle / "manifest.json", args.bundle / "model.engine", args.image, args.video, args.native_vae,
               Path(__file__), Path(__file__).with_name("trt_vae_encoder_probe_worker.py")]
    sources += [PROJECT / name for name in ('h3_t8/trt_vae_build.py', 'h3_t8/trt_vae_engine.py', 'h3_t8/trt_vae_encode.py', 'h3_t8/trt_vae_decode.py', 'h3_t8/trt_vae_contract.py', 'h3_t8/dlss_fi_backend/process.py', 'h3_t8/dlss_fi_backend/resources.py')]
    sources += [core / name for name in ("comfy/ldm/minimax/vae.py", "comfy/ops.py", "comfy/model_management.py", "comfy/cli_args.py")]
    sources += [p for p in args.runtime_site.rglob("*") if p.is_file() and p.suffix in (".py", ".pyd", ".dll")]
    request = {name.replace("-", "_"): str(getattr(args, name.replace("-", "_")).resolve(strict=True))
               for name in ("bundle", "runtime-site", "image", "video", "native-vae")}
    request.update(core_root=str(core), scope=args.scope, sources={str(p.resolve()): digest_file(p) for p in sources})
    root.mkdir(parents=True, exist_ok=False)
    write_new_json(root / "request.json", request)
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    for path, digest in request["sources"].items():
        if Path(path).suffix == ".py" and Path(path).is_relative_to(PROJECT):
            target = snapshot / (digest + "-" + Path(path).name)
            shutil.copyfile(path, target)
            if digest_file(target) != digest:
                raise ValueError("Encoder source snapshot mismatch")
    terminal = {"status": "not_started"}
    try:
        with ExitStack() as stack:
            for path in (research / "serial-gpu.lock", Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock",
                         Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(path))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(GuardPolicy(minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8 * 1024**3))
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            write_new_json(root / "preflight.json", {"snapshot": first, "reason": reason})
            if reason:
                raise RuntimeError(reason)
            if not args.execute:
                terminal = {"status": "preflight_only_no_gpu_worker"}
                return
            cancel = threading.Event()
            last = time.perf_counter()
            with (root / "resources.jsonl").open("x", encoding="utf8") as log:
                log.write(json.dumps(first) + "\n")

                def check():
                    nonlocal last
                    if (root / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("Encoder probe cancelled")
                    if time.perf_counter() - last < .25:
                        return
                    row = reader.sample()
                    last = time.perf_counter()
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if problem := guard.observe(row):
                        raise RuntimeError(problem)

                try:
                    receipt = run_isolated(Path(__file__).with_name("trt_vae_encoder_probe_worker.py"),
                                           ["--request", str(root / "request.json")], timeout=600, cancel=cancel, check=check)
                    terminal = {"status": "worker_complete_result_requires_audit", "isolated": receipt}
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
