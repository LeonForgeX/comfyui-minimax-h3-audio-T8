"""Owned serial T1 encoder/full actual VAE interface qualification. No sampling."""
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
    names = ("bundle", "runtime-site", "native-vae", "source-rgb8", "run-dir")
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reference-tile", type=Path)
    parser.add_argument("--video-interface-probe", action="store_true")
    parser.add_argument("--t1-bundle", type=Path, help="Additional T1 profile for explicit encoder profile selection")
    args = parser.parse_args()
    if args.video_interface_probe:
        if args.reference_tile is not None or args.t1_bundle is None:
            raise ValueError("Video interface probe uses T17 plus T1 bundles, no T1 tile replay")
    elif args.reference_tile is None or args.t1_bundle is not None:
        raise ValueError("T1 probe requires its actual tile reference, no extra bundle")
    root = args.run_dir.resolve()
    if root.exists() or not root.is_relative_to(RESEARCH):
        raise ValueError("Need a new research run directory")
    worker = Path(__file__).with_name("trt_vae_full_encoder_worker.py" if args.video_interface_probe else "trt_vae_t1_probe_worker.py")
    core = PROJECT.parents[1]
    paths = [Path(__file__), worker, args.native_vae, args.source_rgb8,
             args.bundle / "model.engine", args.bundle / "manifest.json"]
    if args.video_interface_probe:
        paths += [args.t1_bundle / "model.engine", args.t1_bundle / "manifest.json"]
    else:
        paths.append(args.reference_tile)
    paths += [PROJECT / name for name in ("trt_vae_build.py", "trt_vae_backend.py", "trt_vae_engine.py",
              "trt_vae_interface.py", "trt_vae_contract.py", "trt_vae_encode.py", "trt_vae_decode.py",
              "dlss_fi_backend/process.py", "dlss_fi_backend/resources.py")]
    paths += [core / name for name in ("comfy/ldm/minimax/vae.py", "comfy/sd.py", "comfy/ops.py",
              "comfy/model_management.py", "comfy/model_patcher.py", "comfy/utils.py", "comfy/cli_args.py")]
    paths += [p for p in args.runtime_site.rglob("*") if p.is_file() and p.suffix in (".py", ".pyd", ".dll")]
    sources = {str(path.resolve(strict=True)): digest_file(path) for path in paths}
    request = {name.replace("-", "_"): str(getattr(args, name.replace("-", "_")).resolve(strict=True))
               for name in names if name != "run-dir"}
    request.update(sources=sources, core_root=str(core), serial_leases_held_by_controller=True)
    if args.video_interface_probe:
        request.update(video_interface_probe=True, t1_bundle=str(args.t1_bundle.resolve(strict=True)))
    else:
        request["reference_tile"] = str(args.reference_tile.resolve(strict=True))
    root.mkdir(parents=True, exist_ok=False)
    write_new_json(root / "request.json", request)
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    for path, sha in sources.items():
        if Path(path).suffix == ".py":
            shutil.copyfile(path, snapshot / (sha + "-" + Path(path).name))
    terminal = {"status": "not_started"}
    try:
        with ExitStack() as stack:
            for path in (RESEARCH / "serial-gpu.lock", Path(tempfile.gettempdir()) / "T8-DLSS-FI-serial.lock",
                         Path(tempfile.gettempdir()) / "T8-TRT-VAE-serial.lock"):
                stack.enter_context(SerialProbeLease(path))
            reader = stack.enter_context(NvmlResourceReader())
            # Explicit12000MiB admission matches both GuardPolicy's default and
            # the compiler. RAM is increased to24GiB; active1GiB GPU floor stays.
            # Video encoder inference has no compiler workspace. Prior full73
            # image+video measurement and scoped9GiB admission support10GiB;
            # original T1 qualification threshold remains unchanged.
            guard = ResourceGuard(GuardPolicy(startup_free_gpu_bytes=10*1024**3 if args.video_interface_probe else 12000*1024**2,
                                  startup_free_ram_bytes=24*1024**3,
                                  minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8*1024**3))
            first = reader.sample()
            reason = guard.observe(first, startup=True)
            write_new_json(root / "preflight.json", {"snapshot": first, "reason": reason})
            if reason:
                raise RuntimeError(reason)
            if not args.execute:
                terminal = {"status": "preflight_only"}
                return
            cancel, last = threading.Event(), time.perf_counter()
            with (root / "resources.jsonl").open("x", encoding="utf8") as log:
                log.write(json.dumps(first) + "\n")
                def check():
                    nonlocal last
                    if (root / "cancel.request").exists():
                        cancel.set()
                        raise InterruptedError("T1 probe cancelled")
                    if time.perf_counter() - last < .25:
                        return
                    row = reader.sample()
                    last = time.perf_counter()
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if problem := guard.observe(row):
                        raise RuntimeError(problem)
                try:
                    receipt = run_isolated(worker, ["--request", str(root / "request.json")], timeout=600, cancel=cancel, check=check)
                    terminal = {"status": "complete_requires_independent_audit", "isolated": receipt}
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
