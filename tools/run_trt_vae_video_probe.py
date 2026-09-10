"""Explicit complete-video experiment; serial GPU leases, owned Job, resource guard."""

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


def probe_policy(*, boundary=False, saved_reference=False, special=False):
    # Boundary inference does not allocate a4GiB builder workspace. Its largest
    # tile matches the already-qualified decoder; frame buffers are smaller
    # than the measured0.5MP/73-frame scoped run. Keep9GiB backend admission
    # plus1GiB initial headroom, and the same active physical1GiB abort floor.
    # This does NOT change the compiler or existing complete-video policy.
    # Saved-reference decode uses the already measured scoped complete73-frame
    # route, without native GPU allocation or a compiler workspace.
    if boundary or saved_reference or special:
        return GuardPolicy(startup_free_gpu_bytes=10*1024**3,
                           minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8*1024**3)
    return GuardPolicy(minimum_free_gpu_bytes=1024**3, minimum_free_ram_bytes=8*1024**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "runtime-site", "latent", "native-vae", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-budget-mib", type=int, default=1024)
    parser.add_argument("--warmup-complete-decodes", type=int, choices=(0, 1), default=0,
                        help="One full untimed decode per route before the measured complete decode")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--interface-probe", action="store_true", help="Real scoped Loader lifecycle on saved short latent; no native rerun")
    parser.add_argument("--saved-reference-probe", action="store_true", help="Complete short scoped decode against already audited native RGB; no sampling/native rerun")
    parser.add_argument("--special-interface-probe", action="store_true", help="Actual Core/scoped/bounded-outpaint decode of saved39-frame prefix")
    parser.add_argument("--reference-rgb", type=Path, help="Earlier identity-bound TRT RGB for interface assembly comparison")
    parser.add_argument("--boundary-probe", action="store_true", help="Real T1/small-tile and short temporal boundary comparisons, no sampling")
    parser.add_argument("--image-latent", type=Path, help="Audited actual image native_latent for genuine single-image decode")
    args = parser.parse_args()
    saved_mode = args.interface_probe or args.saved_reference_probe
    if args.special_interface_probe and (saved_mode or args.boundary_probe or args.warmup_complete_decodes):
        raise ValueError("Special interface probe cannot mix other experiment modes")
    if (saved_mode != (args.reference_rgb is not None) or (saved_mode and args.warmup_complete_decodes)
            or (args.interface_probe and args.saved_reference_probe)):
        raise ValueError("Interface probe requires saved reference RGB and no warmup option")
    if args.boundary_probe != (args.image_latent is not None) or (args.boundary_probe and (saved_mode or args.warmup_complete_decodes)):
        raise ValueError("Boundary probe requires image latent and cannot mix other probe modes")
    worker = Path(__file__).with_name("trt_vae_interface_probe_worker.py" if args.interface_probe else "trt_vae_video_probe_worker.py")
    if args.boundary_probe:
        worker = Path(__file__).with_name("trt_vae_boundary_probe_worker.py")
    if args.saved_reference_probe:
        worker = Path(__file__).with_name("trt_vae_saved_reference_worker.py")
    if args.special_interface_probe:
        worker = Path(__file__).with_name("trt_vae_special_worker.py")
    research = PROJECT / "artifacts/acceleration-research-20260909"
    run = args.run_dir.resolve()
    if not run.is_relative_to(research) or run.exists():
        raise ValueError("New research run directory required")
    if not 1 <= args.output_budget_mib <= 8192:
        raise ValueError("Explicit RGB buffer budget must be 1..8192 MiB")
    if shutil.disk_usage(research).free < 3 * args.output_budget_mib * 1024**2 + 1024**3:
        raise ValueError("Insufficient disk space for two RGB outputs and evidence")
    core = PROJECT.parent.parent
    sources = [args.bundle / "manifest.json", args.bundle / "model.engine", args.latent, args.native_vae,
               Path(__file__), worker]
    reference_sources = {}
    if args.saved_reference_probe:
        from tools.trt_vae_saved_reference import bind_reference
        reference_sources = bind_reference(args.reference_rgb, args.latent, args.native_vae)
        sources += [Path(path) for path in reference_sources]
        sources += [PROJECT / "tools/trt_vae_saved_reference.py"]
    if saved_mode:
        sources += [args.reference_rgb, PROJECT / 'h3_t8/trt_vae_backend.py', PROJECT / 'h3_t8/trt_vae_interface.py', PROJECT / 'h3_t8/trt_vae_encode.py',
                    core / "comfy/sd.py"]
    if args.boundary_probe:
        sources += [args.image_latent, PROJECT / 'h3_t8/trt_vae_backend.py', PROJECT / 'h3_t8/trt_vae_interface.py', PROJECT / 'h3_t8/trt_vae_encode.py',
                    PROJECT / "tools/trt_vae_video_probe_worker.py", core / "comfy/sd.py"]
    if args.special_interface_probe:
        sources += [PROJECT / name for name in ('h3_t8/trt_vae_backend.py', 'h3_t8/trt_vae_interface.py', 'h3_t8/trt_vae_encode.py',
                                                'h3_t8/video_outpaint_plan.py', 'h3_t8/video_outpaint_decode.py')]
        sources += [core / "comfy/sd.py", core / "comfy/model_patcher.py", core / "comfy/utils.py"]
    sources += [PROJECT / name for name in ('h3_t8/trt_vae_engine.py', 'h3_t8/trt_vae_build.py', 'h3_t8/trt_vae_contract.py', 'h3_t8/trt_vae_decode.py',
                                           'h3_t8/dlss_fi_backend/process.py', 'h3_t8/dlss_fi_backend/resources.py')]
    sources += [core / name for name in ("comfy/ldm/minimax/vae.py", "comfy/ops.py", "comfy/ldm/modules/attention.py",
                                        "comfy/model_management.py", "comfy/cli_args.py")]
    sources += [p for p in args.runtime_site.rglob("*") if p.is_file() and p.suffix in (".py", ".pyd", ".dll")]
    request = {"bundle": str(args.bundle.resolve()), "runtime_site": str(args.runtime_site.resolve()),
               "latent": str(args.latent.resolve()), "native_vae": str(args.native_vae.resolve()),
               "core_root": str(core), "output_budget_bytes": args.output_budget_mib * 1024**2,
               "warmup_complete_decodes": args.warmup_complete_decodes,
               "sources": {str(p.resolve()): digest_file(p) for p in sources}}
    if args.interface_probe:
        request.update(interface_probe=True, reference_rgb=str(args.reference_rgb.resolve()),
                       serial_leases_held_by_controller=True)
    if args.saved_reference_probe:
        request.update(saved_reference_probe=True, reference_rgb=str(args.reference_rgb.resolve()),
                       reference_sources=reference_sources, serial_leases_held_by_controller=True)
    if args.boundary_probe:
        request.update(boundary_probe=True, image_latent=str(args.image_latent.resolve()), serial_leases_held_by_controller=True)
    if args.special_interface_probe:
        request.update(special_interface_probe=True, serial_leases_held_by_controller=True)
    run.mkdir(parents=True, exist_ok=False)
    write_new_json(run / "request.json", request)
    # Preserve tested source identity before subsequent development can change it.
    snapshot = run / "source-snapshot"
    snapshot.mkdir()
    for path, digest in request["sources"].items():
        if Path(path).suffix == ".py" and Path(path).is_relative_to(PROJECT):
            target = snapshot / (digest + "-" + Path(path).name)
            shutil.copyfile(path, target)
            if digest_file(target) != digest:
                raise RuntimeError("Source snapshot identity mismatch")
    policy = probe_policy(boundary=args.boundary_probe, saved_reference=args.saved_reference_probe, special=args.special_interface_probe)
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
            if not args.execute:
                terminal = {"status": "preflight_only_no_gpu_worker"}
                return
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
                    receipt = run_isolated(worker,
                                           ["--request", str(run / "request.json")], timeout=1800, cancel=cancel, check=check)
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
