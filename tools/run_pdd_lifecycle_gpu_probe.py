"""Isolated PDD fail-then-success lifecycle probe for the development checkout.

The first prompt deliberately reaches the fallback PDD final-head wrapper with
an off-grid first sigma. The *same owned Core process* must then complete the
released strength-1.0, eight-NFE PDD route with strict H.264/AAC media checks.
This is a mechanical recovery probe, not a perceptual-quality qualification.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import sys
import uuid


TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_clear_clipproj_triplet_probe as clipprobe  # noqa: E402
import run_nfe_resume_real_probe as shared  # noqa: E402
import run_pdd_real_validation as pdd  # noqa: E402
import run_progressive_pilot as pilot  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


SCHEMA = "t8.minimax_h3.pdd_fail_then_success_lifecycle.v1"
EXPECTED_FAILURE = "PDD received a sigma outside its official 8-step"
BAD_FIRST_SIGMA = 0.999


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--comfy-root", type=Path, default=PROJECT.parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--port", type=int, default=8861)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frame-count", type=int, default=22)
    parser.add_argument("--min-free-vram-mib", type=int, default=12000)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT / "artifacts" / "development" / "pdd-lifecycle")
    parser.add_argument("--dry-run", action="store_true", help="Inspect paths and graph only; never start Core or query CUDA")
    return parser


def _pdd_args(args: argparse.Namespace) -> argparse.Namespace:
    values = pdd._parser().parse_args(["--variant", args.variant])
    values.project_root = PROJECT
    values.comfy_root = args.comfy_root.resolve()
    values.python = args.python.resolve()
    values.host = "127.0.0.1"
    values.port = args.port
    values.width = args.width
    values.height = args.height
    values.frame_count = args.frame_count
    values.min_free_vram_mib = args.min_free_vram_mib
    values.timeout_seconds = args.timeout_seconds
    values.server_start_timeout = args.server_start_timeout
    return values


def _graphs(args: argparse.Namespace, run_id: str) -> tuple[dict, dict]:
    bad = pdd._prompt(args, f"{run_id}-bad")
    # Strength below 1 deliberately uses the T8 fallback head lifecycle. A
    # tiny off-grid change keeps the sigma schedule decreasing and reaches
    # its wrapper on the first *actual model forward*, after MODEL load.
    bad["8"]["inputs"]["strength"] = 0.99
    bad["15"] = {"class_type": "SetFirstSigma", "inputs": {
        "sigmas": ["8", 2], "sigma": BAD_FIRST_SIGMA,
    }}
    bad["11"]["inputs"]["sigmas"] = ["15", 0]
    good = pdd._prompt(args, f"{run_id}-good")
    return bad, good


def _failure_message(phase: dict) -> str:
    # Core versions retain error detail in terminal WS data and/or history.
    return json.dumps({
        "terminal": phase.get("terminal"),
        "history_status": (phase.get("history") or {}).get("status"),
    }, ensure_ascii=False)


def _assert_expected_failure(phase: dict) -> None:
    if (phase.get("terminal") or {}).get("type") != "execution_error":
        raise RuntimeError("The deliberately off-grid PDD prompt did not end with execution_error")
    if EXPECTED_FAILURE not in _failure_message(phase):
        raise RuntimeError("PDD failure was not the intended first-forward sigma guard; inspect bad-phase.json")


def _media_executable(comfy_root: Path, name: str) -> str | None:
    command = shutil.which(name)
    if command:
        return command
    bundled = comfy_root.parent / "ffmpeg" / "bin" / f"{name}.exe"
    return str(bundled) if bundled.is_file() else None


def _preflight(args: argparse.Namespace) -> dict:
    paths = pdd._paths(args)
    ffmpeg = _media_executable(args.comfy_root, "ffmpeg")
    ffprobe = _media_executable(args.comfy_root, "ffprobe")
    checks = {
        "all_required_paths_present": all(path.exists() for path in paths.values()),
        "ffmpeg_and_ffprobe": bool(ffmpeg and ffprobe),
        "isolated_port_free": not shared.port_is_listening(args.host, args.port),
        "project_is_development_checkout": PROJECT.resolve() != (args.comfy_root / "custom_nodes" / "minimax-h3-audio-T8").resolve(),
    }
    gpu = shared.gpu_memory_mib()
    checks["free_vram_gate"] = bool(gpu.get("available") and int(gpu.get("free_mib") or 0) >= args.min_free_vram_mib)
    return {"paths": {key: str(value) for key, value in paths.items()},
            "ffmpeg": ffmpeg, "ffprobe": ffprobe, "gpu": gpu, "checks": checks,
            "ready": all(checks.values())}


def _run(args: argparse.Namespace, run_root: Path, bad: dict, good: dict, preflight: dict) -> int:
    config = probe_resource_config(args.comfy_root, PROJECT)
    pilot.write_json(run_root / "paths.json", config)
    pilot.CORE = args.comfy_root
    pilot.PROJECT = PROJECT
    original_command = pilot.server_command

    def command_with_vhs(*command_args, **command_kwargs):
        command = original_command(*command_args, **command_kwargs)
        command[0] = str(args.python)
        marker = command.index("--extra-model-paths-config")
        command[marker:marker] = ["ComfyUI-VideoHelperSuite"]
        return command

    pilot.server_command = command_with_vhs
    monitor = clipprobe.GpuPeakMonitor(interval_seconds=0.25)
    server = pilot.OwnedServer(run_root, args.port, False)
    report = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
              "variant": args.variant, "run_id": "pdd-lifecycle-good",
              "source_project": str(PROJECT.resolve()), "preflight": preflight,
              "contract": {"first": "fallback_strength_0.99_offgrid_first_forward_error",
                           "second": "native_strength_1.0_official_eight_nfe_h264_aac",
                           "same_core_process_required": True,
                           "bad_first_sigma": BAD_FIRST_SIGMA}}
    try:
        server.start()
        pilot.wait_ready(server, lambda: None, seconds=args.server_start_timeout)
        pid = server.process.pid
        report["owned_core_pid"] = pid
        monitor.start()
        bad_phase = asyncio.run(pdd._submit_prompt_capture(
            server=server.url, prompt=bad, timeout_seconds=args.timeout_seconds))
        report["bad_phase"] = bad_phase
        (run_root / "bad-phase.json").write_text(json.dumps(bad_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        _assert_expected_failure(bad_phase)
        server.assert_port_owner()
        if server.process.pid != pid or server.process.poll() is not None:
            raise RuntimeError("Core process did not survive the intended PDD failure")
        good_phase = asyncio.run(pdd._submit_prompt_capture(
            server=server.url, prompt=good, timeout_seconds=args.timeout_seconds))
        report["good_phase"] = good_phase
        (run_root / "good-phase.json").write_text(json.dumps(good_phase, ensure_ascii=False, indent=2), encoding="utf-8")
        server.assert_port_owner()
        if server.process.pid != pid or (good_phase.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("The second official PDD prompt did not complete in the same Core process")
        report["same_process_recovery"] = True
    except BaseException as error:
        report["status"] = "FAIL_LIFECYCLE"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            report["gpu_monitor"] = monitor.stop()
            server.stop()
            report["owned_server_shutdown"] = server.stop_receipt
        finally:
            pilot.server_command = original_command
            (run_root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # The existing real PDD validator performs strict full video/audio decode,
    # native head/adapter accounting and nonempty finite PCM checks.
    # The reused validator prints its entire large websocket/history receipt.
    # Keep that on disk, but emit only a short controller result to the shell.
    with redirect_stdout(io.StringIO()):
        result = pdd._finalize_completed_run(
            args=args, run_root=run_root, phase=report["good_phase"], report=report,
            ffmpeg=preflight["ffmpeg"], ffprobe=preflight["ffprobe"])
    report["bad_phase_expected_error"] = True
    report["same_process_recovery"] = True
    report["status"] = "PDD_LIFECYCLE_MECHANICAL_PASS_HUMAN_REVIEW_PENDING" if result == 0 else "PDD_LIFECYCLE_FAIL_MEDIA_OR_SETUP"
    (run_root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_root": str(run_root),
                      "owned_core_pid": report["owned_core_pid"],
                      "bad_terminal": report["bad_phase"]["terminal"]["type"],
                      "good_terminal": report["good_phase"]["terminal"]["type"],
                      "minimum_free_vram_mib": report["gpu_monitor"]["minimum_free_mib"],
                      "output_video": report.get("output_video")}, ensure_ascii=False), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    args = _pdd_args(options)
    pdd._validate_contract(args)
    bad, good = _graphs(args, "pdd-lifecycle")
    if options.dry_run:
        print(json.dumps({"schema": SCHEMA, "dry_run": True,
                          "development_checkout": str(PROJECT.resolve()),
                          "comfy_root": str(args.comfy_root),
                          "required_paths": {key: str(value) for key, value in pdd._paths(args).items()},
                          "bad_sampler_sigmas": bad["11"]["inputs"]["sigmas"],
                          "bad_strength": bad["8"]["inputs"]["strength"],
                          "good_sampler_sigmas": good["11"]["inputs"]["sigmas"],
                          "good_strength": good["8"]["inputs"]["strength"]}, ensure_ascii=False, indent=2))
        return 0
    preflight = _preflight(args)
    print(json.dumps({"schema": SCHEMA + ".preflight", **preflight}, ensure_ascii=False), flush=True)
    if not preflight["ready"]:
        return 2
    run_root = options.artifact_root.resolve() / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=False)
    pilot.write_json(run_root / "bad-prompt.json", bad)
    pilot.write_json(run_root / "good-prompt.json", good)
    return _run(args, run_root, bad, good, preflight)


if __name__ == "__main__":
    raise SystemExit(main())
