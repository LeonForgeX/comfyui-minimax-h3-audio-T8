"""Explicit, serial, bounded research feature probe; default is identity-only.

Does not install DLLs, change HAGS/drivers, open UI, or qualify a video route.
Only the fixed research worker is accepted. Runtime output is evidence, not code.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dlss_fi_transport import runtime_identity  # noqa: E402
from run_progressive_pilot import RESEARCH, OwnedServer, ContinuousGuard, write_json  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402


def parse_probe(data):
    if not isinstance(data, bytes) or not data or len(data) > 65536:
        raise ValueError("Probe stdout must be a nonempty bounded byte response")
    lines = [line for line in data.decode("utf-8", "strict").splitlines() if line.strip()]
    response = json.loads(lines[-1])
    if not isinstance(response, dict) or type(response.get("available")) is not bool:
        raise ValueError("Probe lacks a boolean feature availability result")
    maximum = response.get("multi_frame_count_max")
    if type(maximum) is not int or not 0 <= maximum <= 32:
        raise ValueError("Probe returned an invalid native generated-frame maximum")
    for key in ("runtime_version", "worker_version", "detail"):
        if not isinstance(response.get(key), str) or len(response[key]) > 8192:
            raise ValueError("Probe lacks bounded version/detail evidence")
    return {"raw": response, "supports_native_2x_by_report": response["available"] and maximum >= 1,
        "actual_device_qualified": False, "generation_qualified": False, "quality_qualified": False}


def collect(owner, check, *, timeout=45, cancel=None):
    """Bounded text pipes. The caller's Popen owner is always stopped on failure."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        owner.stop()
        raise ValueError("Probe timeout must be finite, positive and at most 120 seconds")
    cancel = cancel if cancel is not None else threading.Event()
    stdout, stderr, failures = bytearray(), deque(maxlen=16), []
    def drain(pipe, is_stdout):
        try:
            while block := pipe.read(4096):
                if is_stdout:
                    if len(stdout) + len(block) > 65536:
                        raise ValueError("Probe stdout exceeded 64KiB")
                    stdout.extend(block)
                else:
                    stderr.append(block)
        except BaseException as error:
            failures.append(error)
    threads = [threading.Thread(target=drain, args=(owner.process.stdout, True), daemon=True),
               threading.Thread(target=drain, args=(owner.process.stderr, False), daemon=True)]
    for thread in threads:
        thread.start()
    deadline, modules = time.perf_counter()+timeout, set()
    try:
        import psutil
        observed = psutil.Process(owner.process.pid)
        while owner.process.poll() is None:
            check()
            if cancel.is_set():
                raise RuntimeError("Probe cancelled")
            if failures:
                raise failures[0]
            if time.perf_counter() >= deadline:
                raise TimeoutError("Feature probe exceeded its deadline")
            try:
                modules.update(row.path for row in observed.memory_maps(grouped=True) if "dlss" in row.path.lower())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.05)
        for thread in threads:
            thread.join(timeout=1)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("Probe left inherited output handles open")
        if failures:
            raise failures[0]
        if cancel.is_set():
            raise RuntimeError("Probe cancelled")
        check()
        if owner.process.returncode:
            raise RuntimeError(f"Native feature probe exited {owner.process.returncode}")
        return {"stdout": bytes(stdout), "stderr_tail": b"".join(stderr), "observed_dlss_modules": sorted(modules)}
    finally:
        owner.stop()
        for thread in threads:
            thread.join(timeout=2)
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError("Probe cleanup left pipe threads active; do not start more GPU work")
        owner.process.stdout.close()
        owner.process.stderr.close()
        owner.probe_output = {"stdout": bytes(stdout).decode("utf-8", "replace"),
            "stderr_tail": b"".join(stderr).decode("utf-8", "replace"), "observed_dlss_modules": sorted(modules)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "probe"), default="plan")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    identities = runtime_identity(args.runtime)
    if args.mode == "plan":
        print(json.dumps({"mode": "identity_only_no_process_or_GPU", "identities": identities}))
        return
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError("New dedicated research directory required; no retries")
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        root.mkdir(parents=True)
        receipt = {"status": "incomplete", "device_qualified": False, "video_qualified": False}
        owner, monitor = OwnedServer(root, 0, False), None
        try:
            write_json(root / "runtime-identity.json", identities)
            write_json(root / "probe-source.json", {"controller": file_identity(__file__),
                "transport": file_identity(Path(__file__).with_name("dlss_fi_transport.py"))})
            with NvmlResourceReader() as reader:
                guard = ResourceGuard()
                startup = reader.sample()
                guard.observe(startup, startup=True)
                write_json(root / "startup.json", startup)
                if guard.reason:
                    raise RuntimeError(guard.reason)
                command = [identities["dlssg-worker.exe"]["path"], "--probe"]
                write_json(root / "command.json", command)
                owner.process = subprocess.Popen(command, cwd=args.runtime.resolve(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                print(f"Owned DLSSG feature probe PID {owner.process.pid}", flush=True)
                monitor = ContinuousGuard(reader, guard, root / "resources.jsonl", owner)
                monitor.start()
                try:
                    collected = collect(owner, monitor.check)
                    result = parse_probe(collected["stdout"])
                    result["stderr_tail"] = collected["stderr_tail"].decode("utf-8", "replace")
                    result["observed_dlss_modules"] = collected["observed_dlss_modules"]
                    result["observed_module_identities"] = [file_identity(p) for p in collected["observed_dlss_modules"]]
                    if runtime_identity(args.runtime) != identities:
                        raise RuntimeError("Runtime files changed during feature probe")
                    write_json(root / "probe-result.json", result)
                    receipt["status"] = "feature_probe_completed_device_and_generation_unqualified"
                    receipt["supports_native_2x_by_report"] = result["supports_native_2x_by_report"]
                finally:
                    monitor.close()
                    monitor.check()
                    receipt["resources"] = guard.report()
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            owner.stop()
            if hasattr(owner, "probe_output"):
                write_json(root / "probe-output.json", owner.probe_output)
            receipt["server_stop"] = owner.stop_receipt
            write_json(root / "terminal.json", receipt)
            print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
