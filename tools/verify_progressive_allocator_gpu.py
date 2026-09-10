"""Explicit, bounded allocator verification; no H3 weights, inference or browser.

Only --run-gpu starts a child. Peak payload is 96 MiB plus the CUDA context.
Synchronization belongs to this oracle workload, never the telemetry reader.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_memory_metrics import AllocatorMemorySession, validate_completed_interval  # noqa: E402
from progressive_probe_control import MIB, NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402
from run_progressive_pilot import ContinuousGuard, OwnedServer, RESEARCH, write_json  # noqa: E402


def verify_payload(report, first_bytes=32*MIB, transient_bytes=64*MIB):
    rows = report["snapshots"]
    if [r["boundary"] for r in rows] != ["before_generation", "first_live", "both_live", "after_generation_terminal"]:
        raise ValueError("Missing allocation-oracle boundaries")
    start, first, both, freed = rows
    if first["allocated_current_bytes"] < start["allocated_current_bytes"] + first_bytes:
        raise ValueError("Allocator did not observe the first known allocation")
    if both["allocated_current_bytes"] < first["allocated_current_bytes"] + transient_bytes:
        raise ValueError("Allocator did not observe the transient allocation")
    if freed["allocated_peak_bytes"] < both["allocated_current_bytes"]:
        raise ValueError("Released transient high-water mark was lost")
    if freed["allocated_current_bytes"] >= both["allocated_current_bytes"]:
        raise ValueError("Allocator did not observe release after synchronization")


def worker(root):
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.is_initialized():
        raise RuntimeError("Worker must start fresh with exactly one selected GPU")
    torch.cuda.init()
    if torch.cuda.get_allocator_backend() != "cudaMallocAsync":
        raise RuntimeError("Worker allocator differs from the H3 pilot backend")
    # Establish CUDA pool without including lazy context setup in the interval.
    warm = torch.empty(1, device="cuda:0", dtype=torch.uint8)
    del warm
    torch.cuda.synchronize(0)
    probe = AllocatorMemorySession(run_id=root.name, device_type="cuda", device_index=0)
    try:
        probe.begin()
        first = torch.empty(32*MIB, device="cuda:0", dtype=torch.uint8)
        first.fill_(17)
        torch.cuda.synchronize(0)
        probe.observe("first_live")
        transient = torch.empty(64*MIB, device="cuda:0", dtype=torch.uint8)
        transient.fill_(23)
        torch.cuda.synchronize(0)
        probe.observe("both_live")
        assert first[0].item() == 17 and transient[-1].item() == 23
        del transient, first
        torch.cuda.synchronize(0)
        report = probe.finish()
        verify_payload(report)
        write_json(root / "allocator-memory.json", report)
        write_json(root / "worker-result.json", {"status": "known_allocation_oracle_pass",
            "pid": os.getpid(), "maximum_payload_bytes": 96*MIB,
            "workload_synchronizes": True, "telemetry_forces_sync": False,
            "model_loaded": False, "h3_memory_or_performance_qualification": False})
    except BaseException:
        write_json(root / "failed-allocator.json", probe.report())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError("Use a dedicated acceleration research directory")
    if args.worker:
        if os.environ.get("T8_ALLOCATOR_ORACLE_CHILD") != str(root):
            raise RuntimeError("Worker requires the explicit parent launch")
        worker(root)
        return
    if not args.run_gpu:
        print("No GPU action. Add --run-gpu to run the bounded 96 MiB payload oracle.")
        return
    if root.exists():
        raise ValueError("Do not overwrite or automatically retry an existing run")
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        root.mkdir()
        receipt = {"status": "incomplete", "scope": "allocator_oracle_only_no_H3"}
        owner, monitor = OwnedServer(root, 8197, False), None
        sources = [Path(__file__), Path(__file__).with_name("progressive_memory_metrics.py")]
        source_ids = [file_identity(p) for p in sources]
        write_json(root / "sources.json", source_ids)
        nvml = None
        try:
            nvml = NvmlResourceReader().__enter__()
            guard = ResourceGuard()
            row = nvml.sample()
            write_json(root / "startup.json", row)
            if guard.observe(row, startup=True):
                raise RuntimeError(guard.reason)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": row["gpu_uuid"],
                "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
                "PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync",
                "T8_ALLOCATOR_ORACLE_CHILD": str(root)}
            with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
                owner.process = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                    "--worker", "--run-root", str(root)], env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                monitor = ContinuousGuard(nvml, guard, root / "resources.jsonl", owner)
                monitor.start()
                code = owner.process.wait(timeout=60)
                monitor.check()
                if code != 0:
                    raise RuntimeError(f"Allocator child failed: exit {code}; see stderr.log")
                report = json.loads((root / "allocator-memory.json").read_text(encoding="utf-8"))
                validate_completed_interval(report, run_id=root.name, pid=owner.process.pid,
                    device_type="cuda", gpu_uuid=row["gpu_uuid"])
                verify_payload(report)
                if [file_identity(p) for p in sources] != source_ids:
                    raise RuntimeError("Oracle source changed during execution")
                receipt.update(status="gpu_allocator_oracle_pass_not_H3_qualification", pid=owner.process.pid)
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if monitor:
                monitor.close()
                receipt["resources"] = monitor.guard.report()
            owner.stop()
            if nvml:
                nvml.__exit__()
            receipt["child_stop"] = owner.stop_receipt
            write_json(root / "terminal.json", receipt)
            print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
