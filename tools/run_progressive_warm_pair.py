"""Bounded same-process ABBA pilot: two warmups, four measured jobs, strictly serial.

Default plan mode writes nothing. CPU smoke exercises six real uncached Core
queues without weights. GPU is explicit and uses the unchanged frozen recipes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_pilot import (  # noqa: E402
    CORE, RESEARCH, ContinuousGuard, OwnedServer, allocator_action, execute_graph,
    file_identity, instrument_recipe, preview_report, source_snapshot, wait_ready, write_json,
)
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease  # noqa: E402
from progressive_memory_metrics import validate_completed_interval  # noqa: E402
from progressive_pilot_analysis import audit_media, common_graph, verify_execution_report  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402
from run_progressive_pilot import PROJECT  # noqa: E402


def sequence(task):
    if task not in {"T2VA", "I2VA"}:
        raise ValueError("Warm pair supports only the fixed T2VA/I2VA pilot tasks")
    return [{"index": i, "case": task + "_" + route, "measured": i >= 2}
            for i, route in enumerate(("native8", "progressive6plus2", "native8",
                                       "progressive6plus2", "progressive6plus2", "native8"))]


def summarize(rows, task):
    plan = sequence(task)
    if len(rows) != len(plan):
        raise ValueError("A partial sequence cannot qualify the warm comparison")
    for row, item in zip(rows, plan):
        if any(row.get(k) != v for k, v in item.items()) or row.get("status") != "media_validated":
            raise ValueError("Warm sequence/order/outcome differs from the fixed plan")
        if row.get("cached") is not False:
            raise ValueError("Warm execution must not reuse graph results")
    if len({row["server_pid"] for row in rows}) != 1:
        raise ValueError("Warm samples must share one process")
    if len({row["conditioning_sha256"] for row in rows}) != 1:
        raise ValueError("Warm samples have different actual conditioning")
    values = {}
    for route in ("native8", "progressive6plus2"):
        selected = [r for r in rows if r["measured"] and r["case"] == task + "_" + route]
        seconds = [r["seconds"] for r in selected]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t < 1800 for t in seconds):
            raise ValueError("Invalid measured wall time")
        values[route] = {"seconds": seconds, "median_seconds": statistics.median(seconds),
            "range_seconds": [min(seconds), max(seconds)],
            "allocator_peak_bytes": [r["allocator_peak_bytes"] for r in selected]}
    return {"status": "same_process_ABBA_wall_only_human_review_pending", "routes": values,
        "saved_fraction_from_medians": 1-values["progressive6plus2"]["median_seconds"]/values["native8"]["median_seconds"],
        "scope": "Two measured samples per route after two unscored warmups; no statistical significance.",
        "memory_scope": "Cumulative allocator-pool peaks per job, not whole-device usage.",
        "warm_scope": "Process/library caches warm; graph cache disabled; model reload/offload remains included."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "cpu-smoke", "gpu"), default="plan")
    parser.add_argument("--task", choices=("T2VA", "I2VA"), default="T2VA")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--port", type=int, default=8197)
    args = parser.parse_args()
    plan = sequence(args.task)
    if args.mode == "plan":
        print(json.dumps({"mode": "plan_only_no_GPU_or_files", "sequence": plan}))
        return
    root = args.run_root.resolve()
    if root == RESEARCH or not root.is_relative_to(RESEARCH) or root.exists():
        raise ValueError("Use a new dedicated research directory; no automatic retry")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid isolated port")
    if args.mode == "gpu" and not all((args.identities, args.ffmpeg, args.ffprobe)):
        raise ValueError("GPU needs frozen identities and explicit FFmpeg/FFprobe")
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        root.mkdir()
        result = {"status": "incomplete", "mode": args.mode, "rows": [], "quality_qualified": False}
        server = reader = monitor = None
        try:
            sources = source_snapshot()
            sources[str(Path(__file__).relative_to(PROJECT))] = file_identity(__file__)["sha256"]
            expected = {"core": verify_core_source(CORE), "sources": sources, "mode": args.mode}
            graphs = {route: instrument_recipe(json.loads((RESEARCH / "pilot-api-drafts" /
                f"{args.task}_{route}.prompt.json").read_text(encoding="utf-8"))) for route in ("native8", "progressive6plus2")}
            if common_graph(graphs["native8"]) != common_graph(graphs["progressive6plus2"]):
                raise RuntimeError("Common recipe differs between routes")
            expected["pilot_graphs"] = graphs
            if args.mode == "gpu":
                manifest = json.loads(args.identities.read_text(encoding="utf-8"))
                if manifest["core"] != expected["core"]:
                    raise RuntimeError("Manifest Core identity differs")
                for item in manifest["installed_assets"]:
                    if file_identity(item["path"]) != item:
                        raise RuntimeError("Installed asset changed")
                for route in graphs:
                    path = RESEARCH / "pilot-api-drafts" / f"{args.task}_{route}.prompt.json"
                    if file_identity(path) != manifest["recipes"][path.name]:
                        raise RuntimeError("Fixed pilot recipe changed")
                expected["assets"] = manifest["installed_assets"]
                write_json(root / "assets-verified.json", expected["assets"])
                reader = NvmlResourceReader().__enter__()
                guard = ResourceGuard()
                start = reader.sample()
                write_json(root / "startup.json", start)
                if guard.observe(start, startup=True):
                    raise RuntimeError(guard.reason)
            write_json(root / "environment-expected.json", expected)
            write_json(root / "sequence.json", plan)
            write_json(root / "paths.json", probe_resource_config(CORE, PROJECT))
            server = OwnedServer(root, args.port, args.mode == "cpu-smoke")
            server.start()
            if reader:
                monitor = ContinuousGuard(reader, guard, root / "resources.jsonl", server)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            wait_ready(server, check)
            env_graph = {"1": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                         "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}
            history, _ = execute_graph(server, env_graph, root / "environment", check, timeout=60)
            live = preview_report(history, "2")
            if live["pid"] != server.process.pid:
                raise RuntimeError("Live process identity differs")
            write_json(root / "live-environment.json", live)
            device_type = "cpu" if args.mode == "cpu-smoke" else "cuda"
            condition_identity = None
            for item in plan:
                check()
                if (root / "STOP").exists():
                    raise RuntimeError("Stopped at user-requested inter-job boundary")
                folder = root / f"{item['index']:02d}_{item['case']}"
                folder.mkdir()
                allocator_action(server, folder, "begin", device_type, check)
                graph = deepcopy(graphs[item["case"].removeprefix(args.task + "_")])
                if args.mode == "cpu-smoke":
                    graph = {"1": {"class_type": "PreviewAny", "inputs": {"source": f"CPU sequence {item['index']}"}}}
                else:
                    graph["18"]["inputs"]["filename_prefix"] = f"warm/{folder.name}"
                history, timing = execute_graph(server, graph, folder / "generation", check)
                allocator = allocator_action(server, folder, "finish", device_type, check)
                write_json(folder / "allocator-memory.json", allocator)
                validate_completed_interval(allocator, run_id=folder.name, pid=server.process.pid,
                    device_type=device_type, gpu_uuid=start["gpu_uuid"] if reader else None)
                row = {**item, "server_pid": server.process.pid, "seconds": timing["elapsed_seconds"], "cached": False}
                if args.mode == "gpu":
                    reports = {name: preview_report(history, node) for name, node in {
                        "sampler": "21", "conditioning": "101", "decode": "102", "lora": "103", "save": "19"}.items()}
                    verify_execution_report(item["case"], graph, reports)
                    if condition_identity is None:
                        condition_identity = reports["conditioning"]
                    if condition_identity != reports["conditioning"]:
                        raise RuntimeError("Encoded conditioning or initial AV differs across warm jobs")
                    path = Path(reports["save"]["output"]).resolve(strict=True)
                    if not path.is_relative_to(root / "output") or reports["save"].get("status") != "pass":
                        raise RuntimeError("Save escaped the owned output or failed")
                    media = audit_media(path, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, width=1024, height=512, count=73)
                    if media["file"]["sha256"] != reports["save"]["output_sha256"]:
                        raise RuntimeError("Media identity changed")
                    write_json(folder / "reports.json", reports)
                    write_json(folder / "media-audit.json", media)
                    row.update(status="media_validated", conditioning_sha256=reports["conditioning"]["conditioning_sha256"],
                        allocator_peak_bytes=allocator["qualified_interval_counters"]["allocated_peak_bytes"])
                else:
                    row.update(status="cpu_transport_only_no_GPU")
                write_json(folder / "result.json", row)
                result["rows"].append(row)
                print(json.dumps(row), flush=True)
                current = source_snapshot()
                current[str(Path(__file__).relative_to(PROJECT))] = file_identity(__file__)["sha256"]
                if current != sources or verify_core_source(CORE) != expected["core"]:
                    raise RuntimeError("Source changed during sequence")
            if args.mode == "gpu":
                write_json(root / "comparison.json", summarize(result["rows"], args.task))
            result["status"] = "six_job_cpu_transport_pass" if args.mode == "cpu-smoke" else "warm_pair_media_validated_human_pending"
        except BaseException as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if server:
                    server.stop()
                    result["server_stop"] = server.stop_receipt
                if monitor:
                    monitor.close()
                    result["resources"] = monitor.guard.report()
                    if monitor.failure:
                        result.update(status="failed", monitor_failure=monitor.failure)
            finally:
                if reader:
                    reader.__exit__()
                write_json(root / "terminal.json", result)
                print(json.dumps({"status": result["status"], "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
