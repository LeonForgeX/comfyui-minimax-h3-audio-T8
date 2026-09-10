"""Read-only postflight of a recorded six-job warm session; never starts Core/GPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_warm_pair import sequence, summarize  # noqa: E402
from progressive_pilot_analysis import (  # noqa: E402
    common_graph, load_allocator_evidence, verify_execution_report,
)
from progressive_probe_control import ResourceGuard, file_identity, summarize_execution_events  # noqa: E402
from progressive_memory_metrics import validate_completed_interval  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def queue_evidence(folder):
    """Bind timing to the submitted graph, actual history and uncached events."""
    graph, timing, history = [read(folder / name) for name in ("prompt.json", "timing.json", "history.json")]
    prompt_id = read(folder / "submission.json")["prompt_id"]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    recomputed = summarize_execution_events(events, prompt_id, graph, timing["elapsed_seconds"])
    if (recomputed != timing or not timing["complete_uncached_graph"] or
            history["status"].get("completed") is not True or
            history["status"].get("status_str") != "success" or
            history["prompt"][1] != prompt_id or history["prompt"][2] != graph):
        raise ValueError("Queue history, submitted graph and uncached timing disagree")
    return graph, timing, history


def audit(root):
    root = Path(root).resolve(strict=True)
    terminal, expected, live = [read(root / name) for name in
                                ("terminal.json", "environment-expected.json", "live-environment.json")]
    mode = terminal.get("mode")
    wanted = {"cpu-smoke": "six_job_cpu_transport_pass", "gpu": "warm_pair_media_validated_human_pending"}
    stop = terminal.get("server_stop", {})
    if (mode not in wanted or terminal.get("status") != wanted[mode] or terminal.get("monitor_failure") or
            terminal.get("quality_qualified") is not False or stop.get("owned_children_remaining") != [] or
            type(stop.get("exit_code")) is not int or type(stop.get("pid")) is not int or stop["pid"] <= 0):
        raise ValueError("Not a completed owned warm session; a summary alone cannot qualify it")
    env_graph, _, env_history = queue_evidence(root / "environment")
    if (json.loads(env_history["outputs"]["2"]["text"][0]) != live or
            json.loads(env_graph["1"]["inputs"]["expected_json"]) != expected or
            expected["mode"] != mode or live["status"] != "pass" or live["pid"] != stop["pid"] or
            live["core"]["core_commit"] != expected["core"]["core_commit"] or
            live["sources_checked"] != len(expected["sources"]) or
            (live["device"] == "cpu") != (mode == "cpu-smoke")):
        raise ValueError("Live environment is not bound to the expected sources and owned process")
    plan = read(root / "sequence.json")
    task = plan[0]["case"].split("_", 1)[0]
    if plan != sequence(task) or len(terminal["rows"]) != 6:
        raise ValueError("Incomplete or changed six-job sequence")
    graphs = expected["pilot_graphs"]
    if common_graph(graphs["native8"]) != common_graph(graphs["progressive6plus2"]):
        raise ValueError("Routes do not share the same fixed recipe")
    if mode == "gpu":
        if "tools/progressive_memory_metrics.py" not in {key.replace("\\", "/") for key in expected["sources"]}:
            raise ValueError("Warm GPU measurement requires the allocator implementation identity")
        guard = ResourceGuard()
        guard.observe(read(root / "startup.json"), startup=True)
        samples = (root / "resources.jsonl").read_text(encoding="utf-8").splitlines()
        if not samples:
            raise ValueError("No runtime resource observations")
        for line in samples:
            guard.observe(json.loads(line))
        if guard.report() != terminal.get("resources") or guard.reason:
            raise ValueError("Resource receipt does not match actual observations")
        if not expected.get("assets") or read(root / "assets-verified.json") != expected["assets"]:
            raise ValueError("Model/input identity evidence missing or changed")
    ordinals, prompt_ids, rows, condition = [env_history["prompt"][0]], {env_history["prompt"][1]}, [], None
    for item in plan:
        folder = root / f"{item['index']:02d}_{item['case']}"
        begin_graph, _, begin = queue_evidence(folder / "allocator-begin")
        graph, timing, history = queue_evidence(folder / "generation")
        end_graph, _, end = queue_evidence(folder / "allocator-finish")
        for hist in (begin, history, end):
            if hist["prompt"][1] in prompt_ids:
                raise ValueError("Reused prompt receipt across jobs")
            prompt_ids.add(hist["prompt"][1])
            ordinals.append(hist["prompt"][0])
        device = "cpu" if mode == "cpu-smoke" else "cuda"
        for action, control in (("begin", begin_graph), ("finish", end_graph)):
            if control["1"] != {"class_type": "T8ProgressiveAllocatorAudit", "inputs": {
                    "run_id": folder.name, "action": action, "device_type": device, "expected_pid": live["pid"]}}:
                raise ValueError("Allocator action/run/process binding changed")
        allocator = read(folder / "allocator-memory.json")
        if json.loads(end["outputs"]["2"]["text"][0]) != allocator:
            raise ValueError("Allocator report differs from completed history")
        row = {**item, "server_pid": live["pid"], "seconds": timing["elapsed_seconds"], "cached": False}
        if mode == "cpu-smoke":
            if graph != {"1": {"class_type": "PreviewAny", "inputs": {"source": f"CPU sequence {item['index']}"}}}:
                raise ValueError("CPU transport graph changed")
            validate_completed_interval(allocator, run_id=folder.name, pid=live["pid"], device_type="cpu")
            row["status"] = "cpu_transport_only_no_GPU"
        else:
            route = item["case"].removeprefix(task + "_")
            wanted_graph = deepcopy(graphs[route])
            wanted_graph["18"]["inputs"]["filename_prefix"] = f"warm/{folder.name}"
            if graph != wanted_graph:
                raise ValueError("Executed GPU graph differs from the frozen recipe")
            # Reuse the same strict begin/end validator without inventing on-disk cold receipts.
            binding = {"resource_guard": terminal["resources"], "allocator_interval": {
                "file": "allocator-memory.json", "status": allocator["status"], "counter_scope": allocator["counter_scope"]}}
            load_allocator_evidence(folder, binding, expected, live)
            reports = read(folder / "reports.json")
            for name, node in {"sampler": "21", "conditioning": "101", "decode": "102", "lora": "103", "save": "19"}.items():
                if json.loads(history["outputs"][node]["text"][0]) != reports[name]:
                    raise ValueError("Saved report differs from actual GPU history")
            verify_execution_report(item["case"], graph, reports)
            if condition is None:
                condition = reports["conditioning"]
            if condition != reports["conditioning"]:
                raise ValueError("Actual conditioning differs across warm samples")
            media = read(folder / "media-audit.json")
            output = Path(reports["save"]["output"]).resolve(strict=True)
            if (not output.is_relative_to(root / "output") or file_identity(output) != media["file"] or
                    reports["save"].get("status") != "pass" or
                    reports["save"]["output_sha256"] != media["file"]["sha256"] or
                    media.get("strict_av_decode") is not True or
                    any(media["timeline"].get(k) != v for k, v in
                        {"width": 1024, "height": 512, "frames": 73, "fps": "24", "all_video_pts_checked": True}.items())):
                raise ValueError("Media evidence is stale or outside the fixed pilot")
            row.update(status="media_validated", conditioning_sha256=condition["conditioning_sha256"],
                       allocator_peak_bytes=allocator["qualified_interval_counters"]["allocated_peak_bytes"])
        if row != read(folder / "result.json") or row != terminal["rows"][item["index"]]:
            raise ValueError("Result row differs from independently reconstructed evidence")
        rows.append(row)
    if any(type(n) is not int for n in ordinals) or any(a >= b for a, b in zip(ordinals, ordinals[1:])):
        raise ValueError("Queue ordering is not strictly serial")
    result = {"status": "six_job_CPU_transport_independently_verified", "mode": mode,
              "queues_verified": len(ordinals), "server_pid": live["pid"], "human_quality_qualified": False,
              "scope": "Historical recorded execution; no claim current source or installed assets are unchanged."}
    if mode == "gpu":
        comparison = summarize(rows, task)
        if comparison != read(root / "comparison.json"):
            raise ValueError("Saved comparison differs from reconstructed measurements")
        result.update(status="warm_GPU_evidence_verified_human_pending", comparison=comparison)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), ensure_ascii=False, indent=2, allow_nan=False))
