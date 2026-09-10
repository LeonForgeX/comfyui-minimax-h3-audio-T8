"""Independent read-only audit of every completed exploration queue and resource row."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_progressive_warm_pair import queue_evidence, read  # noqa: E402
from progressive_probe_control import ResourceGuard  # noqa: E402
from progressive_pilot_analysis import load_run, compare_runs  # noqa: E402
from run_progressive_exploration import frozen_plan, validate_job  # noqa: E402


def audit_job(folder, item, suite_expected):
    run = load_run(folder)
    validate_job(run, item, suite_expected)
    env_graph, _, env_history = queue_evidence(folder / "environment")
    expected, live, terminal = run["environment"], run["live"], run["terminal"]
    if (json.loads(env_graph["1"]["inputs"]["expected_json"]) != expected or
            json.loads(env_history["outputs"]["2"]["text"][0]) != live or
            live["sources_checked"] != len(expected["sources"]) or expected["mode"] != "gpu" or
            expected.get("exploration_case") != item["exploration_case"] or
            expected["pilot_graphs"][item["case"]] != item["graph"] or expected["assets"] != run["assets"]):
        raise ValueError("Actual environment does not bind frozen exploration/source/assets")
    histories = [env_history]
    for action in ("begin", "generation", "finish"):
        name = "generation" if action == "generation" else "allocator-" + action
        graph, _, history = queue_evidence(folder / name)
        histories.append(history)
        if action != "generation" and graph["1"] != {"class_type": "T8ProgressiveAllocatorAudit", "inputs": {
                "run_id": folder.name, "action": action, "device_type": "cuda", "expected_pid": live["pid"]}}:
            raise ValueError("Allocator queue belongs to a different run/action/process")
    ordinals = [h["prompt"][0] for h in histories]
    if (len({h["prompt"][1] for h in histories}) != 4 or
            any(type(i) not in (int, float) or not math.isfinite(i) for i in ordinals) or
            any(a >= b for a, b in zip(ordinals, ordinals[1:]))):
        raise ValueError("Four-queue ordering or unique prompt evidence invalid")
    guard = ResourceGuard()
    startup = read(folder / "startup-resources.json")
    guard.observe(startup["sample"], startup=True)
    if guard.report() != startup["guard"] or guard.reason:
        raise ValueError("Startup resource evidence invalid")
    lines = (folder / "resources.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("No continuous resource observations")
    for line in lines:
        guard.observe(json.loads(line))
    if guard.reason or guard.report() != terminal["resource_guard"]:
        raise ValueError("Runtime resource summary differs from raw observations")
    return {"folder": folder.name, "seconds": run["timing"]["elapsed_seconds"],
        "media_sha256": run["media"]["file"]["sha256"]}, run


def audit(root, *, completed_only=False):
    root = Path(root).resolve(strict=True)
    expected, plan = read(root / "plan.json"), frozen_plan()
    if expected["jobs"] != plan:
        raise ValueError("Exploration plan differs from the five predeclared pairs")
    rows, runs, comparisons = [], [], []
    for item in plan:
        folder = root / item["folder"]
        if not (folder / "terminal.json").exists() and completed_only:
            break
        row, run = audit_job(folder, item, expected)
        rows.append(row)
        runs.append(run)
        if len(rows) % 2 == 0:
            comparison = compare_runs(runs[-2], runs[-1])
            path = root / (item["exploration_case"] + "-comparison.json")
            if path.exists():
                if read(path) != comparison:
                    raise ValueError("Stored pair summary differs from independently recomputed evidence")
            elif not completed_only:
                raise ValueError("Missing completed comparison")
            comparisons.append(comparison)
    if not completed_only:
        terminal = read(root / "terminal.json")
        if terminal["status"] != "ten_serial_jobs_five_matched_pairs_human_pending" or terminal["completed"] != rows:
            raise ValueError("Parent terminal does not match ten independent job receipts")
    return {"status": "completed_prefix_only_not_suite_qualification" if completed_only else "ten_jobs_independent_mechanical_audit_pass",
        "jobs": len(rows), "queues": len(rows)*4, "pairs": len(comparisons),
        "rows": rows, "human_quality": "pending", "GPU_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, completed_only=args.completed_only), ensure_ascii=False))
