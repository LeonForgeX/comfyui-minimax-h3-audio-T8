"""Five frozen exploration pairs, ten strictly serial cold-process GPU runs.

Default plan is read-only. GPU mode never retries, changes seeds, opens UI or
touches published workflows. Each child independently owns its GPU lease/server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_pilot import (  # noqa: E402
    CORE, PROJECT, RESEARCH, EXPLORATION_CASES, exploration_recipe, instrument_recipe,
    source_snapshot, write_json,
)
from progressive_probe_control import SerialProbeLease  # noqa: E402
from progressive_pilot_analysis import load_run, compare_runs  # noqa: E402
from vdn_probe_environment import verify_core_source  # noqa: E402


def frozen_plan():
    jobs = []
    for exploration in EXPLORATION_CASES:
        for route in ("native8", "progressive6plus2"):
            case = "T2VA_" + route
            recipe = json.loads((RESEARCH / "pilot-api-drafts" / (case + ".prompt.json")).read_text(encoding="utf-8"))
            jobs.append({"exploration_case": exploration, "case": case,
                "folder": f"{len(jobs):02d}_{exploration}_{route}",
                "graph": instrument_recipe(exploration_recipe(recipe, case, exploration))})
    return jobs


def validate_job(run, item, expected):
    if (run["terminal"].get("exploration_case") != item["exploration_case"] or
            run["terminal"]["case"] != item["case"] or run["graph"] != item["graph"] or
            run["environment"]["sources"] != expected["sources"] or
            run["environment"]["core"] != expected["core"]):
        raise ValueError("Executed exploration differs from the frozen plan/source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "gpu"), default="plan")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--identities", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    plan = frozen_plan()
    if args.mode == "plan":
        print(json.dumps({"mode": "plan_only", "new_gpu_jobs": len(plan),
            "jobs": [{k: v for k, v in item.items() if k != "graph"} for item in plan]}))
        return
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError("Use a new dedicated exploration directory; no resume or retries")
    if not all((args.identities, args.ffmpeg, args.ffprobe)):
        raise ValueError("GPU requires explicit frozen assets and media binaries")
    with SerialProbeLease(RESEARCH / "exploration-controller.lock"):
        root.mkdir(parents=True)
        expected = {"core": verify_core_source(CORE), "sources": source_snapshot(), "jobs": plan}
        write_json(root / "plan.json", expected)
        terminal = {"status": "incomplete", "completed": [], "pairs": [], "human_review": "pending"}
        child = None
        try:
            for item in plan:
                if (root / "STOP").exists():
                    raise RuntimeError("Explicit stop requested between jobs; no further GPU queued")
                if verify_core_source(CORE) != expected["core"] or source_snapshot() != expected["sources"]:
                    raise RuntimeError("Source changed during the frozen exploration")
                target = root / item["folder"]
                command = [sys.executable, "-X", "utf8", str(PROJECT / "tools/run_progressive_pilot.py"),
                    "--mode", "gpu", "--run-root", str(target), "--case", item["case"],
                    "--exploration-case", item["exploration_case"], "--identities", str(args.identities.resolve()),
                    "--ffmpeg", str(args.ffmpeg.resolve()), "--ffprobe", str(args.ffprobe.resolve())]
                print("Starting serial exploration " + item["folder"], flush=True)
                # The child owns cleanup. No timeout-kill here that could orphan its server.
                child = subprocess.Popen(command, cwd=PROJECT)
                code = child.wait()
                if code:
                    raise RuntimeError(f"Exploration child exited {code}; no retry or next case")
                run = load_run(target)
                validate_job(run, item, expected)
                terminal["completed"].append({"folder": item["folder"], "seconds": run["timing"]["elapsed_seconds"],
                    "media_sha256": run["media"]["file"]["sha256"]})
                if item["case"].endswith("progressive6plus2"):
                    baseline = load_run(root / plan[len(terminal["completed"])-2]["folder"])
                    pair = compare_runs(baseline, run)
                    report = item["exploration_case"] + "-comparison.json"
                    write_json(root / report, pair)
                    terminal["pairs"].append(report)
                write_json(root / f"checkpoint-{len(terminal['completed']):02d}.json", terminal)
            terminal["status"] = "ten_serial_jobs_five_matched_pairs_human_pending"
        except BaseException as error:
            terminal.update(status="failed_or_interrupted", error=f"{type(error).__name__}: {error}")
            # Leave the already bounded child to its own watchdog/cleanup if the
            # parent alone is interrupted; explicitly report, never launch another.
            if child is not None and child.poll() is None:
                terminal["bounded_child_still_owns_cleanup_pid"] = child.pid
            raise
        finally:
            write_json(root / "terminal.json", terminal)
            print(json.dumps({"status": terminal["status"], "completed": len(terminal["completed"])}), flush=True)


if __name__ == "__main__":
    main()
