"""Temporary CPU-only native ComfyUI for workflow open/save/API QA.

Artifacts and UI saves stay inside a new research directory. No model inference is
queued; only the lightweight environment probe. A new queue during UI QA causes
the controller to stop its owned server. Maximum lifetime is bounded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_progressive_workflows import build_workflow  # noqa: E402
from progressive_pilot_analysis import load_run  # noqa: E402
from run_progressive_pilot import (  # noqa: E402
    CASES, CORE, PROJECT, RESEARCH, OwnedServer, execute_graph, instrument_recipe,
    preview_report, source_snapshot, wait_ready, write_json,
)
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8198)
    parser.add_argument("--hold-seconds", type=int, default=1200)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if root == RESEARCH or not root.is_relative_to(RESEARCH) or root.exists():
        raise ValueError("Use a new dedicated research QA directory")
    if not 30 <= args.hold_seconds <= 1800:
        raise ValueError("Use a bounded 30-1800 second QA lifetime")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid isolated QA port")
    expected = {"core": verify_core_source(CORE), "sources": source_snapshot(), "mode": "cpu-smoke",
                "pilot_graphs": {case: instrument_recipe(json.loads(
                    (RESEARCH / "pilot-api-drafts" / (case + ".prompt.json")).read_text(encoding="utf-8")))
                    for case in CASES}}
    # These are completed media-validated pilots, not evidence of UI parity.
    for name in ("gpu-t2va-progressive6plus2-v3", "gpu-i2va-progressive6plus2-v1"):
        load_run(RESEARCH / name)
    root.mkdir()
    server = OwnedServer(root, args.port, True)
    result = {"status": "incomplete", "gpu_allowed": False, "workflow_inference_queued": False}
    try:
        write_json(root / "paths.json", probe_resource_config(CORE, PROJECT))
        write_json(root / "source-before.json", expected)
        server.start()
        wait_ready(server, lambda: None)
        graph = {"1": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                 "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}}
        history, _ = execute_graph(server, graph, root / "environment", lambda: None, timeout=60)
        live = preview_report(history, "2")
        if live["pid"] != server.process.pid or live["device"] != "cpu" or live["cuda_initialized"]:
            raise RuntimeError("QA server is not the owned CPU-only runtime")
        write_json(root / "live-environment.json", live)
        info = server.request("GET", "/object_info")
        write_json(root / "object_info.json", info)
        target = root / "user/default/workflows/Progressive QA"
        target.mkdir(parents=True)
        for task in ("T2VA", "I2VA"):
            prompt = json.loads((RESEARCH / "pilot-api-drafts" / f"{task}_progressive6plus2.prompt.json").read_text(encoding="utf-8"))
            write_json(target / f"{task}_Progressive_EXP.json", build_workflow(prompt, info))
        write_json(root / "ready.json", {"status": "ready_cpu_only", "pid": server.process.pid,
            "url": server.url, "workflows": str(target), "stop_file": str(root / "STOP")})
        print("Ready for native workflow UI QA: " + server.url, flush=True)
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline and not (root / "STOP").exists():
            queue = server.request("GET", "/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise RuntimeError("Unexpected inference queued in CPU workflow QA")
            time.sleep(1)
        result["status"] = "cpu_ui_server_closed_not_automatic_roundtrip_acceptance"
        if verify_core_source(CORE) != expected["core"] or source_snapshot() != expected["sources"]:
            raise RuntimeError("Runtime source changed during workflow QA")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        server.stop()
        result["server_stop"] = server.stop_receipt
        write_json(root / "terminal.json", result)


if __name__ == "__main__":
    main()
