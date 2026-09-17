"""Real owned Core interruption after segment0, then a fresh-process resume."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_semantic_bridge_workflow_probe as workflow  # noqa: E402
import run_fast_h3_v2_interrupt_probe as interrupt  # noqa: E402
from audit_semantic_bridge_loop import receipts  # noqa: E402

transport = workflow.transport


def start_phase_guard(reader, previous=None):
    """A stopped service has no live sampling interval; preserve GPU identity.

    Never restart a failed guard. A healthy previous phase may end before the
    next service starts, so that deliberate gap is not stale live telemetry.
    """
    if previous is not None and previous.reason:
        raise RuntimeError("Previous phase failed: " + previous.reason)
    guard = workflow.ResourceGuard()
    row = reader.sample()
    reason = guard.observe(row, startup=True)
    if reason:
        raise RuntimeError("Phase startup resource guard: " + reason)
    if previous is not None and guard.gpu_uuid != previous.gpu_uuid:
        raise RuntimeError("GPU identity changed between service phases")
    return guard


def finish_phase(stage, phase, server, monitor, guard, result):
    """Always stop the owned service and never mask the primary failure."""
    primary = sys.exc_info()[1]
    errors = []
    for action in ([monitor.close] if monitor else []) + [server.stop]:
        try:
            action()
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
    result["server_phases"].append(dict(phase=phase, stop=server.stop_receipt,
        resources=guard.report(), cleanup_errors=errors))
    try:
        # Exclusive writes are intentional; each phase has its own fresh path.
        transport.write_json(stage / "progress.json", result)
    except BaseException as error:
        errors.append(f"progress receipt {type(error).__name__}: {error}")
    if errors:
        result.setdefault("cleanup_errors", []).extend(errors)
        detail = "Owned phase cleanup failed: " + "; ".join(errors)
        if primary is None:
            raise RuntimeError(detail)
        if hasattr(primary, "add_note"):
            primary.add_note(detail)


def first_snapshot(output, chain_id, state):
    interrupt.validate_state(state, chain_id, "interrupted")
    root = interrupt.chain_root(output, chain_id)
    manifest = interrupt.read_json(root / "manifest.json")
    if (manifest.get("schema") != 2 or manifest.get("format") != "minimax_h3_t8_accepted_manifest"
            or manifest.get("chain_id") != chain_id or manifest.get("revision") != 1
            or len(manifest.get("segments", [])) != 1):
        raise ValueError("Expected one actually accepted native segment")
    entry = manifest["segments"][0]
    if entry.get("index") != 0:
        raise ValueError("Accepted segment index differs")
    files = {}
    for directory in (root / "candidates/segment_00000", root / "dual_stages/segment_00000"):
        for path in directory.rglob("*"):
            if path.is_file():
                checked = interrupt.inside(root, str(path))
                files[checked.relative_to(root).as_posix()] = {
                    **workflow.file_identity(checked), "mtime_ns": checked.stat().st_mtime_ns}
    for field in ("video_path", "context_path"):
        path = interrupt.inside(root, entry[field])
        identity = workflow.file_identity(path)
        if identity["sha256"] != entry[field.replace("_path", "_sha256")]:
            raise ValueError("Accepted first media/context hash mismatch")
        files[path.relative_to(root).as_posix()] = {**identity, "mtime_ns": path.stat().st_mtime_ns}
    if not files:
        raise ValueError("First segment has no persistent evidence")
    found = {}
    for name in files:
        if name.endswith(".json"):
            receipts(interrupt.read_json(root / name), found)
    if not found or any(":segment=0:" not in row["encoding_source"] for row in found.values()):
        raise ValueError("First segment lacks actual Bridge encoding receipts")
    return {"entry": entry, "files": files, "bridge_receipts": found,
            "contract_sha256": state["contract_sha256"]}


def verify_resume(output, chain_id, first, report):
    root = interrupt.chain_root(output, chain_id)
    manifest = interrupt.read_json(root / "manifest.json")
    if (report.get("status") != "complete" or report.get("accepted_count") != 2
            or report.get("segment_count") != 2 or len(manifest["segments"]) != 2
            or manifest["segments"][0] != first["entry"]
            or report.get("contract_sha256") != first["contract_sha256"]):
        raise ValueError("Resume changed first entry, contract, or failed to complete")
    for name, identity in first["files"].items():
        path = interrupt.inside(root, name)
        if (workflow.file_identity(path)["sha256"] != identity["sha256"]
                or path.stat().st_mtime_ns != identity["mtime_ns"]):
            raise ValueError("Resume rewrote an accepted first-segment asset")
    found = {}
    for path in (root / "candidates").rglob("*.json"):
        receipts(interrupt.read_json(path), found)
    first_count = len(first["bridge_receipts"])
    if len(found) != first_count * 2 or not set(first["bridge_receipts"]).issubset(found):
        raise ValueError("Expected retained first receipts plus independently encoded second segment")
    if sum(":segment=1:" in row["encoding_source"] for row in found.values()) != first_count:
        raise ValueError("Second segment Bridge receipt scope mismatch")
    return {"first_files_unchanged_including_mtime": True, "bridge_receipts": list(found.values()),
            "partial_second_attempt_forwards": "not_measured_do_not_infer_zero"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8221)
    args = parser.parse_args()
    project, root, core = Path(__file__).resolve().parents[1], args.root.resolve(), args.core.resolve()
    if root.exists() or root == project / "artifacts" or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Fresh dedicated task artifacts required")
    graph = json.loads(args.graph.read_text(encoding="utf8"))
    recipe, canvas = workflow.instrument(graph)
    if recipe != "loop" or canvas[2:] != [192, "24/1"]:
        raise ValueError("Only the exact two-segment8s candidate is allowed")
    chain_id = graph["8"]["inputs"]["chain_id"]
    transport.CORE, transport.PROJECT = core, project
    output = root / "output"
    sources = transport.source_snapshot()
    for name in ("run_semantic_bridge_interrupt_probe.py", "run_semantic_bridge_workflow_probe.py",
                 "run_fast_h3_v2_interrupt_probe.py", "audit_semantic_bridge_loop.py"):
        sources["tools/" + name] = workflow.file_identity(project / "tools" / name)["sha256"]
    assets = workflow.graph_assets(graph, core)
    expected = dict(core=workflow.verify_core_source(core), sources=sources, mode="gpu",
                    pilot_graphs={"bridge_loop": graph},
                    assets=[asset for asset in assets if "/semantic_bridge/" not in asset["path"].replace("\\", "/")])
    paths = workflow.probe_resource_config(core, project)
    paths["t8_runtime_models"]["semantic_bridge"] = str(core / "models/semantic_bridge")
    command = transport.server_command

    def shared_output_command(*values):
        value = command(*values)
        value[value.index("--output-directory") + 1] = str(output)
        return value

    transport.server_command = shared_output_command
    root.mkdir(parents=True)
    output.mkdir()
    transport.write_json(root / "assets.json", assets)
    transport.write_json(root / "expected.json", expected)
    result = dict(status="incomplete", human_qualified=False, recipe=recipe, server_phases=[])
    guard = None
    lease = core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with workflow.SerialProbeLease(lease), workflow.NvmlResourceReader() as reader:
            frozen, old_pid = None, None
            for phase in ("interrupt", "resume"):
                guard = start_phase_guard(reader, guard)
                stage = root / phase
                stage.mkdir()
                transport.write_json(stage / "paths.json", paths)
                server = transport.OwnedServer(stage, args.port, False, 0)
                monitor = None
                try:
                    server.start()
                    if old_pid == server.process.pid:
                        raise RuntimeError("Resume did not use a new owned process")
                    monitor = transport.ContinuousGuard(reader, guard, stage / "resources.jsonl", server)
                    monitor.start()
                    transport.wait_ready(server, monitor.check)
                    env, _ = transport.execute_graph(server, {
                        "98": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                        "99": {"class_type": "PreviewAny", "inputs": {"source": ["98", 0]}}},
                        stage / "environment", monitor.check)
                    transport.write_json(stage / "environment-report.json", transport.preview_report(env, "99"))
                    if phase == "interrupt":
                        watcher = interrupt.InterruptWatcher(server, stage / "generation", output, chain_id)
                        def check():
                            monitor.check()
                            watcher.check()
                        try:
                            transport.execute_graph(server, deepcopy(graph), stage / "generation", check, timeout=3600)
                        except RuntimeError as error:
                            if str(error) != "Graph failed, was cached, or did not execute all instrumented nodes":
                                raise
                        else:
                            raise RuntimeError("No actual interruption occurred")
                        state = interrupt.read_json(interrupt.chain_root(output, chain_id) / interrupt.STATE_NAME)
                        interrupt.validate_trigger(interrupt.read_json(stage / "native-trigger-state.json"), state,
                                                   interrupt.read_json(stage / "interrupt-request.json"), chain_id)
                        interrupt.require_interrupted_history(interrupt.read_json(stage / "generation/history.json"),
                            [json.loads(line) for line in (stage / "generation/events.jsonl").read_text(encoding="utf8").splitlines()],
                            interrupt.read_json(stage / "generation/submission.json"),
                            interrupt.read_json(stage / "interrupt-request.json"))
                        frozen = first_snapshot(output, chain_id, state)
                        transport.write_json(stage / "first-segment-evidence.json", frozen)
                        transport.write_json(stage / "native-interrupted-state.json", state)
                        old_pid = server.process.pid
                    else:
                        history, timing = transport.execute_graph(server, deepcopy(graph), stage / "generation",
                                                                  monitor.check, timeout=3600)
                        report = transport.preview_report(history, "90")
                        result["resume"] = verify_resume(output, chain_id, frozen, report)
                        result["media"] = workflow.audit_video(report["final_video_path"], output, canvas)
                        result["report"], result["timing"] = report, timing
                finally:
                    finish_phase(stage, phase, server, monitor, guard, result)
            for name, sha in sources.items():
                if workflow.file_identity(project / name)["sha256"] != sha:
                    raise RuntimeError("Runtime/controller source changed")
            for asset in assets:
                if workflow.file_identity(Path(asset["path"]))["sha256"] != asset["sha256"]:
                    raise RuntimeError("Model changed")
            result["status"] = "native_interruption_new_process_resume_pass_human_pending"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        transport.server_command = command
        result["resources"] = guard.report() if guard else {"status": "not_started"}
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"root": str(root), "status": result["status"]}))


if __name__ == "__main__":
    main()
