"""Synthetic evidence fault tests; fixtures are not real GPU qualification."""
from copy import deepcopy
import json
import os

import pytest

from tools.audit_progressive_warm_pair import audit
from tools.progressive_memory_metrics import AllocatorMemorySession
from tools.progressive_probe_control import summarize_execution_events
from tools.run_progressive_warm_pair import sequence


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def change(path, mutate):
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    put(path, data)


def queue(folder, graph, ordinal, outputs):
    prompt_id = f"synthetic-prompt-{ordinal}"
    events = [{"type": "executing", "elapsed_seconds": (i+1)/100,
               "data": {"prompt_id": prompt_id, "node": node}} for i, node in enumerate(graph)]
    events.append({"type": "execution_success", "elapsed_seconds": .1, "data": {"prompt_id": prompt_id}})
    put(folder / "prompt.json", graph)
    put(folder / "submission.json", {"prompt_id": prompt_id})
    put(folder / "timing.json", summarize_execution_events(events, prompt_id, graph, .2))
    put(folder / "history.json", {"prompt": [ordinal, prompt_id, graph],
        "status": {"completed": True, "status_str": "success"}, "outputs": outputs})
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "synthetic-cpu-session"
    graph = {"18": {"inputs": {"filename_prefix": "unused"}}}
    expected = {"mode": "cpu-smoke", "sources": {"fake.py": "fixture"},
                "core": {"core_commit": "synthetic"}, "pilot_graphs": {"native8": graph, "progressive6plus2": graph}}
    live = {"status": "pass", "core": expected["core"], "pid": 123, "device": "cpu", "sources_checked": 1}
    put(root / "environment-expected.json", expected)
    put(root / "live-environment.json", live)
    queue(root / "environment", {"1": {"inputs": {"expected_json": json.dumps(expected)}}},
          0, {"2": {"text": [json.dumps(live)]}})
    plan = sequence("T2VA")
    put(root / "sequence.json", plan)
    rows = []
    for item in plan:
        folder = root / f"{item['index']:02d}_{item['case']}"
        probe = AllocatorMemorySession(run_id=folder.name, device_type="cpu")
        begin, end = probe.begin(), probe.finish()
        for action, payload, offset in (("begin", begin, 1), ("finish", end, 3)):
            control = {"1": {"class_type": "T8ProgressiveAllocatorAudit", "inputs": {
                "run_id": folder.name, "action": action, "device_type": "cpu", "expected_pid": 123}}}
            queue(folder / ("allocator-" + action), control, item["index"]*3+offset,
                  {"2": {"text": [json.dumps(payload)]}})
        generation = {"1": {"class_type": "PreviewAny", "inputs": {"source": f"CPU sequence {item['index']}"}}}
        queue(folder / "generation", generation, item["index"]*3+2, {})
        put(folder / "allocator-memory.json", end)
        row = {**item, "server_pid": 123, "seconds": .2, "cached": False, "status": "cpu_transport_only_no_GPU"}
        put(folder / "result.json", row)
        rows.append(row)
    put(root / "terminal.json", {"mode": "cpu-smoke", "status": "six_job_cpu_transport_pass", "rows": rows,
        "quality_qualified": False, "server_stop": {"pid": 123, "exit_code": 1, "owned_children_remaining": []}})
    return root


def test_cpu_evidence_is_verified_but_never_gpu_performance(session):
    report = audit(session)
    assert report["queues_verified"] == 19
    assert report["status"] == "six_job_CPU_transport_independently_verified"
    assert report["human_quality_qualified"] is False
    assert "comparison" not in report


@pytest.mark.parametrize("fault", ["failed", "partial", "monitor", "residual", "pid", "mode", "order",
                                  "elapsed", "history_prompt", "history_graph", "unfinished", "queue_order",
                                  "row", "allocator", "allocator_pid", "recipe", "cached"])
def test_forged_or_incomplete_evidence_rejected(session, fault):
    job = session / "02_T2VA_native8"
    if fault in {"failed", "partial", "monitor", "residual", "pid", "mode"}:
        def mutation(data):
            if fault == "failed":
                data["status"] = "failed"
            elif fault == "partial":
                data["rows"].pop()
            elif fault == "monitor":
                data["monitor_failure"] = "gap"
            elif fault == "residual":
                data["server_stop"]["owned_children_remaining"] = [123]
            elif fault == "pid":
                data["server_stop"]["pid"] = 456
            else:
                data["mode"] = "gpu"
        change(session / "terminal.json", mutation)
    elif fault == "order":
        change(session / "sequence.json", lambda d: d.reverse())
    elif fault == "elapsed":
        change(job / "generation/timing.json", lambda d: d.update(elapsed_seconds=.3))
    elif fault in {"history_prompt", "history_graph", "unfinished", "queue_order"}:
        def history_mutation(data):
            if fault == "history_prompt":
                data["prompt"][1] = "unrelated"
            elif fault == "history_graph":
                data["prompt"][2]["1"]["inputs"]["source"] = "changed"
            elif fault == "unfinished":
                data["status"]["completed"] = False
            else:
                data["prompt"][0] = 0
        change(job / "generation/history.json", history_mutation)
    elif fault == "row":
        change(job / "result.json", lambda d: d.update(seconds=999))
    elif fault == "allocator":
        change(job / "allocator-memory.json", lambda d: d.update(run_id="another"))
    elif fault == "allocator_pid":
        control = json.loads((job / "allocator-begin/prompt.json").read_text())
        control["1"]["inputs"]["expected_pid"] = 456
        put(job / "allocator-begin/prompt.json", control)
        change(job / "allocator-begin/history.json", lambda d: d["prompt"].__setitem__(2, control))
    elif fault == "recipe":
        data = json.loads((session / "environment-expected.json").read_text())
        data["pilot_graphs"]["native8"] = deepcopy(data["pilot_graphs"]["native8"])
        data["pilot_graphs"]["native8"]["99"] = {"class_type": "changed"}
        put(session / "environment-expected.json", data)
    else:
        path = job / "generation/events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events.insert(0, {"type": "execution_cached", "elapsed_seconds": 0,
                         "data": {"prompt_id": "synthetic-prompt-8", "nodes": ["1"]}})
        path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError):
        audit(session)


@pytest.fixture
def gpu_session(session):
    """Synthetic completed CUDA receipts, deliberately no real model or media decoding."""
    from test_progressive_memory_metrics import Reader
    from test_progressive_pilot_analysis import execution_values
    from tools.progressive_probe_control import ResourceGuard, file_identity
    from tools.run_progressive_warm_pair import summarize
    pid = os.getpid()
    expected = json.loads((session / "environment-expected.json").read_text())
    expected.update(mode="gpu", sources={"tools/progressive_memory_metrics.py": "synthetic"},
                    assets=[{"sha256": "synthetic-model"}])
    for route in ("native8", "progressive6plus2"):
        graph, _ = execution_values(route == "progressive6plus2")
        graph["18"] = {"inputs": {"filename_prefix": "unused"}}
        expected["pilot_graphs"][route] = graph
    live = {"status": "pass", "core": expected["core"], "pid": pid, "device": "cuda:0", "sources_checked": 1}
    put(session / "environment-expected.json", expected)
    put(session / "live-environment.json", live)
    put(session / "assets-verified.json", expected["assets"])
    queue(session / "environment", {"1": {"inputs": {"expected_json": json.dumps(expected)}}},
          0, {"2": {"text": [json.dumps(live)]}})
    mib = 1024**2
    sample = {"monotonic": 1, "gpu_uuid": "fixture-only", "gpu_total_bytes": 16000*mib,
              "gpu_used_bytes": 2000*mib, "gpu_free_bytes": 14000*mib,
              "ram_total_bytes": 64000*mib, "ram_available_bytes": 40000*mib}
    guard = ResourceGuard()
    guard.observe(sample, startup=True)
    put(session / "startup.json", sample)
    sample = {**sample, "monotonic": 2}
    guard.observe(sample)
    (session / "resources.jsonl").write_text(json.dumps(sample))
    rows = []
    for item in sequence("T2VA"):
        folder = session / f"{item['index']:02d}_{item['case']}"
        probe = AllocatorMemorySession(run_id=folder.name, device_type="cuda", device_index=0,
                                       reader=Reader("cudaMallocAsync"))
        begin, end = probe.begin(), probe.finish()
        for action, payload, offset in (("begin", begin, 1), ("finish", end, 3)):
            control = {"1": {"class_type": "T8ProgressiveAllocatorAudit", "inputs": {
                "run_id": folder.name, "action": action, "device_type": "cuda", "expected_pid": pid}}}
            queue(folder / ("allocator-" + action), control, item["index"]*3+offset,
                  {"2": {"text": [json.dumps(payload)]}})
        route = item["case"].removeprefix("T2VA_")
        graph = deepcopy(expected["pilot_graphs"][route])
        graph["18"]["inputs"]["filename_prefix"] = f"warm/{folder.name}"
        _, reports = execution_values(route == "progressive6plus2")
        path = session / "output" / f"{item['index']}.fixture"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic media identity only")
        identity = file_identity(path)
        reports.update(conditioning={"conditioning_sha256": "synthetic"}, decode={},
                       save={"output": str(path), "status": "pass", "output_sha256": identity["sha256"]})
        mapping = {"sampler": "21", "conditioning": "101", "decode": "102", "lora": "103", "save": "19"}
        queue(folder / "generation", graph, item["index"]*3+2,
              {node: {"text": [json.dumps(reports[name])]} for name, node in mapping.items()})
        put(folder / "allocator-memory.json", end)
        put(folder / "reports.json", reports)
        put(folder / "media-audit.json", {"file": identity, "strict_av_decode": True,
            "timeline": {"width": 1024, "height": 512, "frames": 73, "fps": "24", "all_video_pts_checked": True}})
        row = {**item, "server_pid": pid, "seconds": .2, "cached": False, "status": "media_validated",
               "conditioning_sha256": "synthetic",
               "allocator_peak_bytes": end["qualified_interval_counters"]["allocated_peak_bytes"]}
        put(folder / "result.json", row)
        rows.append(row)
    put(session / "terminal.json", {"mode": "gpu", "status": "warm_pair_media_validated_human_pending",
        "rows": rows, "quality_qualified": False, "resources": guard.report(),
        "server_stop": {"pid": pid, "exit_code": 1, "owned_children_remaining": []}})
    put(session / "comparison.json", summarize(rows, "T2VA"))
    return session


def test_synthetic_gpu_receipt_binding(gpu_session):
    result = audit(gpu_session)
    assert result["status"] == "warm_GPU_evidence_verified_human_pending"
    assert result["comparison"]["routes"]["native8"]["seconds"] == [.2, .2]
    assert result["human_quality_qualified"] is False


@pytest.mark.parametrize("fault", ["resource", "asset", "model_calls", "media", "frames", "peak", "summary"])
def test_synthetic_gpu_faults_rejected(gpu_session, fault):
    root = gpu_session
    job = root / "03_T2VA_progressive6plus2"
    if fault == "resource":
        change(root / "terminal.json", lambda d: d["resources"].update(samples=999))
    elif fault == "asset":
        put(root / "assets-verified.json", [])
    elif fault == "model_calls":
        change(job / "reports.json", lambda d: d["sampler"]["counts"]["actual_forwards"].update(high=1))
    elif fault == "media":
        (root / "output/3.fixture").write_bytes(b"different")
    elif fault == "frames":
        change(job / "media-audit.json", lambda d: d["timeline"].update(frames=72))
    elif fault == "peak":
        change(job / "result.json", lambda d: d.update(allocator_peak_bytes=0))
    else:
        change(root / "comparison.json", lambda d: d.update(saved_fraction_from_medians=.99))
    with pytest.raises(ValueError):
        audit(root)
