from copy import deepcopy
import json
import subprocess
import sys
import threading

import pytest

from tools.run_progressive_pilot import (
    CASES, CORE, PROJECT, RESEARCH, ContinuousGuard, OwnedServer, instrument_recipe, preview_report, server_command,
)
from tools.progressive_probe_control import ResourceGuard


@pytest.mark.parametrize("case", CASES)
def test_instrumented_routes_preserve_fixed_inputs_and_actual_baseline_semantics(case):
    source = json.loads((RESEARCH / "pilot-api-drafts" / (case + ".prompt.json")).read_text(encoding="utf-8"))
    before = deepcopy(source)
    graph = instrument_recipe(source)
    assert source == before
    for node in ("1", "2", "3", "4", "5", "6", "12", "18", "90", "91"):
        assert graph[node] == source[node]
    assert graph["7"]["inputs"]["av_latent"] == ["100", 1]
    assert graph["100"]["inputs"] == {"positive": ["6", 0], "av_latent": ["6", 1]}
    assert graph["103"]["inputs"]["source"] == ["104", 1]
    assert graph["7"]["inputs"]["model"] == ["104", 0]
    assert graph["11"]["class_type"] == "T8ProgressiveTimedDecode"
    if case.endswith("native8"):
        assert graph["10"]["class_type"] == "T8ProgressiveNativeBaseline"
        assert graph["10"]["inputs"]["seed"] == source["8"]["inputs"]["noise_seed"]
        assert not {"8", "9"}.intersection(graph)
    else:
        assert graph["10"]["inputs"]["upscaler_model"] == source["10"]["inputs"]["upscaler_model"]
        assert graph["13"]["inputs"]["conditioning"] == ["100", 0]
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in graph


def test_server_command_is_scoped_cpu_has_no_gpu_or_extra_plugins(tmp_path):
    command = server_command(tmp_path, 8197, True)
    assert command[:4] == [sys.executable, "-X", "utf8", str(CORE / "main.py")]
    assert "--cpu" in command and "--cache-none" in command
    i = command.index("--whitelist-custom-nodes")
    assert command[i+1:i+3] == [PROJECT.name, "progressive_probe_extension"]
    assert "--use-sage-attention" not in command
    assert str(tmp_path / "output") in command


def test_live_owned_process_stop_is_idempotent_and_does_not_require_a_port(tmp_path):
    server = OwnedServer(tmp_path, 8197, True)
    server.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                      creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    try:
        server.stop()
        assert server.process.poll() is not None
        assert not server.stop_receipt["owned_children_remaining"]
        first = server.stop_receipt
        server.stop()
        assert server.stop_receipt is first
    finally:
        if server.process.poll() is None:
            server.process.kill()
            server.process.wait(timeout=5)


def test_monitor_telemetry_failure_stops_only_supplied_owner(tmp_path):
    stopped = threading.Event()
    class Reader:
        def sample(self):
            raise RuntimeError("NVML disconnected")
    class Owner:
        def stop(self):
            stopped.set()
    monitor = ContinuousGuard(Reader(), ResourceGuard(), tmp_path / "resources.jsonl", Owner())
    monitor.start()
    try:
        assert stopped.wait(3)
        with pytest.raises(RuntimeError, match="NVML disconnected"):
            monitor.check()
    finally:
        monitor.close()


def test_history_preview_report_is_parsed_strictly():
    assert preview_report({"outputs": {"2": {"text": ['{"status":"pass"}']}}}, "2")["status"] == "pass"
    with pytest.raises(RuntimeError, match="structure"):
        preview_report({"outputs": {"2": {"text": []}}}, "2")


def test_allocator_control_graph_binds_owned_pid_run_and_separate_evidence(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import tools.run_progressive_pilot as runner
    calls = []
    def execute(server, graph, folder, check, timeout):
        calls.append((graph, folder, timeout))
        return {"outputs": {"2": {"text": ['{"status":"unavailable"}']}}}, {}
    monkeypatch.setattr(runner, "execute_graph", execute)
    owner = SimpleNamespace(process=SimpleNamespace(pid=123))
    assert runner.allocator_action(owner, tmp_path, "begin", "cpu", lambda: None)["status"] == "unavailable"
    graph, folder, timeout = calls[0]
    assert graph["1"]["inputs"] == {"run_id": tmp_path.name, "action": "begin", "device_type": "cpu", "expected_pid": 123}
    assert folder == tmp_path / "allocator-begin" and timeout == 60


def test_allocator_implementation_is_included_in_runtime_source_freeze():
    from tools.run_progressive_pilot import source_snapshot
    assert "tools/progressive_memory_metrics.py" in {key.replace('\\', '/') for key in source_snapshot()}
