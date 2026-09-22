"""CPU-only contract tests for the owned fail-then-success PDD GPU probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_pdd_lifecycle_gpu_probe.py"


def _tool():
    name = "run_pdd_lifecycle_gpu_probe_test"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_failure_graph_changes_only_fallback_strength_and_first_sigma():
    tool = _tool()
    options = tool._parser().parse_args([])
    args = tool._pdd_args(options)
    bad, good = tool._graphs(args, "pdd-lifecycle")
    assert bad["8"]["inputs"]["strength"] == .99
    assert good["8"]["inputs"]["strength"] == 1.0
    assert bad["15"] == {"class_type": "SetFirstSigma", "inputs": {
        "sigmas": ["8", 2], "sigma": .999}}
    assert bad["11"]["inputs"]["sigmas"] == ["15", 0]
    assert good["11"]["inputs"]["sigmas"] == ["8", 2]
    assert bad["13"]["inputs"]["filename_prefix"].endswith("pdd-lifecycle-bad_fl2va_256x256_22f")
    assert good["13"]["inputs"]["filename_prefix"].endswith("pdd-lifecycle-good_fl2va_256x256_22f")


def test_only_expected_runtime_error_qualifies_first_phase():
    tool = _tool()
    with pytest.raises(RuntimeError, match="did not end with execution_error"):
        tool._assert_expected_failure({"terminal": {"type": "execution_success"}})
    with pytest.raises(RuntimeError, match="not the intended"):
        tool._assert_expected_failure({"terminal": {"type": "execution_error", "data": {
            "exception_message": "CUDA out of memory"}}})
    tool._assert_expected_failure({"terminal": {"type": "execution_error", "data": {
        "exception_message": "MiniMax-H3 PDD received a sigma outside its official 8-step Euler/simple grid"}}})


def test_bundled_ffmpeg_is_found_without_path_mutation(tmp_path, monkeypatch):
    tool = _tool()
    root = tmp_path / "ComfyUI"
    bundled = tmp_path / "ffmpeg" / "bin" / "ffmpeg.exe"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"bin")
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)
    assert tool._media_executable(root, "ffmpeg") == str(bundled)


def test_owned_same_process_dispatches_bad_then_good_and_stops(tmp_path, monkeypatch):
    tool = _tool()
    args = tool._pdd_args(tool._parser().parse_args([]))
    bad, good = tool._graphs(args, "pdd-lifecycle")
    called = []

    class Process:
        pid = 4242

        def poll(self):
            return None

    class Server:
        def __init__(self, _root, _port, _cpu):
            self.process = Process()
            self.url = "http://127.0.0.1:8861"
            self.stop_receipt = None

        def start(self):
            command = tool.pilot.server_command(tmp_path, 8861, False)
            assert command[0] == str(args.python)
            assert "ComfyUI-VideoHelperSuite" in command
            assert tool.PROJECT.name in command
            called.append("start")

        def assert_port_owner(self):
            called.append("owned")

        def stop(self):
            called.append("stop")
            self.stop_receipt = {"pid": 4242, "shutdown": "controller_owned_termination"}

    class Monitor:
        def __init__(self, interval_seconds):
            assert interval_seconds == .25

        def start(self):
            called.append("monitor-start")

        def stop(self):
            called.append("monitor-stop")
            return {"minimum_free_mib": 1000}

    async def submit(*, server, prompt, timeout_seconds):
        assert server == "http://127.0.0.1:8861" and timeout_seconds == 900
        called.append("bad" if prompt["8"]["inputs"]["strength"] < 1 else "good")
        if called[-1] == "bad":
            return {"terminal": {"type": "execution_error", "data": {
                "exception_message": tool.EXPECTED_FAILURE}}}
        return {"terminal": {"type": "execution_success"}}

    def finalize(*, args, run_root, phase, report, ffmpeg, ffprobe):
        assert args.variant == "FL2VA"
        assert run_root == tmp_path
        assert phase["terminal"]["type"] == "execution_success"
        assert report["run_id"] == "pdd-lifecycle-good"
        assert ffmpeg == "ffmpeg" and ffprobe == "ffprobe"
        called.append("media-checked")
        return 0

    monkeypatch.setattr(tool.pilot, "OwnedServer", Server)
    monkeypatch.setattr(tool.pilot, "wait_ready", lambda *_a, **_k: called.append("ready"))
    monkeypatch.setattr(tool.clipprobe, "GpuPeakMonitor", Monitor)
    monkeypatch.setattr(tool.pdd, "_submit_prompt_capture", submit)
    monkeypatch.setattr(tool.pdd, "_finalize_completed_run", finalize)
    preflight = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "ready": True}
    assert tool._run(args, tmp_path, bad, good, preflight) == 0
    assert called.index("bad") < called.index("good") < called.index("stop") < called.index("media-checked")
    report = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))
    assert report["same_process_recovery"] is True
    assert report["bad_phase_expected_error"] is True
    assert report["status"] == "PDD_LIFECYCLE_MECHANICAL_PASS_HUMAN_REVIEW_PENDING"
