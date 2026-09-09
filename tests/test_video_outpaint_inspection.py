from pathlib import Path

import psutil
import pytest

import h3_audio_t8_pkg.video_outpaint_inspection as client


@pytest.fixture
def fixture_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "WORKER_PATH", Path(__file__).with_name("outpaint_inspection_fault_worker.py"))
    source = tmp_path / "source.bin"
    source.write_bytes(b"do not change")
    return source


def test_worker_protocol_does_not_enable_gpu_or_change_input(fixture_worker):
    result = client.isolated_media_contract("inspect", fixture_worker)
    assert result == {"fixture": True, "cuda_visible_devices": "-1"}
    assert fixture_worker.read_bytes() == b"do not change"


@pytest.mark.parametrize("mode,exception,message", [
    ("exit", RuntimeError, "exited 55"),
    ("invalid", RuntimeError, "invalid JSON"),
    ("identity", RuntimeError, "identity mismatch"),
    ("value_error", ValueError, "fixture invalid media"),
])
def test_worker_failure_propagates_without_crashing_parent(fixture_worker, mode, exception, message):
    with pytest.raises(exception, match=message):
        client.isolated_media_contract("inspect", fixture_worker, parameters={"mode": mode})
    assert fixture_worker.read_bytes() == b"do not change"


@pytest.mark.parametrize("mode", ["large_output", "large_log"])
def test_worker_response_and_log_are_bounded(fixture_worker, monkeypatch, mode):
    monkeypatch.setattr(client, "MAX_JSON_BYTES", 1024)
    monkeypatch.setattr(client, "MAX_LOG_BYTES", 1024)
    with pytest.raises(RuntimeError, match="bounds"):
        client.isolated_media_contract("inspect", fixture_worker, parameters={"mode": mode})


def test_cancel_before_worker_start_does_not_spawn(fixture_worker, monkeypatch):
    def no_process(*args, **kwargs):
        raise AssertionError("cancelled request must not spawn")
    monkeypatch.setattr(client.subprocess, "Popen", no_process)
    def cancel():
        raise RuntimeError("user cancelled")
    with pytest.raises(RuntimeError, match="user cancelled"):
        client.isolated_media_contract("inspect", fixture_worker, interrupt_check=cancel)


def test_cancel_running_worker_is_reaped(fixture_worker, monkeypatch):
    real = client.subprocess.Popen
    workers = []
    def start(*args, **kwargs):
        process = real(*args, **kwargs)
        workers.append(process)
        return process
    monkeypatch.setattr(client.subprocess, "Popen", start)
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise RuntimeError("cancel running")
    with pytest.raises(RuntimeError, match="cancel running"):
        client.isolated_media_contract("inspect", fixture_worker, parameters={"mode": "sleep"}, interrupt_check=cancel)
    assert len(workers) == 1 and workers[0].poll() is not None


def test_worker_timeout_is_not_success(fixture_worker):
    with pytest.raises(TimeoutError, match="timed out"):
        client.isolated_media_contract("inspect", fixture_worker, parameters={"mode": "sleep"}, timeout=0.15)


def test_abrupt_worker_exit_reaps_observed_owned_descendant(fixture_worker):
    marker = fixture_worker.parent / "child.pid"
    with pytest.raises(RuntimeError, match="exited 55"):
        client.isolated_media_contract("inspect", fixture_worker,
            parameters={"mode": "orphan", "child_pid_file": str(marker)})
    pid = int(marker.read_text())
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    assert not process.is_running()


def test_real_isolated_contract_matches_original_validator(tmp_path):
    from test_video_outpaint_media import _clip
    from h3_audio_t8_pkg.dlss_nr_advanced import _file_source_contract as original
    from h3_audio_t8_pkg.video_outpaint_media import _file_source_contract as isolated
    source = _clip(tmp_path / "source.mp4", 64, 48, frames=5, sound=True)
    assert isolated(source) == original(source)


def test_real_worker_preserves_invalid_rate_rejection(tmp_path):
    from test_video_outpaint_media import _clip
    from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
    source = _clip(tmp_path / "source.mp4", 64, 48, frames=5, fps=30)
    with pytest.raises(ValueError, match="24fps"):
        inspect_outpaint_source(source)
