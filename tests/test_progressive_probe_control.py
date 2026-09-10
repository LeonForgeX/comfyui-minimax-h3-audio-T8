import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from tools.progressive_probe_control import (
    MIB, GuardPolicy, ResourceGuard, SerialProbeLease, file_identity, summarize_execution_events,
)


def sample(time=1., gpu=13000, ram=20000):
    return dict(monotonic=time, gpu_uuid="GPU-test", gpu_total_bytes=16380*MIB, gpu_used_bytes=(16380-gpu)*MIB,
                gpu_free_bytes=gpu*MIB, ram_total_bytes=32768*MIB, ram_available_bytes=ram*MIB)


def test_real_file_identity_and_changed_payload(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    assert file_identity(path)["sha256"] == hashlib.sha256(b"abc").hexdigest()
    path.write_bytes(b"def")
    assert file_identity(path)["sha256"] != hashlib.sha256(b"abc").hexdigest()


def test_os_lease_excludes_another_actual_cpu_process_and_releases(tmp_path):
    lock = tmp_path / "serial.lock"
    code = "from tools.progressive_probe_control import SerialProbeLease; import sys; lease=SerialProbeLease(sys.argv[1]); lease.__enter__(); lease.__exit__()"
    def attempt():
        return subprocess.run([sys.executable, "-c", code, str(lock)], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, timeout=10)
    with SerialProbeLease(lock):
        result = attempt()
        assert result.returncode != 0 and "owns the serial GPU lease" in result.stderr
    assert attempt().returncode == 0
    assert lock.exists()  # A leftover file is not proof that another task is running.


def test_startup_and_sticky_failure():
    guard = ResourceGuard()
    assert guard.observe(sample(gpu=3504), startup=True) == "startup_resource_margin_insufficient"
    assert guard.observe(sample(time=2)) == "startup_resource_margin_insufficient"


def test_sustained_low_margin_resets_on_recovery_and_then_stops():
    guard = ResourceGuard()
    assert guard.observe(sample(), startup=True) is None
    for time, gpu in [(2, 500), (3, 1000), (4, 500), (5, 500)]:
        assert guard.observe(sample(time, gpu)) is None
    assert guard.observe(sample(6, 500)) == "sustained_resource_margin_insufficient"
    assert guard.report()["minimum_gpu_free_bytes"] == 500*MIB


@pytest.mark.parametrize("values", [dict(gpu=100), dict(ram=500)])
def test_critical_margin_is_immediate(values):
    assert ResourceGuard().observe(sample(**values)) == "critical_resource_margin"


@pytest.mark.parametrize("change", [dict(monotonic=float("nan")), dict(gpu_free_bytes=-1),
                                     dict(gpu_uuid=None), dict(ram_available_bytes=True)])
def test_invalid_telemetry_cannot_pass(change):
    assert ResourceGuard().observe({**sample(), **change}) == "resource_telemetry_invalid_or_stale"


@pytest.mark.parametrize("change", [dict(monotonic=1.), dict(monotonic=5.), dict(gpu_uuid="other")])
def test_stale_or_wrong_device_telemetry_fails(change):
    guard = ResourceGuard()
    guard.observe(sample(), startup=True)
    assert guard.observe({**sample(time=2.), **change}) == "resource_telemetry_invalid_or_stale"


def test_guard_thresholds_are_validated():
    with pytest.raises(ValueError):
        GuardPolicy(consecutive_low_samples=0)
    assert ResourceGuard().report()["status"] == "failed"


def event(time, kind, **data):
    return {"elapsed_seconds": time, "type": kind, "data": {"prompt_id": "p", **data}}


def test_event_timing_counts_loading_decode_save_and_unassigned_wall():
    events = [event(1, "executing", node="load"), event(3, "executing", node="sample"),
              event(9, "executing", node="decode"), event(12, "executing", node="save"),
              event(14, "execution_success")]
    report = summarize_execution_events(events, "p", {"load", "sample", "decode", "save"}, 15)
    assert report["complete_uncached_graph"]
    assert [row["seconds"] for row in report["node_intervals"]] == [2, 6, 3, 2]
    assert report["unassigned_seconds"] == 2


def test_core_post_success_null_executing_notification_is_not_an_extra_execution():
    events = [event(1, "executing", node="a"), event(2, "execution_success"), event(2.1, "executing", node=None)]
    assert summarize_execution_events(events, "p", {"a"}, 3)["complete_uncached_graph"]


@pytest.mark.parametrize("end", ["execution_error", "execution_interrupted"])
def test_failed_prompt_never_qualifies_speed(end):
    report = summarize_execution_events([event(1, "executing", node="a"), event(2, end)], "p", {"a"}, 3)
    assert not report["complete_uncached_graph"]


def test_cache_hits_and_missing_or_foreign_nodes_are_not_success():
    events = [event(1, "execution_cached", nodes=["a"]), event(2, "execution_success")]
    report = summarize_execution_events(events, "p", {"a"}, 3)
    assert not report["complete_uncached_graph"] and report["graph_cached_nodes"] == ["a"]
    events = [event(1, "executing", node="a", prompt_id="other"), event(2, "execution_success")]
    assert summarize_execution_events(events, "p", {"a"}, 3)["missing_nodes"] == ["a"]


def test_reversed_or_repeated_event_times_are_explicit_errors():
    with pytest.raises(ValueError, match="Nonmonotonic"):
        summarize_execution_events([event(2, "executing", node="a"), event(1, "execution_success")], "p", {"a"}, 3)
    with pytest.raises(ValueError, match="Repeated"):
        summarize_execution_events([event(1, "executing", node="a"), event(2, "executing", node="a")], "p", {"a"}, 3)
    with pytest.raises(ValueError, match="after the terminal"):
        summarize_execution_events([event(1, "execution_success"), event(2, "executing", node="a")], "p", {"a"}, 3)


def test_pilot_assets_reject_paths_outside_declared_model_folders(tmp_path):
    from tools.prepare_progressive_pilot import recipe_assets
    with pytest.raises(ValueError, match="leaves its declared"):
        recipe_assets({"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "../../outside"}}}, tmp_path)


def test_pilot_common_graph_detects_model_changes_but_ignores_declared_sampler_routes():
    from copy import deepcopy
    from tools.prepare_progressive_pilot import comparison_common_graph
    a = {"4": {"inputs": {"unet_name": "same"}}, "10": {"class_type": "baseline"},
         "18": {"inputs": {"filename_prefix": "a", "crf": 18}}}
    b = deepcopy(a)
    b["10"]["class_type"] = "progressive"
    b["18"]["inputs"]["filename_prefix"] = "b"
    assert comparison_common_graph(a) == comparison_common_graph(b)
    b["4"]["inputs"]["unet_name"] = "changed"
    assert comparison_common_graph(a) != comparison_common_graph(b)


def test_nvml_reader_uses_high_resolution_monotonic_clock(monkeypatch):
    from types import SimpleNamespace
    import tools.progressive_probe_control as control
    def coarse_clock_must_not_be_used():
        raise AssertionError("Windows coarse monotonic clock can repeat within one tick")
    monkeypatch.setattr(control.time, "monotonic", coarse_clock_must_not_be_used)
    monkeypatch.setattr(control.time, "perf_counter", lambda: 123.456789)
    reader = control.NvmlResourceReader()
    reader.device, reader.uuid = object(), "GPU-test"
    reader.nvml = SimpleNamespace(nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(total=16000*MIB, used=1000*MIB, free=15000*MIB),
                                  nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=0))
    reader.psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64000*MIB, available=40000*MIB))
    row = reader.sample()
    assert row["monotonic"] == 123.456789 and row["clock"] == "perf_counter"
    assert ResourceGuard().observe(row, startup=True) is None
