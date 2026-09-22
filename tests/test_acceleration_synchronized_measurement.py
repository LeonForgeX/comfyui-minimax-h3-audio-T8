from copy import deepcopy

import pytest

from h3_audio_t8_pkg.acceleration_measurement import (
    SynchronizedCudaMeasurement, compare_equal_workload, summarize_repeated_measurements,
)
from test_acceleration_measurement import Clock, workload, measured


class FakeCuda:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def synchronize(self, device):
        self.calls.append(("synchronize", device))

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset", device))

    def memory_allocated(self, device):
        return 12

    def memory_reserved(self, device):
        return 24

    def max_memory_allocated(self, device):
        return 18

    def max_memory_reserved(self, device):
        return 30


def identity():
    return {key: "test-only" for key in ("gpu_uuid", "gpu_name", "torch_version", "cuda_version", "driver_version", "attention_backend")}


def test_sync_stages_and_peak_reset_are_explicit_and_once():
    clock, cuda = Clock(), FakeCuda()
    record = SynchronizedCudaMeasurement(workload(), cache_state="cold", runtime_identity=identity(),
                                        exclusive_process=True, device=2, clock=clock, cuda_backend=cuda)
    with record.stage("model_loading"):
        clock.value = 2
    with record.stage("sampling"):
        clock.value = 5
    report = record.finish()
    assert cuda.calls.count(("reset", 2)) == 1
    assert cuda.calls.count(("synchronize", 2)) == 6
    assert report["elapsed_seconds"] == 5
    assert report["resource_observations"][-1]["torch_peak_allocated_bytes"] == 18
    assert "not_whole_device_or_host_ram" in report["resource_scope"]
    assert report["measurement_runtime"]["device_index"] == 2


def test_shared_process_cannot_reset_peaks():
    cuda = FakeCuda()
    with pytest.raises(ValueError, match="exclusive_process"):
        SynchronizedCudaMeasurement(workload(), cache_state="warm", runtime_identity=identity(), cuda_backend=cuda)
    assert cuda.calls == []


def test_failure_preserved_when_cleanup_sync_fails():
    clock, cuda = Clock(), FakeCuda()
    record = SynchronizedCudaMeasurement(workload(), cache_state="warm", runtime_identity=identity(),
                                        exclusive_process=True, clock=clock, cuda_backend=cuda)
    with pytest.raises(ValueError, match="original"):
        with record.stage("sampling"):
            clock.value = 1
            cuda.synchronize = lambda device: (_ for _ in ()).throw(RuntimeError("sync"))
            raise ValueError("original")
    report = record.finish()
    assert report["status"] == "failed"
    assert report["instrumentation_errors"]


def test_runtime_and_instrumentation_mismatch_cannot_claim_speedup():
    first, other = measured(), measured()
    first["measurement_runtime"] = identity()
    with pytest.raises(ValueError, match="runtime mismatch"):
        compare_equal_workload(first, other)


def test_baseline_median_keeps_cold_warm_separate():
    reports = [measured(10), measured(12), measured(8)]
    summary = summarize_repeated_measurements(reports)
    assert summary["median_seconds"] == 10
    assert summary["stages"] == [{"name": "sampling", "median_seconds": 10}]
    assert summary["speedup_claim"] is False
    wrong = deepcopy(reports)
    wrong[-1]["cache_state"] = "cold"
    with pytest.raises(ValueError, match="cold with warm"):
        summarize_repeated_measurements(wrong)


def test_baseline_rejects_short_or_mismatched_stage_runs():
    with pytest.raises(ValueError, match="Not enough"):
        summarize_repeated_measurements([measured()])
    reports = [measured(), measured(), measured()]
    reports[-1]["stages"][0]["name"] = "different"
    with pytest.raises(ValueError, match="Stage sequence"):
        summarize_repeated_measurements(reports)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_baseline_rejects_forged_stage_duration(bad):
    reports = [measured(), measured(), measured()]
    reports[-1]["stages"][0]["seconds"] = bad
    with pytest.raises(ValueError, match="stage record"):
        summarize_repeated_measurements(reports)
