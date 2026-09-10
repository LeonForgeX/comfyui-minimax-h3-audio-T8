from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.progressive_memory_metrics import (
    AllocatorMemorySession, COUNTERS, TorchAllocatorReader, counter_scope, validated_counters, validate_completed_interval,
)


def counters(used=128, used_peak=512, reserved=1024, reserved_peak=2048):
    return dict(zip(COUNTERS.values(), (used, used_peak, reserved, reserved_peak)))


class Reader:
    def __init__(self, backend="native", version="2.10.0+cu130"):
        self.metadata = {"pid": os.getpid(), "device_index": 0, "gpu_uuid": "fixture-only",
            "backend": backend, "torch_version": version, "cuda_version": "13.0"}
        self.stats = counters()
        self.resets = 0

    def identity(self):
        return deepcopy(self.metadata)

    def read(self):
        return deepcopy(self.stats)

    def reset_peaks(self):
        self.resets += 1
        for kind in ("allocated", "reserved"):
            self.stats[f"{kind}_bytes.all.peak"] = self.stats[f"{kind}_bytes.all.current"]


def session(reader=None, **kwargs):
    return AllocatorMemorySession(run_id="fixture-not-GPU-evidence", device_type="cuda",
        device_index=0, reader=reader or Reader(), **kwargs)


@pytest.mark.parametrize("backend", ["native", "cudaMallocAsync"])
def test_one_reset_preserves_peak_between_boundaries_and_does_not_claim_stage_peaks(backend):
    reader = Reader(backend)
    probe = session(reader)
    initial = probe.begin()
    assert initial["pre_interval_counters"]["allocated_peak_bytes"] == 512
    assert initial["snapshots"][0]["allocated_peak_bytes"] == 128
    reader.stats = counters(used=300, used_peak=800, reserved=1024, reserved_peak=2048)
    probe.observe("after_sampling")
    reader.stats = counters(used=100, used_peak=800, reserved=1024, reserved_peak=2048)
    result = probe.finish()
    assert reader.resets == result["reset_count"] == 1
    assert result["qualified_interval_counters"]["allocated_peak_bytes"] == 800
    assert result["complete_interval_observed"]
    assert result["stage_peak_attribution"] is False
    assert result["whole_device_memory"] is False
    assert result["forced_cuda_synchronization"] is False
    assert result["cleared_allocator_cache"] is False
    assert ("default_pool" in result["counter_scope"]) == (backend == "cudaMallocAsync")
    json.dumps(result, allow_nan=False)
    result["snapshots"].clear()
    assert len(probe.report()["snapshots"]) == 3  # No mutable report alias.


def test_cpu_path_never_touches_injected_cuda_reader_and_uses_null_not_zero():
    class Bomb:
        def __getattr__(self, name):
            raise AssertionError("CPU must not access CUDA")
    probe = AllocatorMemorySession(run_id="cpu", device_type="cpu", reader=Bomb())
    assert probe.begin()["status"] == "unavailable"
    report = probe.finish()
    assert report["qualified_interval_counters"] is None and report["snapshots"] == []
    assert report["reset_count"] == 0 and not report["complete_interval_observed"]
    with pytest.raises(RuntimeError):
        probe.finish()


@pytest.mark.parametrize("key", list(COUNTERS.values()))
@pytest.mark.parametrize("bad", [None, True, -1, 1.0, float("nan"), "0"])
def test_unavailable_or_malformed_stats_cannot_become_zero(key, bad):
    raw = counters()
    raw[key] = bad
    with pytest.raises(ValueError):
        validated_counters(raw)


@pytest.mark.parametrize("values", [counters(800, 700), counters(128, 512, 2048, 1024),
                                    counters(2048, 2048, 1024, 2048), counters(128, 4096)])
def test_inconsistent_counters_are_not_qualified(values):
    with pytest.raises(ValueError):
        validated_counters(values)


def test_only_known_async_semantics_are_qualified_without_switching_allocator():
    for backend, version in [("unknown", "2.10.0"), ("cudaMallocAsync", "2.9.0"),
                             ("cudaMallocAsync", "2.11.0.dev20260101")]:
        reader = Reader(backend, version)
        probe = session(reader)
        with pytest.raises(ValueError, match="semantics unqualified"):
            probe.begin()
        assert reader.resets == 0 and probe.report()["status"] == "failed"
    assert "native" in counter_scope("native", "2.10.0+cu130")


@pytest.mark.parametrize("mutation", [{"pid": -1}, {"device_index": 1},
                                      {"backend": "cudaMallocAsync"}, {"gpu_uuid": "other"}])
def test_identity_changes_fail_and_preserve_prior_evidence(mutation):
    reader = Reader()
    probe = session(reader)
    probe.begin()
    reader.metadata.update(mutation)
    with pytest.raises(RuntimeError, match="identity changed"):
        probe.observe("after_sampling")
    report = probe.report()
    assert len(report["snapshots"]) == 1
    assert report["qualified_interval_counters"] is None and report["status"] == "failed"


def test_wrong_initial_process_and_device_rejected_before_reset():
    for field in ("pid", "device_index"):
        reader = Reader()
        reader.metadata[field] = -1
        with pytest.raises(RuntimeError, match="owned process/device"):
            session(reader).begin()
        assert reader.resets == 0


def test_peak_reset_by_another_component_fails_closed():
    reader = Reader()
    probe = session(reader)
    probe.begin()
    reader.stats = counters()
    probe.observe("sampling")
    reader.reset_peaks()
    with pytest.raises(RuntimeError, match="peaks decreased"):
        probe.finish()
    assert not probe.report()["complete_interval_observed"]


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_incomplete_generation_never_gets_qualified_peak(outcome):
    probe = session()
    probe.begin()
    result = probe.finish(outcome=outcome)
    assert result["outcome"] == outcome and result["qualified_interval_counters"] is None
    assert len(result["snapshots"]) == 2


def test_read_failure_is_sticky_not_silently_reported_as_completion():
    reader = Reader()
    probe = session(reader)
    probe.begin()
    reader.stats = {}
    with pytest.raises(ValueError):
        probe.observe("decode")
    assert probe.report()["error"]
    reader.stats = counters()
    with pytest.raises(RuntimeError):
        probe.finish()


@pytest.mark.parametrize("ticks", [[float("nan")], [True], [2., 1.]])
def test_invalid_clock_fails_closed(ticks):
    probe = session(clock=iter(ticks).__next__)
    with pytest.raises(ValueError):
        probe.begin()
    assert probe.report()["status"] == "failed"


def test_no_duplicate_begin_or_finish_and_observe_before_begin():
    probe = session()
    with pytest.raises(RuntimeError):
        probe.observe("too_early")
    probe.begin()
    with pytest.raises(RuntimeError):
        probe.begin()
    probe.finish()
    with pytest.raises(RuntimeError):
        probe.finish()


@pytest.mark.parametrize("bad", [None, -1, True, 0.0])
def test_cuda_device_is_never_implicitly_selected(bad):
    with pytest.raises(ValueError):
        TorchAllocatorReader(bad, torch_module=object())


def test_torch_reader_uses_only_selected_device_and_documented_four_counters():
    calls = []
    cuda = SimpleNamespace(is_initialized=lambda: True, current_device=lambda: 2,
        get_device_properties=lambda i: SimpleNamespace(uuid=f"GPU-{i}"),
        get_allocator_backend=lambda: "cudaMallocAsync",
        memory_stats=lambda i: calls.append(("read", i)) or counters(),
        reset_peak_memory_stats=lambda i: calls.append(("reset", i)))
    torch = SimpleNamespace(cuda=cuda, __version__="2.10.0+cu130", version=SimpleNamespace(cuda="13.0"))
    reader = TorchAllocatorReader(2, torch)
    assert reader.identity()["gpu_uuid"] == "GPU-2"
    assert reader.read() == counters()
    reader.reset_peaks()
    assert calls == [("read", 2), ("reset", 2)]
    cuda.current_device = lambda: 1
    with pytest.raises(RuntimeError, match="Current CUDA device"):
        reader.read()


def test_real_cpu_process_imports_without_initializing_cuda():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from tools.progressive_memory_metrics import AllocatorMemorySession, TorchAllocatorReader
assert 'torch' not in sys.modules
p = AllocatorMemorySession(run_id='real-cpu', device_type='cpu')
p.begin(); assert p.finish()['qualified_interval_counters'] is None
assert 'torch' not in sys.modules
import torch
assert not torch.cuda.is_initialized()
try:
    TorchAllocatorReader(0, torch).read()
except RuntimeError as e:
    assert 'will not initialize' in str(e)
else:
    raise AssertionError('Expected refusal before CUDA initialization')
assert not torch.cuda.is_initialized()
print('real CPU import and CUDA non-initialization verified')
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "non-initialization verified" in result.stdout


def gpu_receipt():
    reader = Reader("cudaMallocAsync")
    reader.metadata["gpu_uuid"] = "GPU-fixture"
    probe = session(reader)
    probe.begin()
    return probe.finish()


def check_receipt(report):
    validate_completed_interval(report, run_id="fixture-not-GPU-evidence", pid=os.getpid(),
        device_type="cuda", gpu_uuid="GPU-fixture")


def test_complete_receipt_is_bound_to_run_process_device_and_pool_semantics():
    check_receipt(gpu_receipt())


@pytest.mark.parametrize("fault", ["run", "pid", "uuid", "status", "reset", "scope", "sync", "peak", "clock", "missing", "outcome"])
def test_tampered_allocator_receipts_are_not_accepted(fault):
    report = gpu_receipt()
    if fault == "run":
        report["run_id"] = "another"
    elif fault == "pid":
        report["identity"]["pid"] = -1
    elif fault == "uuid":
        report["identity"]["gpu_uuid"] = "another"
    elif fault == "status":
        report["status"] = "active"
    elif fault == "reset":
        report["reset_count"] = 2
    elif fault == "scope":
        report["counter_scope"] = "tensor_exclusive"
    elif fault == "sync":
        report["forced_cuda_synchronization"] = True
    elif fault == "peak":
        report["qualified_interval_counters"]["allocated_peak_bytes"] = -1
    elif fault == "clock":
        report["snapshots"][0]["read_started_monotonic"] = float("nan")
    elif fault == "missing":
        report["snapshots"] = []
    elif fault == "outcome":
        report["outcome"] = "cancelled"
    with pytest.raises(ValueError):
        check_receipt(report)
