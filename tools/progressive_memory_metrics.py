"""Allocator telemetry for an owned, serial benchmark process; not a node patch.

No tensor allocation, device switching, cache clearing or CUDA synchronization.
Snapshots carry cumulative peaks since ONE reset, not per-stage peak claims.
cudaMallocAsync 2.10 reports default-pool usage, not tensor-exclusive bytes.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
import time


SCHEMA = "t8.progressive.allocator-memory.v1"
COUNTERS = {
    "allocated_current_bytes": "allocated_bytes.all.current",
    "allocated_peak_bytes": "allocated_bytes.all.peak",
    "reserved_current_bytes": "reserved_bytes.all.current",
    "reserved_peak_bytes": "reserved_bytes.all.peak",
}


def validated_counters(raw):
    """Missing counters are unavailable, never implicit zeroes."""
    if not isinstance(raw, dict):
        raise ValueError("Allocator statistics must be a mapping")
    result = {}
    for name, key in COUNTERS.items():
        value = raw.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Missing or invalid allocator counter: {key}")
        result[name] = value
    for kind in ("allocated", "reserved"):
        if result[f"{kind}_current_bytes"] > result[f"{kind}_peak_bytes"]:
            raise ValueError("Allocator current exceeds its peak")
    if (result["allocated_current_bytes"] > result["reserved_current_bytes"] or
            result["allocated_peak_bytes"] > result["reserved_peak_bytes"]):
        raise ValueError("Allocator used bytes exceed reserved bytes")
    return result


def counter_scope(backend, torch_version):
    if backend == "native":
        return "pytorch_native_caching_allocator_not_whole_device"
    if backend == "cudaMallocAsync" and torch_version.split("+")[0] == "2.10.0":
        return "cuda_default_pool_used_and_reserved_not_tensor_exclusive"
    raise ValueError("Allocator semantics unqualified for this backend/version")


class TorchAllocatorReader:
    """Use only an already-initialized current CUDA device in this process."""

    def __init__(self, device_index, torch_module=None):
        if type(device_index) is not int or device_index < 0:
            raise ValueError("An explicit nonnegative CUDA device index is required")
        if torch_module is None:
            import torch as torch_module
        self.torch, self.device_index = torch_module, device_index

    def _check(self):
        cuda = self.torch.cuda
        if not cuda.is_initialized():
            raise RuntimeError("CUDA is not initialized; telemetry will not initialize it")
        if cuda.current_device() != self.device_index:
            raise RuntimeError("Current CUDA device differs from the owned benchmark device")

    def identity(self):
        self._check()
        props = self.torch.cuda.get_device_properties(self.device_index)
        uuid = getattr(props, "uuid", None)
        return {"pid": os.getpid(), "device_index": self.device_index,
                "gpu_uuid": str(uuid) if uuid is not None else None,
                "backend": self.torch.cuda.get_allocator_backend(),
                "torch_version": str(self.torch.__version__),
                "cuda_version": self.torch.version.cuda}

    def read(self):
        self._check()
        return self.torch.cuda.memory_stats(self.device_index)

    def reset_peaks(self):
        self._check()
        self.torch.cuda.reset_peak_memory_stats(self.device_index)


class AllocatorMemorySession:
    """One explicit benchmark interval in an isolated process.

    Call begin immediately before the generation graph and finish after its
    terminal boundary. Observations name boundaries, not isolated stage maxima.
    CPU mode does not instantiate or call a CUDA reader. It reports unavailable.
    The caller must ensure exclusive job ownership; this is not a GPU lock.
    """

    def __init__(self, *, run_id, device_type, device_index=None, reader=None,
                 clock=time.perf_counter):
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 256:
            raise ValueError("A bounded nonempty benchmark run identity is required")
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("Only explicit CPU/CUDA benchmark modes are supported")
        if device_type == "cuda" and (type(device_index) is not int or device_index < 0):
            raise ValueError("CUDA measurement needs an explicit device index")
        if device_type == "cpu" and device_index is not None:
            raise ValueError("CPU measurement must not select a CUDA device")
        self.run_id, self.device_type, self.device_index = run_id, device_type, device_index
        self.reader = (reader if reader is not None else TorchAllocatorReader(device_index)) if device_type == "cuda" else None
        self.clock, self.pid = clock, os.getpid()
        self.state, self.outcome, self.error = "new", None, None
        self.identity, self.scope, self.before_reset = None, None, None
        self.snapshots, self.reset_count = [], 0
        self.last_time = None

    def _now(self):
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Allocator observation clock must be finite")
        if self.last_time is not None and value < self.last_time:
            raise ValueError("Allocator observation clock moved backwards")
        self.last_time = value
        return value

    def _check_owner(self):
        if os.getpid() != self.pid:
            raise RuntimeError("Allocator session cannot cross process ownership")

    def _check_identity(self):
        self._check_owner()
        if self.reader.identity() != self.identity:
            raise RuntimeError("Allocator/device identity changed during the interval")

    def _fail(self, error):
        self.state, self.outcome = "failed", "failed"
        self.error = f"{type(error).__name__}: {error}"

    def begin(self, boundary="before_generation"):
        if self.state != "new":
            raise RuntimeError("An allocator interval can only begin once")
        try:
            self._check_owner()
            if self.device_type == "cpu":
                self.state, self.scope = "unavailable", "cpu_run_no_cuda_measurement"
                return self.report()
            self.identity = deepcopy(self.reader.identity())
            if (self.identity.get("pid") != self.pid or
                    self.identity.get("device_index") != self.device_index):
                raise RuntimeError("Reader identity is not the owned process/device")
            self.scope = counter_scope(self.identity["backend"], self.identity["torch_version"])
            self.before_reset = validated_counters(self.reader.read())
            self.reader.reset_peaks()
            self.reset_count += 1
            self.state = "active"
            self.observe(boundary)
            return self.report()
        except BaseException as error:
            self._fail(error)
            raise

    def observe(self, boundary):
        if self.state != "active":
            raise RuntimeError("Allocator observation requires an active interval")
        try:
            if not isinstance(boundary, str) or not boundary.strip() or len(boundary) > 128:
                raise ValueError("Observation needs a bounded boundary label")
            self._check_identity()
            started = self._now()
            values = validated_counters(self.reader.read())
            ended = self._now()
            if self.snapshots:
                for kind in ("allocated", "reserved"):
                    name = f"{kind}_peak_bytes"
                    if values[name] < self.snapshots[-1][name]:
                        raise RuntimeError("Allocator peaks decreased; possible external reset")
            self.snapshots.append({"boundary": boundary, "read_started_monotonic": started,
                "read_ended_monotonic": ended, "read_seconds": ended-started, **values})
            return deepcopy(self.snapshots[-1])
        except BaseException as error:
            self._fail(error)
            raise

    def finish(self, boundary="after_generation_terminal", *, outcome="complete"):
        if outcome not in {"complete", "failed", "cancelled"}:
            raise ValueError("Unknown benchmark outcome")
        if self.state == "unavailable":
            self._check_owner()
            self.state, self.outcome = "closed_unavailable", outcome
            return self.report()
        if self.state != "active":
            raise RuntimeError("Cannot finish an inactive allocator interval")
        self.observe(boundary)
        self.state, self.outcome = "closed", outcome
        return self.report()

    def report(self):
        qualified = self.state == "closed" and self.outcome == "complete"
        return deepcopy({"schema": SCHEMA, "run_id": self.run_id,
            "status": self.state, "outcome": self.outcome, "error": self.error,
            "identity": self.identity, "device_type": self.device_type,
            "counter_scope": self.scope, "reset_count": self.reset_count,
            "pre_interval_counters": self.before_reset, "snapshots": self.snapshots,
            "qualified_interval_counters": self.snapshots[-1] if qualified else None,
            "complete_interval_observed": qualified,
            "peak_scope": "cumulative_since_single_begin_reset_including_live_starting_allocations",
            "stage_peak_attribution": False, "whole_device_memory": False,
            "forced_cuda_synchronization": False, "cleared_allocator_cache": False,
            "limits": ["No proof of inference execution, graph-cache exclusion or CUDA completion.",
                       "Allocator counters do not include all external CUDA allocations or other processes.",
                       "cudaMallocAsync pool counters may include other libraries sharing the default pool.",
                       "Still requires whole-device NVML and system RAM observations."]})


def validate_completed_interval(report, *, run_id, pid, device_type, gpu_uuid=None):
    """Validate a returned probe receipt; not an inference or image-quality gate."""
    if (report.get("schema") != SCHEMA or report.get("run_id") != run_id or
            report.get("device_type") != device_type or report.get("outcome") != "complete" or
            report.get("error") is not None):
        raise ValueError("Allocator receipt identity/outcome is invalid")
    if device_type == "cpu":
        if (report.get("status") != "closed_unavailable" or report.get("identity") is not None or
                report.get("snapshots") != [] or report.get("qualified_interval_counters") is not None or
                report.get("reset_count") != 0 or report.get("complete_interval_observed") is not False):
            raise ValueError("CPU receipt must not claim CUDA measurements")
        return
    if device_type != "cuda":
        raise ValueError("Unsupported allocator receipt device")
    identity = report.get("identity") or {}
    if (identity.get("pid") != pid or type(identity.get("device_index")) is not int or
            identity["device_index"] < 0 or report.get("status") != "closed" or
            report.get("complete_interval_observed") is not True or report.get("reset_count") != 1):
        raise ValueError("GPU allocator interval was not completely observed")
    def normalized_uuid(value):
        if not isinstance(value, str) or not value:
            raise ValueError("GPU UUID is required to bind allocator and device telemetry")
        return value.lower().removeprefix("gpu-")
    if normalized_uuid(identity.get("gpu_uuid")) != normalized_uuid(gpu_uuid):
        raise ValueError("Allocator and NVML GPU UUID differ")
    if report.get("counter_scope") != counter_scope(identity["backend"], identity["torch_version"]):
        raise ValueError("Allocator counter semantics changed")
    for flag in ("stage_peak_attribution", "whole_device_memory", "forced_cuda_synchronization", "cleared_allocator_cache"):
        if report.get(flag) is not False:
            raise ValueError("Allocator observation policy differs")
    snapshots = report.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) < 2:
        raise ValueError("Allocator interval boundaries are missing")
    if (snapshots[0].get("boundary") != "before_generation" or
            snapshots[-1].get("boundary") != "after_generation_terminal" or
            report.get("qualified_interval_counters") != snapshots[-1]):
        raise ValueError("Allocator terminal counters do not bind the complete interval")
    previous_time, previous_peaks = None, None
    for row in snapshots:
        values = validated_counters({source: row.get(name) for name, source in COUNTERS.items()})
        start, end = row.get("read_started_monotonic"), row.get("read_ended_monotonic")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (start, end)):
            raise ValueError("Allocator interval clock is invalid")
        if end < start or (previous_time is not None and start < previous_time):
            raise ValueError("Allocator interval clock moved backwards")
        if row.get("read_seconds") != end-start:
            raise ValueError("Allocator observation duration is inconsistent")
        if previous_peaks and any(values[key] < previous_peaks[key] for key in ("allocated_peak_bytes", "reserved_peak_bytes")):
            raise ValueError("Allocator peaks decreased during the interval")
        previous_time, previous_peaks = end, values
