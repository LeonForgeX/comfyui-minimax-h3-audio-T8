"""Scoped wall-time accounting; never equates speed with quality acceptance.

No automatic CUDA synchronization, resource reset, model loading or filesystem
writes. The caller supplies resource observations and marks actual cache hits.
SynchronizedCudaMeasurement is a separate, explicit opt-in for an exclusively
owned benchmark process. It synchronizes stage boundaries and resets that
process's allocator peaks; it never changes pinning, residency, or model policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import time
import statistics


SCHEMA = "t8.h3.acceleration.measurement.v1"
WORKLOAD_FIELDS = frozenset({"model_sha256", "conditioning_sha256", "video_vae_sha256",
                             "audio_vae_sha256", "width", "height", "frames", "fps_num",
                             "fps_den", "seed", "nfe", "cfg", "recipe_sha256", "core_revision"})


class Measurement:
    def __init__(self, workload, *, cache_state, clock=time.perf_counter):
        if not isinstance(workload, dict) or not WORKLOAD_FIELDS.issubset(workload):
            raise ValueError("A complete workload identity is required")
        if cache_state not in ("cold", "warm"):
            raise ValueError("cache_state must be cold or warm")
        for name in ("model_sha256", "conditioning_sha256", "recipe_sha256", "video_vae_sha256", "audio_vae_sha256"):
            value = workload[name]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name in ("width", "height", "frames", "fps_num", "fps_den", "nfe"):
            value = workload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        seed = workload["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        cfg = workload["cfg"]
        if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 0:
            raise ValueError("cfg must be finite and nonnegative")
        if not isinstance(workload["core_revision"], str) or not workload["core_revision"]:
            raise ValueError("core_revision is required")
        self.workload = deepcopy(workload)
        self.cache_state = cache_state
        self.clock = clock
        self.started = self._now()
        self.stages = []
        self.observations = []
        self.active = False
        self.closed = False
        self.failed = False
        self.cache_hit = False

    def _now(self):
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError("Clock must be finite")
        return value

    def _open(self):
        if self.closed:
            raise RuntimeError("Measurement is already closed")

    def record_cache_hit(self):
        self._open()
        self.cache_hit = True

    def observe_resources(self, observation):
        self._open()
        if not isinstance(observation, dict) or not observation:
            raise ValueError("Resource observation must be a nonempty mapping")
        values = {}
        for name, value in observation.items():
            if not isinstance(name, str) or not name.endswith("_bytes"):
                raise ValueError("Resource fields must explicitly use bytes")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Resource bytes must be nonnegative integers")
            values[name] = value
        self.observations.append({"elapsed_seconds": self._now() - self.started, **values})

    @contextmanager
    def stage(self, name):
        self._open()
        if self.active:
            raise RuntimeError("Nested stages would double-count elapsed time")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Stage needs a name")
        began = self._now()
        self.active = True
        status = "complete"
        try:
            yield
        except BaseException:
            self.failed = True
            status = "failed"
            raise
        finally:
            self.active = False
            duration = self._now() - began
            if duration < 0:
                self.failed = True
                raise RuntimeError("Clock moved backwards")
            self.stages.append({"name": name, "seconds": duration, "status": status})

    def finish(self):
        self._open()
        if self.active:
            raise RuntimeError("Cannot finish during a stage")
        elapsed = self._now() - self.started
        accounted = sum(stage["seconds"] for stage in self.stages)
        if elapsed <= 0 or accounted > elapsed + 1e-9:
            raise RuntimeError("Invalid total or double-counted wall time")
        self.closed = True
        return {"schema": SCHEMA, "status": "failed" if self.failed else "complete",
                "workload": deepcopy(self.workload), "cache_state": self.cache_state,
                "cache_hit": self.cache_hit, "elapsed_seconds": elapsed,
                "stage_seconds": accounted, "unassigned_seconds": max(0., elapsed - accounted),
                "stages": deepcopy(self.stages), "resource_observations": deepcopy(self.observations),
                "resource_scope": "observed_samples_not_continuous_peaks",
                "quality": "not_evaluated", "synchronization": "caller_controlled"}


def compare_equal_workload(baseline, candidate):
    """Compare wall time only. Different-model/product comparisons are separate."""
    for report in (baseline, candidate):
        if report.get("schema") != SCHEMA or report.get("status") != "complete":
            raise ValueError("Both timing reports must be complete")
        if report.get("cache_hit") is not False:
            raise ValueError("Cached graph outputs cannot prove acceleration")
        value = report.get("elapsed_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid elapsed time")
        if not isinstance(report.get("workload"), dict) or not WORKLOAD_FIELDS.issubset(report["workload"]):
            raise ValueError("Missing workload identity")
        if report.get("cache_state") not in ("cold", "warm"):
            raise ValueError("Missing cache state")
    if baseline["workload"] != candidate["workload"]:
        raise ValueError("Workload mismatch; report separately as a product comparison")
    if baseline["cache_state"] != candidate["cache_state"]:
        raise ValueError("Cannot compare cold with warm timing")
    if baseline.get("measurement_runtime") != candidate.get("measurement_runtime"):
        raise ValueError("Measurement runtime mismatch; hardware, software and synchronization must match")
    ratio = candidate["elapsed_seconds"] / baseline["elapsed_seconds"]
    return {"scope": "wall_time_only_quality_and_memory_not_accepted",
            "saved_fraction": 1 - ratio, "speedup": 1 / ratio,
            "baseline_seconds": baseline["elapsed_seconds"],
            "candidate_seconds": candidate["elapsed_seconds"]}


class SynchronizedCudaMeasurement(Measurement):
    """Synchronized wall time + allocator peaks in an isolated CUDA worker.

The caller owns cold/warm preparation and places real work inside stage blocks.
No empty-cache, weight offload, pinning, or implicit warmup is performed. Do not
use in a shared Core: reset_peak_memory_stats affects the whole local allocator.
GPU peaks cover this process's torch allocator, NOT whole-device VRAM. RAM is
not sampled here and must not be labelled as a process/system peak.
"""
    def __init__(self, workload, *, cache_state, runtime_identity, exclusive_process=False,
                 device=0, clock=time.perf_counter, cuda_backend=None):
        if exclusive_process is not True:
            raise ValueError("CUDA peak measurement requires an explicitly exclusive_process")
        if type(device) is not int or device < 0:
            raise ValueError("device must be an explicit nonnegative CUDA index")
        required = {"gpu_uuid", "gpu_name", "torch_version", "cuda_version", "driver_version", "attention_backend"}
        if (not isinstance(runtime_identity, dict) or not required.issubset(runtime_identity)
                or any(not isinstance(runtime_identity[key], str) or not runtime_identity[key] for key in required)):
            raise ValueError("A complete runtime_identity is required")
        # Validate the workload before creating a CUDA context or resetting peaks.
        super().__init__(workload, cache_state=cache_state, clock=clock)
        if cuda_backend is None:
            import torch
            cuda_backend = torch.cuda
        if not cuda_backend.is_available():
            raise RuntimeError("CUDA is unavailable; use Measurement for CPU timing")
        self.cuda = cuda_backend
        self.device = device
        self.runtime_identity = deepcopy(runtime_identity)
        self.instrumentation_errors = []
        self.cuda.synchronize(self.device)
        self.cuda.reset_peak_memory_stats(self.device)
        self.started = self._now()

    def _allocator_observation(self):
        return {"torch_allocated_bytes": int(self.cuda.memory_allocated(self.device)),
                "torch_reserved_bytes": int(self.cuda.memory_reserved(self.device)),
                "torch_peak_allocated_bytes": int(self.cuda.max_memory_allocated(self.device)),
                "torch_peak_reserved_bytes": int(self.cuda.max_memory_reserved(self.device))}

    @contextmanager
    def stage(self, name):
        self._open()
        if self.active:
            raise RuntimeError("Nested stages would double-count elapsed time")
        self.cuda.synchronize(self.device)
        with super().stage(name):
            body_failed = False
            try:
                yield
            except BaseException:
                body_failed = True
                raise
            finally:
                try:
                    self.cuda.synchronize(self.device)
                    self.observe_resources(self._allocator_observation())
                except BaseException as error:
                    self.failed = True
                    self.instrumentation_errors.append(f"{type(error).__name__}: {error}")
                    if not body_failed:
                        raise

    def finish(self):
        self._open()
        if self.active:
            raise RuntimeError("Cannot finish during a stage")
        try:
            self.cuda.synchronize(self.device)
            self.observe_resources(self._allocator_observation())
        except Exception as error:
            self.failed = True
            self.instrumentation_errors.append(f"{type(error).__name__}: {error}")
        report = super().finish()
        report.update(
            measurement_runtime={**self.runtime_identity, "device_index": self.device,
                                 "synchronization": "cuda_at_each_stage_boundary"},
            synchronization="cuda_at_each_stage_boundary",
            resource_scope="isolated_process_torch_allocator_peaks_not_whole_device_or_host_ram",
            instrumentation_errors=list(self.instrumentation_errors),
            cache_preparation="caller_declared_no_implicit_warmup_or_eviction",
        )
        return report


def summarize_repeated_measurements(reports, *, minimum_runs=3):
    """Summarize one exact workload/runtime/cache class; never pool cold and warm."""
    reports = list(reports)
    if type(minimum_runs) is not int or minimum_runs < 1 or len(reports) < minimum_runs:
        raise ValueError("Not enough repeated measurements")
    first = reports[0]
    for report in reports:
        compare_equal_workload(first, report)
        stages = report.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("A nonempty stage record is required")
        for stage in stages:
            seconds = stage.get("seconds") if isinstance(stage, dict) else None
            if (not isinstance(stage, dict) or not isinstance(stage.get("name"), str)
                    or not stage["name"] or stage.get("status") != "complete"
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError("Invalid or incomplete stage record")
        if sum(stage["seconds"] for stage in stages) > report["elapsed_seconds"] + 1e-9:
            raise ValueError("Stage durations exceed total elapsed time")
    stage_names = [stage["name"] for stage in first["stages"]]
    if any([stage["name"] for stage in report["stages"]] != stage_names for report in reports):
        raise ValueError("Stage sequence mismatch")
    totals = [report["elapsed_seconds"] for report in reports]
    return {"schema": "t8.h3.acceleration.baseline.v1", "runs": len(reports),
            "workload": deepcopy(first["workload"]), "cache_state": first["cache_state"],
            "measurement_runtime": deepcopy(first.get("measurement_runtime")),
            "median_seconds": statistics.median(totals), "min_seconds": min(totals), "max_seconds": max(totals),
            "stages": [{"name": name, "median_seconds": statistics.median(report["stages"][i]["seconds"] for report in reports)}
                       for i, name in enumerate(stage_names)],
            "quality": "not_evaluated", "speedup_claim": False}
