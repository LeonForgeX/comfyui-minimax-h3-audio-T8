"""Read-only telemetry qualification, including failures and ordinary Core cancellation."""

import json
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.meridian_resources import (
    ObservationErrors,
    ResourceReader,
    ResourceTrace,
    owned_offload,
    note_cleanup_failure,
)
from h3_audio_t8_pkg.meridian_stage_store import StageStore


class Reader:
    def __init__(self):
        self.errors = []
        self.calls = 0
        self.closed = False

    def sample(self):
        self.calls += 1
        return dict(
            stamp=float(self.calls),
            process_rss_bytes=self.calls * 10,
            system_available_bytes=100 - self.calls,
            devices=[
                dict(
                    uuid="GPU-owned-sample",
                    total_bytes=1000,
                    used_bytes=self.calls * 20,
                    free_bytes=1000 - self.calls * 20,
                    utilization_percent=self.calls,
                )
            ],
        )

    def close(self):
        self.closed = True


def test_trace_extrema_endpoints_offload_and_same_return():
    reader = Reader()
    sentinel = object()
    with ResourceTrace("encode", reader_factory=lambda: reader) as trace:
        trace.observe()
        assert owned_offload(lambda: sentinel, "test-owned") is sentinel
    record = trace.report()
    assert reader.closed and record["samples"] == 3
    assert record["before"]["stamp"] == 1 and record["after"]["stamp"] == 3
    assert record["maximum_observed"]["process_rss_bytes"] == 30
    assert record["minimum_observed"]["system_available_bytes"] == 97
    assert record["devices"]["GPU-owned-sample"]["maximum_observed_used_bytes"] == 60
    assert not record["observer_still_finishing"]
    assert record["owned_offloads"][0]["status"] == "complete"
    assert record["seconds"] >= 0 and not record["admission_gate"]
    assert "not node-only" in record["scope"]
    json.dumps(record, allow_nan=False)


def test_nested_observation_restores_owned_offload_parent():
    with ResourceTrace("parent", reader_factory=Reader) as parent:
        with ResourceTrace("child", reader_factory=Reader) as child:
            owned_offload(lambda: None, "child-owned")
        owned_offload(lambda: None, "parent-owned")
    assert [v["name"] for v in child.report()["owned_offloads"]] == ["child-owned"]
    assert [v["name"] for v in parent.report()["owned_offloads"]] == ["parent-owned"]


def test_real_operation_error_not_masked_by_reader_close_failure():
    failure = RuntimeError("actual operation")

    class BadClose(Reader):
        def close(self):
            raise ValueError("optional observer")

    with pytest.raises(RuntimeError) as caught:
        with ResourceTrace("sample", reader_factory=BadClose) as trace:
            owned_offload(lambda: (_ for _ in ()).throw(failure), "owned-failure")
    assert caught.value is failure
    assert trace.report()["owned_offloads"][0]["status"] == "failed"
    assert "close:ValueError" in trace.report()["errors"]


@pytest.mark.parametrize("interval", [-1, 0, 0.01, 61, float("nan"), float("inf")])
def test_invalid_interval_rejected_without_thread(interval):
    with pytest.raises(ValueError):
        ResourceTrace("warp", interval=interval)


def test_missing_reader_and_thread_start_failure_do_not_block(monkeypatch):
    def missing():
        raise ImportError("optional telemetry")

    with ResourceTrace("geometry", reader_factory=missing) as trace:
        value = 71
    assert value == 71 and trace.report()["samples"] == 0
    assert trace.report()["errors"] == ["start:ImportError"]
    monkeypatch.setattr(
        "threading.Thread.start",
        lambda self: (_ for _ in ()).throw(RuntimeError("no thread")),
    )
    reader = Reader()
    with ResourceTrace("geometry", reader_factory=lambda: reader) as trace:
        value = 73
    assert (
        value == 73 and reader.closed and not trace.report()["observer_still_finishing"]
    )


def test_failing_sampler_is_optional_and_errors_bounded():
    class BadSample(Reader):
        def sample(self):
            raise RuntimeError("driver unavailable")

    with ResourceTrace("sample", reader_factory=BadSample) as trace:
        for _ in range(100):
            trace.observe()
    assert trace.report()["samples"] == 0
    assert trace.report()["errors"] == ["sample:RuntimeError"]
    errors = ObservationErrors()
    for i in range(10000):
        errors.append(str(i % 100))
    assert len(errors) == 32


def test_reader_multi_GPU_and_RAM_no_single_GPU_gate(monkeypatch):
    import sys

    calls = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda i: f"GPU-{i}".encode(),
        nvmlDeviceGetMemoryInfo=lambda i: SimpleNamespace(
            total=100, used=30 + i, free=70 - i
        ),
        nvmlDeviceGetUtilizationRates=lambda i: SimpleNamespace(gpu=5 + i),
    )
    fake_psutil = SimpleNamespace(
        Process=lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=21, private=27)
        ),
        virtual_memory=lambda: SimpleNamespace(available=89),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    reader = ResourceReader()
    row = reader.sample()
    reader.close()
    assert len(row["devices"]) == 2 and not reader.errors
    assert row["process_rss_bytes"] == 21 and row["process_private_bytes"] == 27
    assert calls == ["init", "shutdown"]


def test_no_Torch_RNG_change_or_CUDA_initialization():
    before = torch.random.get_rng_state().clone()
    initialized = torch.cuda.is_initialized()
    with ResourceTrace("read-only"):
        pass
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("loaded,initialized", [(False, False), (True, False)])
def test_allocator_does_not_import_or_initialize_Torch(
    monkeypatch, loaded, initialized
):
    import sys

    fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: initialized))
    if loaded:
        monkeypatch.setitem(sys.modules, "torch", fake)
    else:
        monkeypatch.delitem(sys.modules, "torch")
    reader = ResourceReader()
    try:
        result = reader.allocator()
        assert result["devices"] == []
        assert result["status"] == (
            "CUDA_not_initialized" if loaded else "torch_not_loaded"
        )
    finally:
        reader.close()


def test_allocator_explicit_device_reads_and_trace_extrema(monkeypatch):
    import sys

    calls = []
    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            device_count=lambda: 2,
            memory_allocated=lambda i: calls.append(("allocated", i)) or 10 + i,
            memory_reserved=lambda i: calls.append(("reserved", i)) or 20 + i,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    reader = ResourceReader()
    try:
        result = reader.allocator()
        assert result["devices"] == [
            dict(logical_device=i, allocated_bytes=10 + i, reserved_bytes=20 + i)
            for i in range(2)
        ]
        assert calls == [
            ("allocated", 0),
            ("reserved", 0),
            ("allocated", 1),
            ("reserved", 1),
        ]
    finally:
        reader.close()

    class AllocatorReader(Reader):
        def sample(self):
            result = super().sample()
            result["torch_allocator"] = dict(
                devices=[
                    dict(
                        logical_device=0,
                        allocated_bytes=self.calls * 10,
                        reserved_bytes=40,
                    )
                ]
            )
            return result

    with ResourceTrace("sample", reader_factory=AllocatorReader) as trace:
        trace.observe()
    assert trace.report()["torch_allocator_devices"]["0"] == dict(
        maximum_observed_allocated_bytes=30, maximum_observed_reserved_bytes=40
    )


def test_allocator_failure_is_optional(monkeypatch):
    import sys

    fake = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            device_count=lambda: (_ for _ in ()).throw(
                RuntimeError("optional telemetry")
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    reader = ResourceReader()
    try:
        assert reader.allocator()["status"] == "unavailable"
        assert "Torch-allocator:RuntimeError" in reader.errors
    finally:
        reader.close()


@pytest.mark.parametrize(
    "portable,budget", [(True, 1000000), (False, 1000000), (True, 0)]
)
def test_stage_exact_tensor_and_durable_observation(tmp_path, portable, budget):
    store = StageStore(tmp_path, budget_bytes=budget)
    expected = torch.arange(7)
    value, receipt = store.run("sample", {}, lambda: expected, portable=portable)
    assert value is expected and torch.equal(value, expected)
    assert receipt["resources"]["stage"] == "sample"
    saved = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert len(saved) == 1 and saved[0]["resources"]["samples"] >= 2
    if portable and budget:
        _, reused = store.run("sample", {}, lambda: pytest.fail("operation repeated"))
        assert reused["reused"]
        assert (
            reused["resources"]["status"]
            == "operation_skipped_warm_cache_not_new_peak_observation"
        )


def test_real_Core_interrupt_observation_and_same_contract_recovery(tmp_path):
    import comfy.model_management as mm

    store = StageStore(tmp_path)
    error = mm.InterruptProcessingException()
    with pytest.raises(mm.InterruptProcessingException) as caught:
        store.run("sample", {}, lambda: (_ for _ in ()).throw(error))
    assert caught.value is error
    failed = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert (
        failed["status"] == "interrupted" and failed["resources"]["stage"] == "sample"
    )
    assert not list(tmp_path.glob("*.partial-*"))
    value, receipt = store.run("sample", {}, lambda: torch.arange(4))
    assert receipt["status"] == "complete" and len(value) == 4


def test_actual_Core_ordinary_interrupt_flag_is_consumed_and_next_request_recovers(
    tmp_path, monkeypatch
):
    import comfy.model_management as mm

    # This is a private CPU test process, not the user's Core instance.
    monkeypatch.setattr(mm, "interrupt_processing", False)
    store = StageStore(tmp_path)

    def ordinary_cancel():
        mm.interrupt_current_processing(True)
        assert mm.processing_interrupted()
        mm.throw_exception_if_processing_interrupted()
        pytest.fail("Core must throw its ordinary interruption exception")

    with pytest.raises(mm.InterruptProcessingException):
        store.run("sample", {}, ordinary_cancel)
    # Core clears the flag while throwing: do not require True after the exception.
    assert not mm.processing_interrupted()
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["status"] == "interrupted"
    assert not list(tmp_path.glob("*.partial-*"))
    value, receipt = store.run("sample", {}, lambda: torch.arange(5))
    assert receipt["status"] == "complete" and len(value) == 5


@pytest.mark.parametrize("note_api", ["native", "absent", "broken"])
def test_cleanup_note_portable_optional_and_original_error_preserved(note_api, caplog):
    class LegacyError(RuntimeError):
        add_note = None

    class BrokenNote(RuntimeError):
        def add_note(self, message):
            raise RuntimeError("Foreign note handler failed")

    original = {"native": RuntimeError, "absent": LegacyError, "broken": BrokenNote}[
        note_api
    ]("primary error")
    cleanup = RuntimeError("owned offload failed")
    note_cleanup_failure(original, cleanup, "Omega")
    if note_api == "native":
        assert original.__notes__ == [
            "Owned Omega cleanup also failed: owned offload failed"
        ]
        assert not caplog.records
    else:
        assert "Owned Omega cleanup also failed: owned offload failed" in caplog.text
        assert f"original {type(original).__name__} retained" in caplog.text
    try:
        raise original
    except RuntimeError as caught:
        assert caught is original and str(caught) == "primary error"
