from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import trt_vae_backend as backend
from h3_audio_t8_pkg.trt_vae_build import INPUT_SHAPE, graph_spec, validate_request
from test_trt_vae_build import request


def bundle(tmp_path):
    value = request(tmp_path)
    root = Path(value["staging_path"])
    payload = b"Not an executable engine; CPU contract fixture"
    (root / "model.engine").write_bytes(payload)
    ni, no, si, so = graph_spec(value)
    manifest = {"source_request": value, "request_sha256": validate_request(value),
                "engine_bytes": len(payload), "engine_sha256": hashlib.sha256(payload).hexdigest(),
                "input_name": ni, "output_name": no, "input_shape": list(si), "output_shape": list(so),
                "io_dtype": "float16", "runtime_version": value["runtime_version"],
                "gpu_uuid": value["gpu_uuid"], "driver_version": value["driver_version"],
                "compute_capability": [8, 9], "torch_cuda": "13.0"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return root, manifest


def factory(tmp_path, fail=None):
    root, manifest = bundle(tmp_path)
    events = []
    class Memory:
        def identity(self):
            events.append("identity")
            if fail == "identity":
                return dict(manifest, driver_version="different")
            return manifest
        def reserve(self, amount):
            events.append(("reserve", amount))
            if fail == "reserve":
                raise RuntimeError("not enough free memory")
            return 10 * backend.GIB
        def check(self, minimum):
            events.append("check")
            return 5 * backend.GIB
        def free_now(self):
            return 10 * backend.GIB
        def execution_context(self):
            return nullcontext()
        def release(self):
            events.append("release")
            if fail == "release":
                raise RuntimeError("cleanup test")
    class Engine:
        device_memory_size_v2 = 2 * backend.GIB
        def create_execution_context(self):
            events.append("context")
            return None if fail == "context" else object()
    class Runtime:
        def __init__(self, logger):
            events.append("runtime")
        def deserialize_cuda_engine(self, payload):
            events.append("deserialize")
            return None if fail == "deserialize" else Engine()
    class Logger:
        WARNING = 1
        def __init__(self, level):
            pass
    class Runner:
        def __init__(self, *args, **kwargs):
            events.append("runner")
            self.calls = 0
            if fail == "runner":
                raise RuntimeError("runner rejected I/O")
        def __call__(self, value):
            events.append("tile")
            if fail == "tile":
                raise RuntimeError("tile failure")
            self.calls += 1
            return value + 1
        def close(self):
            events.append("close")
    def load(*args):
        events.append("import")
        if fail == "import":
            raise RuntimeError("runtime import failure")
        return SimpleNamespace(Runtime=Runtime, Logger=Logger), {}
    result = backend.ScopedBackend({"decoder": root}, tmp_path, memory=Memory(), runtime_loader=load,
                                   runner_factory=Runner, serial_lease=nullcontext)
    return result, events


def test_profile_selection_checks_all_manifests_and_rejects_ambiguity(tmp_path):
    from h3_audio_t8_pkg import trt_vae_build as build
    paths = []
    for index, t1 in enumerate((False, True)):
        directory = tmp_path / str(index)
        directory.mkdir()
        root, manifest = bundle(directory)
        value = manifest["source_request"]
        value.update(kind="encoder", schema="t8-trt-encoder-t1-build-v1" if t1 else "t8-trt-encoder-build-v1",
                     model_sha256=build.ENCODER_T1_MODEL_SHA if t1 else build.ENCODER_MODEL_SHA,
                     weights_path=None, weights_sha256=None,
                     input_shape=list(build.ENCODER_T1_INPUT_SHAPE if t1 else build.ENCODER_INPUT_SHAPE))
        ni, no, si, so = build.graph_spec(value)
        manifest.update(request_sha256=validate_request(value), input_name=ni, output_name=no,
                        input_shape=list(si), output_shape=list(so))
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
        paths.append(root)
    assert backend.select_bundle(paths, "encoder", [build.ENCODER_T1_INPUT_SHAPE])[0] == paths[1]
    assert backend.select_bundle(paths, "encoder", [build.ENCODER_INPUT_SHAPE])[0] == paths[0]
    for shapes, selected in (([], paths), ([INPUT_SHAPE], paths),
                             ([build.ENCODER_T1_INPUT_SHAPE], [paths[1], paths[1]])):
        with pytest.raises(ValueError):
            backend.select_bundle(selected, "encoder", shapes)
    # A corrupt unselected entry must not be swallowed as 'try next engine'.
    (paths[0] / "manifest.json").write_text("{}", encoding="utf8")
    with pytest.raises(KeyError):
        backend.select_bundle(paths, "encoder", [build.ENCODER_T1_INPUT_SHAPE])


def test_real_backend_load_order_context_reservation_and_cleanup_contract(tmp_path):
    lease, events = factory(tmp_path)
    with lease("decoder", [INPUT_SHAPE]) as tile:
        assert torch.equal(tile(torch.zeros(1)), torch.ones(1))
        assert "release" not in events
    first, second = [e for e in events if isinstance(e, tuple)]
    assert first == ("reserve", 9 * backend.GIB)
    assert second == ("reserve", 3 * backend.GIB + 256 * 1024**2)
    assert events.index(first) < events.index("import") < events.index("deserialize") < events.index(second) < events.index("context")
    assert events.index("close") < events.index("release")
    assert lease.last_report["status"] == "complete" and lease.last_report["calls"] == 1
    assert lease.last_report["context_device_bytes"] == 2 * backend.GIB
    assert not backend._ACTIVE.locked()


@pytest.mark.parametrize("fail", ["identity", "reserve", "import", "deserialize", "context", "runner", "tile", "release"])
def test_all_failure_stages_release_and_never_keep_global_lease(tmp_path, fail):
    lease, events = factory(tmp_path, fail)
    with pytest.raises(RuntimeError):
        with lease("decoder", [INPUT_SHAPE]) as tile:
            tile(torch.zeros(1))
    assert events[-1] == "release" and not backend._ACTIVE.locked()
    if fail == "tile":
        assert "close" in events
    if fail in ("identity", "reserve", "import"):
        assert "deserialize" not in events
    assert lease.last_report["status"] in ("failed", "cleanup_failed")


def test_cancellation_inside_operation_closes_and_has_no_hidden_retry(tmp_path):
    lease, events = factory(tmp_path)
    with pytest.raises(InterruptedError):
        with lease("decoder", [INPUT_SHAPE]):
            raise InterruptedError("user cancelled")
    assert events.count("deserialize") == 1 and events.count("close") == 1
    assert lease.last_report["status"] == "failed" and not backend._ACTIVE.locked()


def test_global_mutex_rejects_nested_lease_before_import_or_offload(tmp_path):
    lease, events = factory(tmp_path)
    with lease("decoder", [INPUT_SHAPE]):
        before = events.copy()
        with pytest.raises(RuntimeError, match="active"):
            with lease("decoder", [INPUT_SHAPE]):
                pytest.fail("concurrent lease entered")
        assert events == before
    assert not backend._ACTIVE.locked()


@pytest.mark.parametrize("bad", ["hash", "size", "shape", "kind", "names", "request", "driver"])
def test_bundle_tampering_is_rejected_before_runtime(tmp_path, bad):
    root, manifest = bundle(tmp_path)
    kind, shape = "decoder", INPUT_SHAPE
    if bad == "hash":
        (root / "model.engine").write_bytes(b"x" * manifest["engine_bytes"])
    elif bad == "size":
        manifest["engine_bytes"] += 1
    elif bad == "shape":
        shape = (1, 24, 1, 16, 16)
    elif bad == "kind":
        kind = "encoder"
    elif bad == "names":
        manifest["input_name"] = "bad"
    elif bad == "request":
        manifest["request_sha256"] = "a" * 64
    elif bad == "driver":
        manifest["driver_version"] = "bad"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError):
        checked_root, checked = backend.inspect_bundle(root, kind, [shape])
        backend.read_payload(checked_root, checked)


def test_conflicting_import_is_detected_without_replacing_user_modules(tmp_path, monkeypatch):
    existing = SimpleNamespace(__file__=str(tmp_path.parent / "other-runtime/__init__.py"))
    monkeypatch.setitem(sys.modules, "tensorrt", existing)
    with pytest.raises(RuntimeError, match="already imported"):
        backend.check_loaded_modules(tmp_path)
    assert sys.modules["tensorrt"] is existing


def test_old_and_new_core_common_memory_api_frees_before_actual_driver_check(monkeypatch):
    import comfy.model_management as mm
    events = []
    monkeypatch.setattr(mm, "free_memory", lambda amount, device: events.append(("offload", amount, device)))
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: events.append("cache"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (events.append("driver") or (99, 100)))
    memory = backend.ComfyMemory("cuda:0")
    monkeypatch.setattr(memory, "physical_free", lambda: 99)
    assert memory.reserve(80) == 99
    assert events == [("offload", 80, torch.device("cuda:0")), "cache", "driver"]
    with pytest.raises(RuntimeError, match="actual GPU free"):
        memory.reserve(100)


def test_windows_physical_deficit_requests_extra_comfy_offload(monkeypatch):
    import comfy.model_management as mm
    events = []
    monkeypatch.setattr(mm, "free_memory", lambda amount, device: events.append(amount))
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100, 200))
    memory = backend.ComfyMemory("cuda:0")
    readings = iter((50, 85))
    monkeypatch.setattr(memory, "physical_free", lambda: next(readings))
    assert memory.reserve(80) == 85
    assert events == [80, 130]
