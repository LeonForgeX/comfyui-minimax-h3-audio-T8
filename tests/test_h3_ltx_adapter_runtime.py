"""Ownership tests use tiny weights; full pinned-weight evidence is separate."""
from contextlib import contextmanager
import hashlib
import json
import sys

import pytest
import torch

from h3_audio_t8_pkg import h3_ltx_adapter_runtime as runtime
from h3_audio_t8_pkg import nodes_h3_ltx_latent_adapter as nodes


@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    source, model = tmp_path / "source", tmp_path / "weights"
    source.mkdir()
    model.mkdir()
    texts = {name: "" for name in runtime.SOURCE_HASHES}
    texts["__init__.py"] = '''
from types import SimpleNamespace
import torch
class H3ToLTXAdapter:
    @classmethod
    def from_pretrained(cls, path, device, dtype):
        return SimpleNamespace(model=torch.nn.Linear(2, 2).to(dtype), device=torch.device(device))
'''
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    for name, text in texts.items():
        (source / name).write_bytes(text.replace("\n", "\r\n").encode())
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test-only")
    monkeypatch.setattr(runtime, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(runtime, "MODEL_SHA", runtime.file_hash(model / "model.safetensors"))
    monkeypatch.setattr(runtime, "CONFIG_SHA", runtime.file_hash(model / "config.json"))
    return source, model


def owned_modules():
    return {name for name in sys.modules if name.startswith("_t8_h3_ltx_owned_")}


def test_line_endings_rng_and_success_cleanup(tiny_assets):
    before = owned_modules()
    rng = torch.get_rng_state().clone()
    with runtime.owned_adapter(*tiny_assets) as adapter:
        assert owned_modules() - before
        assert torch.equal(rng, torch.get_rng_state())
        assert sum(len(m._forward_pre_hooks) for m in adapter.model.modules()) > 0
        adapter.model(torch.zeros(1, 2))
    assert owned_modules() == before
    assert not adapter.model._forward_pre_hooks and adapter.device.type == "cpu"


def test_exception_cleanup_does_not_touch_unrelated_module(tiny_assets):
    sentinel = object()
    sys.modules["h3_ltx_adapter_other_task"] = sentinel
    before = owned_modules()
    try:
        with pytest.raises(RuntimeError, match="test failure"):
            with runtime.owned_adapter(*tiny_assets) as adapter:
                raise RuntimeError("test failure")
        assert owned_modules() == before and not adapter.model._forward_pre_hooks
        assert sys.modules["h3_ltx_adapter_other_task"] is sentinel
    finally:
        sys.modules.pop("h3_ltx_adapter_other_task")


@pytest.mark.parametrize("file", ["adapter.py", "config.json", "model.safetensors"])
def test_modified_asset_rejected_before_import(tiny_assets, file):
    source, model = tiny_assets
    target = (source if file.endswith(".py") else model) / file
    target.write_text("changed", encoding="utf8")
    before = owned_modules()
    with pytest.raises(ValueError, match="Unqualified|hash mismatch"):
        with runtime.owned_adapter(source, model):
            pytest.fail("must not load")
    assert before == owned_modules()


def test_inflight_asset_change_prevents_success(tiny_assets):
    with pytest.raises(ValueError, match="hash mismatch"):
        with runtime.owned_adapter(*tiny_assets):
            (tiny_assets[1] / "config.json").write_text("changed")
    assert not owned_modules()


def test_cancel_in_forward_unloads_and_allows_next_task(tiny_assets):
    cancelled = False
    def check():
        if cancelled:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        with runtime.owned_adapter(*tiny_assets, check_cancel=check) as adapter:
            cancelled = True
            adapter.model(torch.zeros(1, 2))
    assert not adapter.model._forward_pre_hooks and not owned_modules()
    with runtime.owned_adapter(*tiny_assets) as next_adapter:
        assert next_adapter is not adapter
        next_adapter.model(torch.zeros(1, 2))


def test_load_failure_cleans_partial_modules(tiny_assets, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("load failed")
    monkeypatch.setattr(torch.nn, "Linear", fail)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="load failed"):
        with runtime.owned_adapter(*tiny_assets):
            pass
    assert not owned_modules() and torch.equal(rng, torch.get_rng_state())


def test_cleanup_failure_preserves_original_error_and_removes_modules(tiny_assets, monkeypatch):
    def cleanup_failure(*args, **kwargs):
        raise RuntimeError("cleanup failure")
    with pytest.raises(ValueError, match="original conversion error") as found:
        with runtime.owned_adapter(*tiny_assets) as adapter:
            monkeypatch.setattr(adapter.model, "to", cleanup_failure)
            raise ValueError("original conversion error")
    assert not owned_modules() and not adapter.model._forward_pre_hooks
    assert any("cleanup failure" in note for note in found.value.__notes__)


def test_missing_cuda_never_silently_falls_back(tiny_assets, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no silent CPU fallback"):
        with runtime.owned_adapter(*tiny_assets, device="cuda"):
            pytest.fail("must not run")
    assert not owned_modules()


@pytest.mark.parametrize("device,precision", [("auto", "float32"), ("cpu", "float16")])
def test_unknown_execution_rejected(tiny_assets, device, precision):
    with pytest.raises(ValueError, match="Explicit"):
        with runtime.owned_adapter(*tiny_assets, device=device, precision=precision):
            pass


def test_node_keeps_precise_frame_and_fps_outputs(monkeypatch):
    class Adapter:
        def convert(self, value, **kwargs):
            return torch.ones(1, 128, 17, 2, 3)
    @contextmanager
    def owned(*args, **kwargs):
        yield Adapter()
    monkeypatch.setattr(nodes, "owned_adapter", owned)
    source = {"samples": torch.ones(1, 24, 37, 4, 6)}
    result = nodes.MiniMaxH3LTXLatentAdapterEXPT8.execute(source, source_frames=124, frame_policy="crop_to_ltx_grid")
    latent, original, count, fps, serialized = result.result
    assert original is source and count == 121 and fps == 24
    assert latent["samples"].shape == (1, 128, 16, 2, 3)
    assert json.loads(serialized)["audio_duration_adjustment_required"]


def test_invalid_grid_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(nodes, "owned_adapter", lambda *a, **k: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="explicitly choose"):
        nodes.MiniMaxH3LTXLatentAdapterEXPT8.execute({"samples": torch.ones(1,24,37,4,6)}, source_frames=124)


def test_schema_defers_optional_dependency_and_has_no_frame_ceiling():
    schema = nodes.MiniMaxH3LTXLatentAdapterEXPT8.define_schema()
    assert schema.node_id == "MiniMaxH3LTXLatentAdapterEXPT8"
    assert len(schema.outputs) == 5
    assert not owned_modules()
