"""Actual Core LoadedModel weakref lifetime, without loading a GPU model."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from comfy.model_patcher import ModelPatcher
from h3_audio_t8_pkg import long_video_dual_residency as residency


def make_patcher(model):
    return ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def test_actual_core_dead_patcher_does_not_break_unrelated_stage_release(monkeypatch):
    orphan_model = torch.nn.Linear(2, 2)
    orphan_patcher = make_patcher(orphan_model)
    orphan = residency.mm.LoadedModel(orphan_patcher)
    orphan.real_model = weakref.ref(orphan_model)
    del orphan_patcher
    gc.collect()
    assert orphan.model is None and orphan.is_dead()
    active = make_patcher(torch.nn.Linear(2, 2))
    entry = residency.mm.LoadedModel(active)
    entries = [orphan, entry]
    monkeypatch.setattr(residency.mm, "current_loaded_models", entries)
    calls = []
    monkeypatch.setattr(residency.mm, "free_memory", lambda amount, device, keep_loaded: calls.append(keep_loaded))
    report = residency.release_stage_residency(active)
    assert calls == [[orphan]] and entries == [orphan, entry]
    assert report["unloaded_entries"] == 1 and report["stale_entries_skipped"] == 1


def test_resolved_target_patcher_retained_through_native_unload(monkeypatch):
    model = object()
    events = []
    requested = SimpleNamespace(model=model, load_device="cpu")
    clone = SimpleNamespace(model=model, load_device="cpu", partially_unload_ram=lambda _: events.append("ram") or 21)
    entry = SimpleNamespace(model=clone, device="cpu")
    monkeypatch.setattr(residency.mm, "current_loaded_models", [entry])
    def unload(*args, **kwargs):
        events.append("native_unload")
        entry.model = None
    monkeypatch.setattr(residency.mm, "free_memory", unload)
    result = residency.release_stage_residency(requested)
    assert events == ["native_unload", "ram"] and result["core_reported_ram_freed_bytes"] == 21


def test_absent_target_model_rejected_before_native_cleanup(monkeypatch):
    monkeypatch.setattr(residency.mm, "current_loaded_models", [])
    with pytest.raises(ValueError, match="actual Core model patcher"):
        residency.release_stage_residency(SimpleNamespace(model=None, load_device="cpu"))


def test_native_unload_error_preserved_without_ram_or_boundary_cleanup(monkeypatch):
    error = RuntimeError("native unload failure")
    patcher = SimpleNamespace(model=object(), load_device="cpu",
        partially_unload_ram=lambda _: pytest.fail("RAM cleanup after native failure"))
    monkeypatch.setattr(residency.mm, "current_loaded_models", [SimpleNamespace(model=patcher, device="cpu")])
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(residency.mm, "free_memory", fail)
    with pytest.raises(RuntimeError) as caught:
        residency.release_stage_residency(patcher)
    assert caught.value is error


def test_actual_core_clone_parent_fallback_is_live_not_stale(monkeypatch):
    model = torch.nn.Linear(2, 2)
    parent = make_patcher(model)
    clone = parent.clone()
    clone.parent = parent
    entry = residency.mm.LoadedModel(clone)
    del clone
    gc.collect()
    assert entry.model is parent
    monkeypatch.setattr(residency.mm, "current_loaded_models", [entry])
    calls = []
    monkeypatch.setattr(residency.mm, "free_memory", lambda *a, **k: calls.append(k["keep_loaded"]))
    result = residency.release_stage_residency(parent)
    assert calls == [[]] and result["stale_entries_skipped"] == 0


def test_all_stale_entries_no_global_mutation_or_unload(monkeypatch):
    stale = SimpleNamespace(model=None, device="cpu")
    entries = [stale]
    monkeypatch.setattr(residency.mm, "current_loaded_models", entries)
    monkeypatch.setattr(residency.mm, "free_memory", lambda *a, **k: pytest.fail("No live target"))
    result = residency.release_stage_residency(SimpleNamespace(model=object(), load_device="cpu"))
    assert result["unloaded_entries"] == 0 and result["stale_entries_skipped"] == 1
    assert residency.mm.current_loaded_models is entries and entries == [stale]
