from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import long_video_dual_residency as residency


def test_only_finished_stage_loaded_clones_are_released(monkeypatch):
    freed = []
    base, other = object(), object()
    original = SimpleNamespace(model=base, load_device='cuda:0')
    clone = SimpleNamespace(model=base, load_device='cuda:0', partially_unload_ram=lambda amount: freed.append(amount) or 123)
    unrelated = SimpleNamespace(model=other, load_device='cuda:0', partially_unload_ram=lambda _: (_ for _ in ()).throw(AssertionError()))
    first, keep = SimpleNamespace(model=clone, device='cuda:0'), SimpleNamespace(model=unrelated, device='cuda:0')
    monkeypatch.setattr(residency.mm, 'current_loaded_models', [first, keep])
    calls = []
    monkeypatch.setattr(residency.mm, 'free_memory', lambda amount, device, keep_loaded: calls.append((device, keep_loaded)))
    report = residency.release_stage_residency(SimpleNamespace(patcher=original), original)
    assert calls == [('cuda:0', [keep])]
    assert len(freed) == 1
    assert report['core_reported_ram_freed_bytes'] == 123


def test_never_loaded_component_does_not_touch_dynamic_pin_state(monkeypatch):
    monkeypatch.setattr(residency.mm, 'current_loaded_models', [])
    patcher = SimpleNamespace(model=object(), load_device='cuda:0',
        partially_unload_ram=lambda _: (_ for _ in ()).throw(AssertionError('uninitialized pin buffers')))
    report = residency.release_stage_residency(patcher)
    assert report['unloaded_entries'] == 0


def test_dynamic_internal_stage_performs_native_node_boundary_cleanup(monkeypatch):
    events = []
    patcher = SimpleNamespace(model=object(), load_device='cuda:0', is_dynamic=lambda: True,
        partially_unload_ram=lambda _: events.append('ram') or 0)
    entry = SimpleNamespace(model=patcher, device='cuda:0')
    monkeypatch.setattr(residency.mm, 'current_loaded_models', [entry])
    monkeypatch.setattr(residency.mm, 'free_memory', lambda *a, **k: events.append('unload'))
    monkeypatch.setattr(residency, '_complete_native_dynamic_stage',
        lambda: events.append('native_boundary') or 'completed', raising=False)
    report = residency.release_stage_residency(patcher)
    assert events == ['unload', 'ram', 'native_boundary']
    assert report['native_stage_boundary'] == 'completed'


def test_unused_dynamic_component_does_not_reset_shared_core_buffers(monkeypatch):
    monkeypatch.setattr(residency.mm, 'current_loaded_models', [])
    monkeypatch.setattr(residency, '_complete_native_dynamic_stage',
        lambda: (_ for _ in ()).throw(AssertionError('No executed stage')), raising=False)
    patcher = SimpleNamespace(model=object(), load_device='cuda:0', is_dynamic=lambda: True)
    report = residency.release_stage_residency(patcher)
    assert report['native_stage_boundary'] == 'not_dynamic_or_not_loaded'


def test_native_cleanup_order_matches_core_execution_boundary(monkeypatch):
    import comfy.memory_management as memory
    import comfy.model_prefetch as prefetch
    import comfy_aimdo.model_vbar as model_vbar
    events = []
    monkeypatch.setattr(memory, 'aimdo_enabled', True)
    monkeypatch.setattr(prefetch, 'cleanup_prefetch_queues', lambda: events.append('prefetch'))
    monkeypatch.setattr(residency.mm, 'reset_cast_buffers', lambda: events.append('casts_mmaps'))
    monkeypatch.setattr(model_vbar, 'vbars_reset_watermark_limits', lambda: events.append('watermarks'))
    assert residency._complete_native_dynamic_stage() == 'completed'
    assert events == ['prefetch', 'casts_mmaps', 'watermarks']
    monkeypatch.setattr(model_vbar, 'vbars_reset_watermark_limits', None)
    events.clear()
    with pytest.raises(RuntimeError, match='cleanup API'):
        residency._complete_native_dynamic_stage()
    assert events == []


def test_nondynamic_core_skips_aimdo_cleanup(monkeypatch):
    import comfy.memory_management as memory
    monkeypatch.setattr(memory, 'aimdo_enabled', False)
    assert residency._complete_native_dynamic_stage() == 'aimdo_disabled'
