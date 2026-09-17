"""Installed KJ/Sol code with actual tiny Core Euler, CPU fallbacks explicit."""

import json

import pytest
import torch

import test_progressive_sampling_runtime as runtime_fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_progressive_sampling_runtime import tiny_model
from test_progressive_stage_models import run
from test_relay_kj_backend import kj as kj
stub_lifter = runtime_fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


@pytest.mark.parametrize('configured', [False, True])
def test_real_sol_selector_runs_without_hiding_cpu_fallback(stub_lifter, installed_sol, configured):
    original = tiny_model()
    selected = installed_sol.SolAttentionPatch().patch(original, True, .5, min_tokens=256)[0]
    override = selected.model_options['transformer_options']['optimized_attention_override']
    output, report = run(selected, **({'model_hires': selected} if configured else {}))
    report = json.loads(report)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    assert selected.model_options['transformer_options']['optimized_attention_override'] is override
    assert not original.wrappers and not selected.wrappers
    if not configured:
        # Legacy Core dispatch is retained, not silently replaced to obtain
        # scoped kernel counters. No Sol CUDA execution is certified on CPU.
        assert 'attention' not in report
        return
    phases = report['attention']
    backend = phases['low']
    assert backend['configuration']['tau'] == .5
    assert backend['completed_calls'].get('sol:completed', 0) == 0
    assert sum(backend['completed_calls'].values()) > 0
    assert sum(backend['completed_calls'].values()) == sum(phases['high']['completed_calls'].values()) == 4
    assert phases['shared_backend_instance'] is True and phases['shared_counter'] is False
    assert not original.wrappers and not selected.wrappers


@pytest.mark.parametrize('kind', ['sage', 'sage_lowmem_ffn', 'lowmem_ffn', 'ffn'])
@pytest.mark.parametrize('configured', [False, True])
def test_actual_kj_memory_patches_execute_full_progressive_sampling(stub_lifter, memory_nodes, kind, configured):
    lowmem, sage, calls = memory_nodes
    original = tiny_model()
    selected = original
    if 'sage' in kind:
        selected = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(selected).result[0]
    if 'lowmem' in kind:
        selected = lowmem.MiniMaxLowVRAMAttention.execute(selected, 2).result[0]
    if 'ffn' in kind:
        selected = lowmem.MiniMaxChunkFeedForward.execute(selected, 2, 256).result[0]
    before = dict(selected.object_patches)
    blocks = selected.model.diffusion_model.blocks
    live_before = [(block.attn.forward, block.mlp.forward) for block in blocks]
    output, report = run(selected, **({'model_hires': selected} if configured else {}))
    report = json.loads(report)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    assert selected.object_patches == before
    assert [(block.attn.forward, block.mlp.forward) for block in blocks] == live_before
    assert not selected.wrappers and not original.wrappers
    if 'sage' in kind:
        if configured:
            assert calls == []  # explicit composition uses the scoped delegate
            assert sum(report['attention']['low']['completed_calls'].values()) > 0
        else:
            # The original callable really runs on the unchanged legacy route.
            per_forward = 2 if 'lowmem' in kind else 1
            assert calls == ['original_direct_sage'] * (8 * len(blocks) * per_forward)
            assert 'attention' not in report


@pytest.mark.parametrize('configured', [False, True])
def test_unknown_override_is_retained_and_really_delegated(stub_lifter, configured):
    model = tiny_model()
    calls = []

    def selected(original, *args, **kwargs):
        calls.append('selected')
        return original(*args, **kwargs)

    model.model_options['transformer_options']['optimized_attention_override'] = selected
    output, report = run(model, **({'model_hires': model} if configured else {}))
    assert calls == ['selected'] * (8 * len(model.model.diffusion_model.blocks))
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    assert json.loads(report)['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert model.model_options['transformer_options']['optimized_attention_override'] is selected
    assert not model.wrappers


@pytest.mark.parametrize('configured', [False, True])
def test_unknown_override_real_kernel_error_is_not_hidden(stub_lifter, configured):
    model = tiny_model()

    def selected(*args, **kwargs):
        raise RuntimeError('selected kernel failed')

    model.model_options['transformer_options']['optimized_attention_override'] = selected
    with pytest.raises(RuntimeError, match='selected kernel failed'):
        run(model, **({'model_hires': model} if configured else {}))
    assert model.model_options['transformer_options']['optimized_attention_override'] is selected
    assert not model.wrappers
    assert not stub_lifter


def test_core_sage_bias_delegate_preserves_missing_kernel_diagnostics(monkeypatch):
    from h3_audio_t8_pkg.progressive_attention import _PlainDelegate
    from h3_audio_t8_pkg import relay_kj_backend
    monkeypatch.setattr(relay_kj_backend, '_audited_mask_kernel', lambda candidate: False)
    owner = _PlainDelegate(None, 'sage')
    backend = owner._masked_sage_delegate()
    assert backend.masked_kernel is None
    assert backend.report()['biased_rows'] == 'pytorch_sdpa'
    assert backend.report()['source_sha256'] is None
    assert owner.report()['scoped_masked_sage']['kind'] == 'audited_sage_bias_delegate_for_core_selector'


def test_core_sage_nested_mask_counters_are_fresh_per_stage_and_deltas_are_exact():
    from h3_audio_t8_pkg.progressive_attention import _PlainDelegate, backend_phase_report, backend_snapshot
    from h3_audio_t8_pkg.progressive_relay import _fresh_delegate
    from h3_audio_t8_pkg.relay_kj_backend import KJRelayBackend
    owner = _PlainDelegate(None, 'sage')
    owner.masked_delegate = KJRelayBackend('auto', None, None)
    owner.masked_delegate.counters['pytorch:non_cuda'] = 3
    fresh = _fresh_delegate(owner)
    assert fresh.masked_delegate is not owner.masked_delegate
    assert not fresh.masked_delegate.counters
    before = backend_snapshot(fresh)
    q = torch.ones(1, 2, 3, 8)
    fresh.masked_delegate.attention(q, q, q, 2, mask=torch.zeros(3, 3), skip_reshape=True)
    delta = backend_phase_report(fresh, before)
    assert delta['scoped_masked_sage']['completed_calls'] == {'pytorch:non_cuda': 1}
    assert owner.masked_delegate.counters == {'pytorch:non_cuda': 3}
