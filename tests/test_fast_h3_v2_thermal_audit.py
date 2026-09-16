"""Actual function calls and disposable sampler observer, strictly CPU-only."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from tools.fast_h3_v2_thermal_audit import completed_thermal_backend, observe_thermal_backend


def kernel(value=1):
    return value


def selector(value=1):
    return kernel(value)


def pytorch(value=1):
    return value


def observe(profile, *, recipe='trained_vsa_exp', head_chunks=1, kind='audited_kj_selector'):
    receipt = SimpleNamespace(profile=recipe, runtime=SimpleNamespace(head_chunks=head_chunks,
        sparse=SimpleNamespace(h3_sparse_attention=selector, ck=SimpleNamespace(sol_attn_chunked=kernel))))
    model = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': selector}})
    backend = SimpleNamespace(kernel=kernel, report=lambda: dict(kind=kind, completed_calls={}))
    def capture(model):
        return receipt if profile == 'trained_v2_dmd8' else None
    return observe_thermal_backend(model, profile, capture, lambda _: backend,
                                   SimpleNamespace(attention_pytorch=pytorch))


@pytest.mark.parametrize('profile, count', [('production_ema_b_native8', 1600), ('trained_v2_dmd8', 400)])
def test_both_profiles_use_actual_equal_selector_kernel_call_policy(profile, count):
    observer = observe(profile)
    with observer:
        for _ in range(count):
            selector()
        pytorch()
    report = completed_thermal_backend(observer)
    assert report['observed'] and report['counts']['selector']['successful_returns'] == count
    assert report['counts']['selected_kernel']['successful_returns'] == count
    assert report['counts']['pytorch']['successful_returns'] == 1
    assert sys.getprofile() is None


@pytest.mark.parametrize('change', ['empty', 'kernel_only', 'short_vsa', 'failed_return'])
def test_configured_missing_partial_or_failed_calls_never_qualify(change):
    observer = observe('trained_v2_dmd8')
    with observer:
        if change == 'kernel_only':
            kernel()
        elif change == 'short_vsa':
            selector()
        elif change == 'failed_return':
            selector(None)
    with pytest.raises(RuntimeError):
        completed_thermal_backend(observer)


@pytest.mark.parametrize('profile, kwargs', [('unknown', {}), ('trained_v2_dmd8', {'recipe': 'dense_compat_exp'}),
    ('trained_v2_dmd8', {'head_chunks': 4}), ('production_ema_b_native8', {'kind': 'untrusted'})])
def test_foreign_profiles_or_owners_are_refused(profile, kwargs):
    with pytest.raises(ValueError):
        observe(profile, **kwargs)


def test_existing_profiler_is_not_taken_over():
    def prior(*args):
        return None
    sys.setprofile(prior)
    try:
        with pytest.raises(RuntimeError, match='existing'):
            with observe('trained_v2_dmd8'):
                pass
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize('profile', ['production_ema_b_native8', 'trained_v2_dmd8'])
@pytest.mark.parametrize('outcome', ['pass', 'short', 'network_failure', 'nan'])
def test_sampler_counts_completed_calls_preserves_input_and_cleans_on_all_exits(monkeypatch, profile, outcome):
    from comfy_extras import nodes_custom_sampler as native
    from tools import progressive_probe_extension as extension
    from tools import fast_h3_v2_thermal_audit as audit

    class Model:
        def __init__(self):
            self.model_options = {'transformer_options': {'optimized_attention_override': selector}}
            self.wrappers = {}
        def clone(self):
            self.cloned = Model()
            return self.cloned
        def add_wrapper_with_key(self, role, key, value):
            self.wrappers[key] = value
        def remove_wrappers_with_key(self, role, key):
            self.wrappers.pop(key)

    model = Model()
    parts = [torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 2)]
    source = {'samples': parts}
    receipt = SimpleNamespace(profile='trained_vsa_exp', runtime=SimpleNamespace(head_chunks=1,
        sparse=SimpleNamespace(h3_sparse_attention=selector, ck=SimpleNamespace(sol_attn_chunked=kernel))))
    def capture(model):
        return receipt if profile == 'trained_v2_dmd8' else None
    validated = []
    modules = {'fast_h3_v2_advanced': SimpleNamespace(capture_fast_h3_v2_owner=capture),
        'tools.fast_h3_v2_thermal_audit': audit,
        'progressive_sampling_runtime': SimpleNamespace(validate_native_model=lambda *args: validated.append(True)),
        'relay_sol_backend': SimpleNamespace(capture_composed_backend=lambda _: SimpleNamespace(kernel=kernel,
            report=lambda: dict(kind='audited_kj_selector'))),
        'sampling': SimpleNamespace(nested_av_parts=lambda output: output['samples'])}
    monkeypatch.setattr(extension, 'project_module', modules.__getitem__)
    from comfy.ldm.modules import attention
    monkeypatch.setattr(attention, 'attention_pytorch', pytorch)
    monkeypatch.setattr(native.RandomNoise, 'execute', lambda seed: SimpleNamespace(result=[seed]))
    monkeypatch.setattr(native.BasicGuider, 'execute', lambda clone, positive: SimpleNamespace(result=[clone]))

    def execute(noise, clone, sampler, sigmas, input_latent):
        assert input_latent is source
        wrapper = next(iter(clone.wrappers.values()))
        def network(*args, **kwargs):
            if outcome == 'network_failure':
                raise RuntimeError('test_network_failure')
            for _ in range(50):
                selector()
            return parts
        for _ in range(7 if outcome == 'short' else 8):
            wrapper(network, parts)
        if outcome == 'nan':
            parts[0].fill_(float('nan'))
        return SimpleNamespace(result=[source])

    monkeypatch.setattr(native.SamplerCustomAdvanced, 'execute', execute)
    if outcome == 'pass':
        output, raw = extension.ThermalSamplerProbe.execute(model, [], source, object(), torch.zeros(9), 1, profile).result
        report = json.loads(raw)
        assert output is source and report['completed_network_forwards'] == 8
        assert report['backend_calls']['counts']['selected_kernel']['successful_returns'] == 400
    else:
        with pytest.raises(RuntimeError):
            extension.ThermalSamplerProbe.execute(model, [], source, object(), torch.zeros(9), 1, profile)
    assert validated == ([True] if profile == 'production_ema_b_native8' else [])
    assert not model.wrappers and not model.cloned.wrappers and sys.getprofile() is None
    assert not torch.cuda.is_initialized()
