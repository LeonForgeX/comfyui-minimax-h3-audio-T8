"""Actual tiny native H3/EAV execution, stage clocks and composition guards."""

import copy
import json
from types import SimpleNamespace

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_eav import audit_progressive_eav_stage
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_first_sample
from h3_audio_t8_pkg.sampling import native_flow_sigmas
import test_progressive_sampling_runtime as base_fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_progressive_relay import paired
from test_relay_kj_backend import kj as kj
from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8

stub_lifter = base_fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def test_progressive_eav_defaults_match_moderate_protected_window():
    inputs = {item.id: item for item in MiniMaxH3ProgressiveSamplerEXPT8.define_schema().inputs}
    assert inputs['eav_tau'].default == pytest.approx(4.)
    assert inputs['eav_start_video_progress'].default == pytest.approx(.15)
    assert inputs['eav_end_video_progress'].default == pytest.approx(.90)


def run(model, positive=None, *, steps=8, low=4, **kwargs):
    return runtime.sample_progressive_h3(
        model, positive if positive is not None else base_fixtures.conditioning(),
        base_fixtures.conditioning(), base_fixtures.latent(), comfy.samplers.ksampler('euler'),
        native_flow_sigmas(steps, 12.), upscaler_model='test', seed=19,
        low_evaluations=low, eav_tau=.2, **kwargs)


@pytest.mark.parametrize('task', ['t2va', 'i2va'])
@pytest.mark.parametrize('with_relay', [False, True])
def test_eav_report_only_identity_and_actual_apply_in_both_stages(stub_lifter, task, with_relay):
    model = base_fixtures.tiny_model()
    if with_relay:
        model, positive, _ = paired(model, task)
    else:
        positive = base_fixtures.conditioning(task)
    baseline, _ = run(model, positive, task=task)
    control, text = run(model, positive, task=task, eav_mode='report_only')
    applied, applied_text = run(model, positive, task=task, eav_mode='apply_exp')
    for a, b in zip(baseline['samples'].unbind(), control['samples'].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(applied['samples'].unbind(), control['samples'].unbind()))
    for report in (json.loads(text), json.loads(applied_text)):
        all_sigmas = []
        for phase in ('low', 'high'):
            stage = report['eav'][phase]
            assert stage['assigned_nfe'] == stage['model_forward_count'] == 4
            assert stage['attention_calls_per_active_forward'] == [1] * 4
            assert stage['g_min'] > 1.
            assert stage['config']['direct_audio_scaling'] is False
            all_sigmas.extend(f['sigma_video'] for f in stage['forwards'])
            if with_relay:
                assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 4}
        torch.testing.assert_close(torch.tensor(all_sigmas), native_flow_sigmas(8, 12.)[:-1])
        assert report['eav']['high']['forwards'][0]['progress_video'] > 0.
        assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
        assert report['eav']['summary']['full_schedule_nfe'] == 8
        assert report['eav']['summary']['output_gain_above_one_applied'] == (report['eav']['low']['config']['mode'] == 'apply_exp')


@pytest.mark.parametrize('steps,low', [(8, 6), (20, 12)])
def test_full_schedule_windows_do_not_restart_at_high_stage(stub_lifter, steps, low):
    _, text = run(base_fixtures.tiny_model(), steps=steps, low=low, eav_mode='apply_exp',
                  eav_start_video_progress=.05, eav_end_video_progress=.25)
    report = json.loads(text)
    for phase, expected in [('low', low), ('high', steps - low)]:
        assert report['eav'][phase]['model_forward_count'] == expected
        for f in report['eav'][phase]['forwards']:
            assert f['active'] == (.05 <= f['progress_video'] <= .25)


@pytest.mark.parametrize('with_relay', [False, True])
@pytest.mark.parametrize('backend', ['kj_memory', 'sol'])
def test_progressive_eav_preserves_authenticated_backend_and_lora(stub_lifter, memory_nodes, installed_sol, backend, with_relay):
    base = base_fixtures.tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(base).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
    else:
        model = installed_sol.SolAttentionPatch().patch(base, True, .5, min_tokens=256)[0]
    if with_relay:
        model, positive, _ = paired(model)
    else:
        positive = base_fixtures.conditioning()
    key, weight = next(iter(dict(model.model.named_parameters()).items()))
    model.add_patches({key: ('diff', (torch.ones_like(weight) * .1,))}, strength_patch=.3)
    model.add_patches({key: ('diff', (torch.ones_like(weight) * .2,))}, strength_patch=.2)
    _, text = run(model, positive, eav_mode='apply_exp')
    report = json.loads(text)
    for phase in ('low', 'high'):
        assert report['stage_models'][phase]['weight_patch_entries'] == 2
        assert sum(report['attention'][phase]['completed_calls'].values()) > 0
        if backend == 'sol':
            assert report['attention'][phase]['completed_calls'].get('sol:completed', 0) == 0
    assert len(model.patches[key]) == 2


def test_eav_bad_schedule_fails_before_sampling(stub_lifter, monkeypatch):
    monkeypatch.setattr(runtime, '_native_stage', lambda *a, **k: pytest.fail('unexpected sampling'))
    with pytest.raises(ValueError, match='complete8 or20'):
        run(base_fixtures.tiny_model(), steps=4, low=2, eav_mode='apply_exp')
    assert not stub_lifter


def test_inactive_window_is_reported_without_claiming_enhancement(stub_lifter):
    model = base_fixtures.tiny_model()
    baseline, _ = run(model)
    output, text = run(model, eav_mode='apply_exp', eav_start_video_progress=.99)
    for a, b in zip(baseline['samples'].unbind(), output['samples'].unbind()):
        assert torch.equal(a, b)
    summary = json.loads(text)['eav']['summary']
    assert summary['status'] == 'inactive_window_no_enhancement'
    assert summary['active_forwards'] == 0
    assert summary['output_gain_above_one_applied'] is False


def test_eav_high_cancel_reusable(stub_lifter):
    model, positive, _ = paired(base_fixtures.tiny_model())

    def cancel(step, *_):
        if step == 4:
            raise RuntimeError('cancel EAV high')

    with pytest.raises(RuntimeError, match='cancel EAV high'):
        run(model, positive, eav_mode='apply_exp', callback=cancel)
    _, text = run(model, positive, eav_mode='apply_exp')
    assert json.loads(text)['eav']['high']['model_forward_count'] == 4


@pytest.mark.parametrize('with_relay', [False, True])
def test_explicit_core_sage_is_not_lost_to_global_pytorch(stub_lifter, monkeypatch, with_relay):
    from comfy.ldm.modules import attention
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    calls = []

    def cpu_sage_fixture(q, k, v, *, tensor_layout, **kwargs):
        assert tensor_layout == 'HND'
        calls.append(kwargs.get('attn_mask') is not None)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=kwargs.get('attn_mask'))

    monkeypatch.setattr(attention, 'sageattn', cpu_sage_fixture, raising=False)
    monkeypatch.setattr(attention, 'SAGE_ATTENTION_SUPPORTS_MASK', True, raising=False)
    monkeypatch.setattr(attention, 'optimized_attention', attention.attention_pytorch)
    model = base_fixtures.tiny_model()
    set_h3_attention_backend(model, attention.attention_sage)
    if with_relay:
        model, positive, _ = paired(model)
    else:
        positive = base_fixtures.conditioning()
    _, text = run(model, positive, eav_mode='apply_exp')
    report = json.loads(text)
    assert len(calls) >= 8
    assert any(calls) is with_relay
    for phase in ('low', 'high'):
        assert report['attention'][phase]['requested_backend'] == 'sage'
        assert report['attention'][phase]['completed_calls']['delegate:completed'] >= 4


@pytest.mark.parametrize('change', ['count', 'sigma', 'blocks', 'layout'])
def test_eav_audit_warns_for_coverage_but_refuses_wrong_clock_or_layout(stub_lifter, change, caplog):
    model = base_fixtures.tiny_model()
    _, text = run(model, eav_mode='apply_exp')
    stage = copy.deepcopy(json.loads(text)['eav']['high'])
    if change == 'count':
        stage['forwards'].pop()
    elif change == 'sigma':
        stage['forwards'][0]['sigma_video'] = 1.
    elif change == 'blocks':
        stage['forwards'][0]['attention_count'] = 0
    else:
        stage['forwards'][0]['spatial_tokens'] = 1
    plan = plan_progressive_first_sample(*base_fixtures.latent()['samples'].unbind(),
        native_flow_sigmas(8, 12.), low_evaluations=4)
    if change in {'count', 'blocks'}:
        report = audit_progressive_eav_stage(SimpleNamespace(snapshot=lambda **k: stage), plan, 'high', model)
        assert report['status'] == 'executed_user_stack_unverified'
        assert report['composition_verified'] is False
        assert report['forwards'] == stage['forwards']
        assert 'advisory' in caplog.text
    else:
        with pytest.raises(RuntimeError):
            audit_progressive_eav_stage(SimpleNamespace(snapshot=lambda **k: stage), plan, 'high', model)
