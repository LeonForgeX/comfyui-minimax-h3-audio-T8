"""Actual tiny H3/Euler TST composition. Lifter is an explicit test double."""

import copy
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_progressive_relay import paired
import test_progressive_sampling_runtime as fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj

stub_lifter = fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def run(model, positive=None, **kwargs):
    return runtime.sample_progressive_h3(model,
        positive if positive is not None else fixtures.conditioning(), fixtures.conditioning(),
        fixtures.latent(), comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
        upscaler_model='test', seed=19, low_evaluations=4, eav_tau=.2, **kwargs)


def same(left, right):
    for a, b in zip(left['samples'].unbind(), right['samples'].unbind(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize('eav', [False, True])
@pytest.mark.parametrize('relay', [False, True])
def test_actual_tst_clock_identity_and_eav_relay_composition(stub_lifter, eav, relay):
    model = fixtures.tiny_model()
    # Exercise two actual DiT blocks: diagnostics must use Core block_index.
    model.model.diffusion_model.blocks.append(copy.deepcopy(model.model.diffusion_model.blocks[0]))
    positive = fixtures.conditioning()
    if relay:
        model, positive, _ = paired(model)
    options = copy.deepcopy(model.model_options)
    kwargs = dict(eav_mode='apply_exp' if eav else 'disabled')
    baseline, _ = run(model, positive, **kwargs)
    report_only, text = run(model, positive, tst_mode='report_only', **kwargs)
    same(baseline, report_only)
    output, applied_text = run(model, positive, tst_mode='apply_exp', **kwargs)
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    for text in (text, applied_text):
        report = json.loads(text)
        for phase, indices in [('low', list(range(4))), ('high', list(range(4, 8)))]:
            tst = report['tst'][phase]
            assert tst['completed'] and [f['step_index'] for f in tst['forwards']] == indices
            assert all(f['query_transform_calls'] == 2 for f in tst['forwards'])
            assert [f['sigma_video'] for f in tst['forwards']] == list(native_flow_sigmas(8, 12.).tolist()[indices[0]:indices[-1] + 1])
            if relay:
                assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 8}
            if eav:
                assert report['eav'][phase]['attention_calls_per_active_forward'] == [2] * 4
        assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert json.loads(applied_text)['tst']['low']['forwards'][0]['applied_layers'] == 2
    assert json.loads(applied_text)['tst']['high']['forwards'][-1]['applied_layers'] == 0
    assert model.model_options == options


@pytest.mark.parametrize('backend', ['kj_memory', 'sol'])
@pytest.mark.parametrize('relay_eav', [False, True])
def test_authenticated_backend_preserved(stub_lifter, memory_nodes, installed_sol, backend, relay_eav):
    model = fixtures.tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
    else:
        model = installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
    positive = fixtures.conditioning()
    if relay_eav:
        model, positive, _ = paired(model)
    before = dict(model.object_patches)
    _, text = run(model, positive, tst_mode='apply_exp', eav_mode='apply_exp' if relay_eav else 'disabled')
    report = json.loads(text)
    for phase in ('low', 'high'):
        assert report['tst'][phase]['completed']
        assert sum(report['attention'][phase]['completed_calls'].values()) > 0
        assert report['attention'][phase]['completed_calls'].get('sol:completed', 0) == 0
    assert model.object_patches == before


def test_cancel_and_low_cache_resume_preserve_full_clock(tmp_path, stub_lifter):
    model, positive, _ = paired(fixtures.tiny_model())
    before = copy.deepcopy(model.model_options)
    baseline, _ = run(model, positive, tst_mode='apply_exp', eav_mode='apply_exp')

    def cancel(step, *_):
        if step == 4:
            raise InterruptedError('cancel HIGH')

    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        with pytest.raises(InterruptedError, match='cancel HIGH'):
            run(model, positive, tst_mode='apply_exp', eav_mode='apply_exp', callback=cancel, checkpoint=checkpoint)
    callbacks = []
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        output, text = run(model, positive, tst_mode='apply_exp', eav_mode='apply_exp', checkpoint=checkpoint,
            callback=lambda step, *_: callbacks.append(step))
    same(output, baseline)
    report = json.loads(text)
    assert callbacks == [4, 5, 6, 7]
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert report['tst']['low']['execution_scope'] == 'restored_low_not_current_forwards'
    assert [f['step_index'] for f in report['tst']['high']['forwards']] == [4, 5, 6, 7]
    assert model.model_options == before


@pytest.mark.parametrize('change', [dict(tst_mode='bad'), dict(tst_tau=-1), dict(tst_max_workspace_mib=0), dict(cfg=2.)])
def test_bad_config_before_sampling(stub_lifter, monkeypatch, change):
    monkeypatch.setattr(runtime, '_native_stage', lambda *a, **k: pytest.fail('invalid TST config sampled'))
    args = dict(tst_mode='apply_exp')
    args.update(change)
    with pytest.raises(ValueError):
        run(fixtures.tiny_model(), **args)
    assert not stub_lifter


def test_tst_settings_are_bound_to_job_identity(stub_lifter):
    from test_progressive_job import job
    baseline = job()
    candidate = job(sampling_options=dict(tst_mode='apply_exp', tst_tau=.2, tst_max_workspace_mib=128))
    assert candidate.sha256 != baseline.sha256
    assert candidate.verify() == candidate.sha256
    candidate.options['tst_tau'] = .3
    with pytest.raises(ValueError):
        candidate.verify()


def test_changed_tst_settings_do_not_reuse_low_cache(tmp_path, stub_lifter):
    model = fixtures.tiny_model()
    for tau in (.2, .3):
        with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
            _, text = run(model, tst_mode='apply_exp', tst_tau=tau, checkpoint=checkpoint)
        assert not json.loads(text)['checkpoint']['reused_low']
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 2
