"""Standalone MODEL patch with actual tiny native H3 sampler and owner guards."""

import copy
from collections import Counter

import comfy.sample
import comfy.samplers
import pytest
import torch
from comfy.patcher_extension import WrappersMP

from h3_audio_t8_pkg.tst_model import build_tst_model, TST_MODEL_KEY
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from h3_audio_t8_pkg.progressive_sampling_runtime import _native_stage
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_first_sample
from h3_audio_t8_pkg.progressive_eav import prepare_progressive_eav
from test_progressive_sampling_runtime import tiny_model, conditioning, latent
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401
from test_progressive_relay import paired
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj

memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def sample(model, positive=None, *, sigmas=None, callback=None, sampler=None, cfg=1.):
    source = latent()['samples']
    result = _native_stage(model, sampler or comfy.samplers.ksampler('euler'),
        native_flow_sigmas(8, 12.) if sigmas is None else sigmas,
        source, comfy.sample.prepare_noise(source, 19), positive or conditioning(), conditioning(),
        cfg, 19, callback)
    return tuple(result.unbind())


def prepare(base, *, relay=False, eav=False):
    positive = conditioning()
    if relay:
        base, positive, _ = paired(base)
    if eav:
        sigmas = native_flow_sigmas(8, 12.)
        plan = plan_progressive_first_sample(*latent()['samples'].unbind(), sigmas, low_evaluations=4)
        base, _ = prepare_progressive_eav(base, sigmas, plan, mode='apply_exp', tau=.2,
            start=0., end=1., workspace=32, hard_limit=1.5,
            relay_report={'completed_calls': Counter()} if relay else None)
    return base, positive


@pytest.mark.parametrize('relay,eav', [(False, False), (True, False), (False, True), (True, True)])
def test_standalone_report_only_identity_and_apply_native_layers(relay, eav):
    model, positive = prepare(tiny_model(), relay=relay, eav=eav)
    options = copy.deepcopy(model.model_options)
    baseline = sample(model, positive)
    reporting, configured = build_tst_model(model, native_flow_sigmas(8, 12.), mode='report_only')
    assert configured['status'] == 'configured_not_executed'
    control = sample(reporting, positive)
    assert all(torch.equal(a, b) for a, b in zip(baseline, control))
    applied, _ = build_tst_model(model, native_flow_sigmas(8, 12.), mode='apply_exp')
    output = sample(applied, positive)
    assert all(torch.isfinite(part).all() for part in output)
    state = applied.get_attachment(TST_MODEL_KEY)
    assert state.active is None and state.last_report['completed']
    assert [f['step_index'] for f in state.last_report['forwards']] == list(range(8))
    assert all(f['query_transform_calls'] == 1 for f in state.last_report['forwards'])
    assert state.last_report['forwards'][0]['applied_layers'] == 1
    assert state.last_report['forwards'][-1]['applied_layers'] == 0
    assert model.model_options == options


@pytest.mark.parametrize('backend', ['kj_memory', 'sol'])
def test_standalone_installed_backends_and_repeated_sampler_use(memory_nodes, installed_sol, backend):
    model = tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
    else:
        model = installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
    patched, _ = build_tst_model(model, native_flow_sigmas(8, 12.), mode='apply_exp')
    first, second = sample(patched), sample(patched)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert len(patched.get_attachment(TST_MODEL_KEY).last_report['forwards']) == 8


def test_partial_invocation_uses_full_indices_and_cancellation_clears_state():
    sigmas = native_flow_sigmas(8, 12.)
    model, _ = build_tst_model(tiny_model(), sigmas, mode='apply_exp')
    def cancel(*args):
        raise InterruptedError('cancel')
    with pytest.raises(InterruptedError):
        sample(model, callback=cancel)
    state = model.get_attachment(TST_MODEL_KEY)
    assert state.active is None and not state.last_report['completed']
    sample(model, sigmas=sigmas[4:])
    assert [f['step_index'] for f in state.last_report['forwards']] == [4, 5, 6, 7]


@pytest.mark.parametrize('fault', ['cfg', 'sampler', 'sigmas', 'override', 'wrapper'])
def test_runtime_rejects_wrong_sampler_or_owner(fault):
    model, _ = build_tst_model(tiny_model(), native_flow_sigmas(8, 12.), mode='apply_exp')
    kwargs = {}
    if fault == 'cfg':
        kwargs['cfg'] = 2.
    elif fault == 'sampler':
        kwargs['sampler'] = comfy.samplers.ksampler('heun')
    elif fault == 'sigmas':
        kwargs['sigmas'] = native_flow_sigmas(4, 12.)
    elif fault == 'override':
        model = model.clone()
        model.model_options['transformer_options']['optimized_attention_override'] = lambda *a, **k: None
    else:
        model = model.clone()
        model.add_wrapper_with_key(WrappersMP.APPLY_MODEL, 'foreign', lambda executor, *a, **k: executor(*a, **k))
    with pytest.raises((ValueError, RuntimeError)):
        sample(model, **kwargs)


def test_disabled_is_identity_and_double_install_refused():
    sentinel = object()
    assert build_tst_model(sentinel, None)[0] is sentinel
    model, _ = build_tst_model(tiny_model(), native_flow_sigmas(8, 12.), mode='apply_exp')
    with pytest.raises(ValueError, match='already installed'):
        build_tst_model(model, native_flow_sigmas(8, 12.), mode='apply_exp')


def test_detach_preserves_real_lora_added_after_configuration():
    from h3_audio_t8_pkg.tst_model import detach_tst_model
    from test_progressive_continuation_lora import stack
    source = tiny_model()
    model, _ = build_tst_model(source, native_flow_sigmas(8, 12.), mode='apply_exp')
    stack(model, 10)
    base, spec = detach_tst_model(model)
    assert base.model_options == source.model_options
    assert base.patches == model.patches and base.patches
    assert not base.wrappers and base.get_attachment(TST_MODEL_KEY) is None
    assert spec['mode'] == 'apply_exp'
    spec['tau'] = 7
    assert model.get_attachment(TST_MODEL_KEY).spec['tau'] == .2


@pytest.mark.parametrize('which', ['both', 'low', 'high'])
def test_progressive_consumes_public_model_config_per_phase(which, request):
    request.getfixturevalue('stub_lifter')
    from test_progressive_tst import run
    low, high = tiny_model(), tiny_model()
    if which in ('both', 'low'):
        low, _ = build_tst_model(low, native_flow_sigmas(8, 12.), mode='apply_exp')
    if which in ('both', 'high'):
        high, _ = build_tst_model(high, native_flow_sigmas(8, 12.), mode='report_only')
    _, text = run(low, model_hires=high, eav_mode='apply_exp')
    import json
    report = json.loads(text)
    for phase in ('low', 'high'):
        if which == 'both' or which == phase:
            assert report['tst'][phase]['completed']
            assert report['tst'][phase]['config']['mode'] == ('apply_exp' if phase == 'low' else 'report_only')
        else:
            assert report['tst'][phase]['identity']


def test_node_schema_config_report_and_append_only_registration():
    import asyncio
    import json
    from h3_audio_t8_pkg.nodes_tst import MiniMaxH3TSTModelEXPT8
    from h3_audio_t8_pkg.nodes import comfy_entrypoint
    cls = MiniMaxH3TSTModelEXPT8
    schema = cls.define_schema()
    assert schema.is_experimental
    assert {item.id: item for item in schema.inputs}['mode'].default == 'disabled'
    nodes = asyncio.run(comfy_entrypoint().get_node_list())
    # The two public progressive nodes append AFTER TST; no existing node moves.
    from h3_audio_t8_pkg.nodes_progressive_long_video import PROGRESSIVE_LONG_VIDEO_NODE_CLASSES
    assert nodes[-3] is cls
    assert nodes[-2:] == PROGRESSIVE_LONG_VIDEO_NODE_CLASSES
    assert nodes.count(cls) == 1
    model = tiny_model()
    output = cls.execute(model=model, sigmas=native_flow_sigmas(8, 12.), mode='disabled', tau=.2, max_workspace_mib=256)
    assert output.result[0] is model
    assert json.loads(output.result[1])['status'] == 'not_installed'


def test_progressive_model_relay_eav_low_checkpoint_reuse(tmp_path, request):
    request.getfixturevalue('stub_lifter')
    import json
    from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession
    from test_progressive_tst import run
    low, positive, _ = paired(tiny_model())
    low, _ = build_tst_model(low, native_flow_sigmas(8, 12.), mode='apply_exp')
    high, _ = build_tst_model(tiny_model(), native_flow_sigmas(8, 12.), mode='report_only')
    outputs = []
    for iteration in range(2):
        with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
            output, text = run(low, positive, model_hires=high, eav_mode='apply_exp', checkpoint=checkpoint)
        report = json.loads(text)
        assert report['checkpoint']['reused_low'] is bool(iteration)
        assert report['counts']['actual_forwards'] == {'low': 0 if iteration else 4, 'high': 4}
        assert report['tst']['high']['config']['mode'] == 'report_only'
        outputs.append(output['samples'].unbind())
    assert all(torch.equal(a, b) for a, b in zip(*outputs))


def test_wrapped_models_have_content_bound_identity():
    from h3_audio_t8_pkg.progressive_checkpoint import native_model_identity
    from h3_audio_t8_pkg.tst_model import detach_tst_model
    model, _ = build_tst_model(tiny_model(), native_flow_sigmas(8, 12.), mode='apply_exp')
    identity = native_model_identity(model, comfy.samplers.ksampler('euler'))
    assert identity['tst_model']['mode'] == 'apply_exp'
    base, _ = detach_tst_model(model)
    assert identity['source'] == native_model_identity(base, comfy.samplers.ksampler('euler'))
    model.get_attachment(TST_MODEL_KEY).spec['tau'] = .9
    with pytest.raises(RuntimeError, match='configuration changed'):
        native_model_identity(model, comfy.samplers.ksampler('euler'))
