"""Real native tiny H3 Relay across the resolution boundary; CPU only."""

import copy
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_progressive_sampling_runtime import conditioning, latent, tiny_model
import test_progressive_sampling_runtime as runtime_fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj
stub_lifter = runtime_fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def paired(model, task='t2va', query_route='joint_av_exp'):
    positive = conditioning(task)
    positive[0][0][0, 0] = torch.linspace(-1, 1, 8)
    positive[0][0][0, 1] = torch.linspace(1, -1, 8)
    keyframes = positive[0][1].get('minimax_keyframes', [])
    layout = relay.build_packed_layout(2, 2, 4, 8, 8, keyframes=keyframes, refs=[], frame_count=5)
    binding = relay._bind_layout_contract({
        'schema': relay.PROMPT_RELAY_PATCH_VERSION, 'plan_hash': 'tiny-two-events',
        'text_len': 2, 'query_route': query_route,
        'events': [{'text_key_start': i, 'text_key_end': i + 1, 'midpoint': float(i * 4),
                    'window': .1, 'sigma': .2} for i in range(2)],
    }, layout, resolved_task=task, keyframes=keyframes, refs=[])
    marked = [[value, {**meta, relay.PROMPT_RELAY_BINDING_KEY: binding}] for value, meta in positive]
    marked = relay._attach_binding_model_cond(marked, binding['binding_hash'])
    bound, _ = relay.patch_prompt_relay_model(model, binding, 32)
    return bound, marked, positive


def sample(model, positive, **kwargs):
    return runtime.sample_progressive_h3(
        model, positive, conditioning(), latent(), comfy.samplers.ksampler('euler'),
        native_flow_sigmas(8, 12.), upscaler_model='test', seed=19, low_evaluations=4, **kwargs)


@pytest.mark.parametrize('task', ['t2va', 'i2va'])
@pytest.mark.parametrize('query_route', ['video_only_paper', 'joint_av_exp'])
def test_relay_executes_both_stage_layouts_with_unchanged_event_times(stub_lifter, monkeypatch, task, query_route):
    base = tiny_model()
    model, positive, plain = paired(base, task, query_route)
    binding_before = copy.deepcopy(positive[0][1][relay.PROMPT_RELAY_BINDING_KEY])
    calls = []
    original = relay.route_prompt_relay_attention

    def observed(*args, **kwargs):
        route = kwargs['transformer_options'][relay.PROMPT_RELAY_RUNTIME_KEY]
        calls.append((route['binding_hash'], route['seq_len'],
                      [(part['kind'], part['query_times'].unique().tolist()) for part in route['query_segments']]))
        return original(*args, **kwargs)

    monkeypatch.setattr(relay, 'route_prompt_relay_attention', observed)
    output, text = sample(model, positive, task=task)
    report = json.loads(text)
    assert len(calls) == 8
    assert len({value[0] for value in calls[:4]}) == len({value[0] for value in calls[4:]}) == 1
    assert calls[0][0] != calls[4][0] and calls[0][1] < calls[4][1]
    assert calls[0][2] == calls[4][2]
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert report['prompt_relay']['low']['events'] == report['prompt_relay']['high']['events']
    for phase in ('low', 'high'):
        assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 4}
    assert relay.prompt_relay_model_contract(model)['binding'] == binding_before
    assert positive[0][1][relay.PROMPT_RELAY_BINDING_KEY] == binding_before
    plain_output, _ = sample(base, plain, task=task)
    assert any(not torch.equal(a, b) for a, b in zip(output['samples'].unbind(), plain_output['samples'].unbind()))
    assert not base.wrappers


def test_mismatched_pair_refused_before_lift(stub_lifter):
    model, positive, _ = paired(tiny_model())
    positive[0][1][relay.PROMPT_RELAY_BINDING_KEY] = {**positive[0][1][relay.PROMPT_RELAY_BINDING_KEY], 'plan_hash': 'other'}
    with pytest.raises(ValueError, match='not paired'):
        sample(model, positive)
    assert not stub_lifter


def test_target_layout_mismatch_rejected_before_any_sampling(stub_lifter, monkeypatch):
    model, positive, _ = paired(tiny_model())

    def forbidden(*args, **kwargs):
        pytest.fail('must reject target mismatch before native sampling')

    monkeypatch.setattr(runtime, '_native_stage', forbidden)
    larger = latent()
    larger['samples'].tensors[0] = torch.zeros(1, 24, 2, 8, 16)
    with pytest.raises(ValueError, match='target layout'):
        runtime.sample_progressive_h3(model, positive, conditioning(), larger,
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
            upscaler_model='test', seed=19, low_evaluations=4)


def test_relay_high_stage_retains_independent_lora_stack(stub_lifter):
    base, high = tiny_model(), tiny_model()
    model, positive, _ = paired(base)
    expected, _ = sample(model, positive, model_hires=high)
    key = next(iter(dict(high.model.named_parameters())))
    weight = dict(high.model.named_parameters())[key]
    high.add_patches({key: ('diff', (torch.ones_like(weight) * .2,))}, strength_patch=.5)
    high.add_patches({key: ('diff', (torch.ones_like(weight) * .1,))}, strength_patch=.3)
    actual, text = sample(model, positive, model_hires=high)
    assert json.loads(text)['stage_models']['high']['weight_patch_entries'] == 2
    assert any(not torch.equal(a, b) for a, b in zip(actual['samples'].unbind(), expected['samples'].unbind()))
    assert len(high.patches[key]) == 2


@pytest.mark.parametrize('backend', ['kj_memory', 'sol'])
def test_stage_specific_relay_preserves_installed_backends(stub_lifter, memory_nodes, installed_sol, backend):
    base = tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes
        selected = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(base).result[0]
        selected = lowmem.MiniMaxLowVRAMAttention.execute(selected, 2).result[0]
    else:
        selected = installed_sol.SolAttentionPatch().patch(base, True, .5, min_tokens=256)[0]
    model, positive, _ = paired(selected)
    _, text = sample(model, positive, model_hires=selected)
    report = json.loads(text)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    for phase in ('low', 'high'):
        assert sum(report['attention'][phase]['completed_calls'].values()) > 0
        if backend == 'sol':
            assert report['attention'][phase]['completed_calls'].get('sol:completed', 0) == 0
    # Original bound owner and unbound source remain independently usable.
    assert relay.prompt_relay_model_contract(model)
    assert not selected.wrappers


def test_high_cancellation_leaves_original_relay_reusable(stub_lifter):
    model, positive, _ = paired(tiny_model())
    before = relay.prompt_relay_model_contract(model)['binding_hash']

    def cancel(step, *_):
        if step == 4:
            raise RuntimeError('cancel-high')

    with pytest.raises(RuntimeError, match='cancel-high'):
        sample(model, positive, callback=cancel)
    assert relay.prompt_relay_model_contract(model)['binding_hash'] == before
    _, text = sample(model, positive)
    assert json.loads(text)['counts']['actual_forwards'] == {'low': 4, 'high': 4}
