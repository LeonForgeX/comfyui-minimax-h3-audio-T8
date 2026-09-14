"""Continuation owner, projection and backend gates; CPU fixture scope only."""

import copy
import json

import pytest
import torch

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import progressive_eav
from h3_audio_t8_pkg import progressive_continuation_relay as composer
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
from test_progressive_continuation import accepted, capture  # noqa: F401
from test_progressive_continuation_runtime import run
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_progressive_masking import memory_nodes, installed_sol, kj  # noqa: F401
from helpers import make_audio


def arguments(route='joint_av_exp'):
    plan = relay.build_prompt_relay_plan('Scene.', 'Walk.\nStop.\nTurn.', 345,
        'auto_equal', '', 'paper_v1', .1, False, False)[0]
    plan['query_route'] = route
    plan.pop('plan_hash')
    plan['plan_hash'] = relay._sha256_json(plan)
    return dict(clip=NativeLikeFakeClip(), prompt=None, prompt_relay_plan=plan,
                query_chunk_rows=64, eav_mode='apply_exp')


@pytest.mark.parametrize('fault', ['prompt', 'global_hash', 'timeline', 'chunks', 'locked_audio'])
def test_relay_preparation_rejects_bad_inputs_before_sampling(accepted, stub_lifter, fault):  # noqa: F811
    args = arguments()
    if fault == 'prompt':
        args['prompt'] = 'Not the compiled script'
    elif fault == 'global_hash':
        args['prompt_relay_plan']['events'][0]['midpoint'] += 1
    elif fault == 'timeline':
        args['accepted_end_frame'] = 123
    elif fault == 'chunks':
        args['query_chunk_rows'] = True
    else:
        args['options'] = dict(audio_mode='lock_source', drive_audio=make_audio(6),
                               add_source_as_reference=False)
    with pytest.raises((ValueError, RuntimeError)):
        run(capture(accepted), tiny_model(), **args)
    assert not stub_lifter


@pytest.mark.parametrize('fault', ['binding_hash', 'projected_plan_hash', 'accepted_source_sha256', 'segment_index'])
def test_eav_rejects_unpaired_motion_projection_before_lift(accepted, stub_lifter, monkeypatch, fault):  # noqa: F811
    original = progressive_eav.prepare_progressive_eav

    def replace(model, *args, **kwargs):
        attachment = dict(model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY))
        attachment[fault] = -1 if fault == 'segment_index' else '0' * 64
        model.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, attachment)
        return original(model, *args, **kwargs)

    monkeypatch.setattr(progressive_eav, 'prepare_progressive_eav', replace)
    with pytest.raises(ValueError, match='paired accepted-parent'):
        run(capture(accepted), tiny_model(), **arguments())
    assert not stub_lifter


def test_mutated_prepared_relay_is_rejected(accepted, stub_lifter, monkeypatch):  # noqa: F811
    original = composer.prepare_continuation_relay_stage

    def mutate(base, prepared, **kwargs):
        prepared.relay['bindings']['low']['events'][0]['midpoint'] += 1
        return original(base, prepared, **kwargs)

    monkeypatch.setattr(composer, 'prepare_continuation_relay_stage', mutate)
    with pytest.raises(ValueError, match='prepared inputs changed'):
        run(capture(accepted), tiny_model(), **arguments())
    assert not stub_lifter


def test_cancelled_relay_eav_reuse_is_bitwise_and_preserves_caller_models(accepted, stub_lifter):  # noqa: F811
    source = capture(accepted)
    first, second = tiny_model(), tiny_model()
    args = arguments()
    before = [copy.deepcopy(m.model_options) for m in (first, second)]
    baseline, _ = run(source, first, model_hires=second, **args)

    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('cancel native HIGH Relay/EAV')

    with pytest.raises(InterruptedError):
        run(source, first, model_hires=second, callback=cancel, **args)
    output, _ = run(source, first, model_hires=second, **args)
    for a, b in zip(output['samples'].unbind(), baseline['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert [m.model_options for m in (first, second)] == before
    assert not first.wrappers and not second.wrappers
    source.revalidate()


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
@pytest.mark.parametrize('eav', ['disabled', 'apply_exp'])
def test_continuation_relay_executes_both_lora_stacks_and_backend_routes(
        accepted, stub_lifter, memory_nodes, installed_sol, monkeypatch, backend, eav):  # noqa: F811
    from comfy.weight_adapter.lora import LoRAAdapter
    low, high = tiny_model(), tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes

        def patch(model):
            model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
            model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
            return lowmem.MiniMaxChunkFeedForward.execute(model, 2, 256).result[0]
    elif backend == 'sol':
        def patch(model):
            return installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
    else:
        def patch(model):
            return model
    low, high = patch(low), patch(high)
    source = capture(accepted)
    args = {**arguments(), 'eav_mode': eav, 'model_hires': high}
    baseline, _ = run(source, low, **args)
    adapters, executed = [], []
    native = LoRAAdapter.calculate_weight

    def calculate(self, *args, **kwargs):
        output = native(self, *args, **kwargs)
        executed.append(self)
        return output

    monkeypatch.setattr(LoRAAdapter, 'calculate_weight', calculate)
    for index, model in enumerate((low, high)):
        key, weight = next((k, w) for k, w in model.model.named_parameters() if w.ndim == 2)
        for magnitude, strength in ((.1, .7), (.2, .2)):
            adapter = LoRAAdapter(set(), (torch.full((weight.shape[0], 2), magnitude * (index + 1)),
                torch.full((2, weight.shape[1]), .1), 2., None, None, None))
            assert model.add_patches({key: adapter}, strength_patch=strength)
            adapters.append(adapter)
    output, text = run(source, low, **args)
    assert all(any(a is e for e in executed) for a in adapters)
    assert any(not torch.equal(a, b) for a, b in zip(output['samples'].unbind(), baseline['samples'].unbind()))
    report = json.loads(text)
    for phase in ('low', 'high'):
        assert report['stage_models'][phase]['weight_patch_entries'] == 2
        assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 4}
        if backend != 'plain':
            assert sum(report['attention'][phase]['completed_calls'].values()) > 0
        if backend == 'sol':
            assert report['attention'][phase]['completed_calls'].get('sol:completed', 0) == 0
