"""Stage ownership and native AV boundary tests; no trained-media qualification."""

import copy
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_first_sample
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_progressive_sampling_runtime import conditioning, latent, tiny_model
import test_progressive_sampling_runtime as runtime_fixtures

stub_lifter = runtime_fixtures.stub_lifter


def run(model, **kwargs):
    return runtime.sample_progressive_h3(
        model, conditioning(), conditioning(), latent(), comfy.samplers.ksampler('euler'),
        native_flow_sigmas(8, 12.), upscaler_model='test', seed=19, low_evaluations=4, **kwargs)


def test_distinct_models_execute_exactly_four_steps_each(stub_lifter):
    low, high = tiny_model(), tiny_model()
    with torch.no_grad():
        next(high.model.parameters()).add_(0.4)
    seen = []
    handles = [m.model.diffusion_model.register_forward_pre_hook(
        lambda module, args, stage=stage: seen.append(stage)) for stage, m in [('low', low), ('high', high)]]
    try:
        _, report = run(low, model_hires=high)
    finally:
        for handle in handles:
            handle.remove()
    assert seen == ['low'] * 4 + ['high'] * 4
    data = json.loads(report)
    assert data['stage_models']['separate_input'] is True
    assert data['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    assert not low.wrappers and not high.wrappers


def test_explicit_same_model_is_identical_to_legacy(stub_lifter):
    model = tiny_model()
    legacy, _ = run(model)
    explicit, _ = run(model, model_hires=model)
    for left, right in zip(legacy['samples'].unbind(), explicit['samples'].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('field,value', [('audio_shift', 4.), ('shift', 8.), ('noise_scale', 1.5)])
def test_incompatible_stage_clocks_fail_before_lift(stub_lifter, field, value):
    low, high = tiny_model(), tiny_model()
    setattr(high.model.model_sampling, field, value)
    with pytest.raises(ValueError, match='stage|shift mismatch'):
        run(low, model_hires=high)
    assert not stub_lifter


def test_changed_architecture_fails_before_sampling(stub_lifter):
    low, high = tiny_model(), tiny_model()
    high.model.diffusion_model.register_parameter('unexpected_stage_parameter', torch.nn.Parameter(torch.zeros(1)))
    with pytest.raises(ValueError, match='architecture'):
        run(low, model_hires=high)
    assert not stub_lifter


def test_high_stage_cancellation_preserves_both_inputs(stub_lifter):
    low, high = tiny_model(), tiny_model()
    before = [copy.deepcopy(m.model_options) for m in (low, high)]

    def cancel(step, *args):
        if step == 4:
            raise RuntimeError('cancel high')

    with pytest.raises(RuntimeError, match='cancel high'):
        run(low, model_hires=high, callback=cancel)
    assert [m.model_options for m in (low, high)] == before
    assert not low.wrappers and not high.wrappers
    run(low, model_hires=high)


def test_independent_stacked_weight_patches_survive_and_affect_high_stage(stub_lifter):
    low, high = tiny_model(), tiny_model()
    parameter = next(iter(dict(high.model.named_parameters())))
    shape = dict(high.model.named_parameters())[parameter].shape
    first = high.add_patches({parameter: ('diff', (torch.full(shape, 0.01),))}, 0.7)
    second = high.add_patches({parameter: ('diff', (torch.full(shape, 0.02),))}, 0.2)
    assert first and second
    descriptors = {key: list(value) for key, value in high.patches.items()}
    baseline, _ = run(low)
    patched, report = run(low, model_hires=high)
    assert high.patches == descriptors and not low.patches
    assert json.loads(report)['stage_models']['high']['weight_patch_entries'] == 2
    assert any(not torch.equal(a, b) for a, b in zip(baseline['samples'].unbind(), patched['samples'].unbind()))


def test_mean_preserving_reference_resize_keeps_high_original():
    video, audio = latent()['samples'].unbind()
    plan = plan_progressive_first_sample(video, audio, native_flow_sigmas(8, 12.), low_evaluations=4)
    plan = type(plan)(**{**plan.__dict__, 'task': 'i2va', 'low_width': 32, 'low_height': 32})
    source = conditioning('i2va')
    ref = torch.zeros(1, 24, 1, 4, 8)
    ref[..., 0, 0] = (torch.arange(24, dtype=torch.float32) + 1).reshape(1, 24, 1)
    source[0][1]['minimax_keyframes'][0]['latent'] = ref
    before = ref.clone()
    low, high = runtime.prepare_stage_conditioning(source, plan, positive=True, guide_resize='preserve_mean')
    small = low[0][1]['minimax_keyframes'][0]['latent']
    torch.testing.assert_close(small.mean((-2, -1)), ref.mean((-2, -1)), rtol=1e-6, atol=1e-6)
    assert torch.equal(ref, before)
    assert high[0][1]['minimax_keyframes'][0]['latent'] is ref


def test_unknown_resize_policy_fails_before_sampling(stub_lifter):
    with pytest.raises(ValueError, match='guide_resize'):
        run(tiny_model(), guide_resize='guess')
    assert not stub_lifter
