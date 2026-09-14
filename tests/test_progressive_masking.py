"""Native CPU H3 mask oracles. No trained lift, CUDA or media qualification."""

import copy
import json

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import pytest
import torch

from h3_audio_t8_pkg.progressive_masking import (
    normalize_av_masks, resize_video_source, sampler_with_clean_anchor,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_initialized_sample
import test_progressive_sampling_runtime as fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj

stub_lifter = fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def nested(parts):
    return comfy.nested_tensor.NestedTensor(parts)


def source():
    generator = torch.Generator().manual_seed(23)
    return nested([torch.randn(t.shape, generator=generator) * .2 for t in fixtures.latent()['samples'].unbind()])


def run(model, sampler, initial, noise, mask, *, sigmas=None, callback=None):
    return comfy.samplers.sample(
        model, noise, fixtures.conditioning(), fixtures.conditioning(), 1., model.load_device,
        sampler, native_flow_sigmas(8, 12.)[4:] if sigmas is None else sigmas,
        model.model_options, latent_image=initial, denoise_mask=mask,
        callback=callback, disable_pbar=True, seed=3)


@pytest.mark.parametrize('rank', [2, 3, 4, 5])
def test_video_mask_spatial_resize_matches_core_without_changing_audio(rank):
    video, audio = source().unbind()
    mask = torch.tensor([[0., .2, .8], [1., .7, .4]])
    for _ in range(rank - 2):
        mask = mask.unsqueeze(0)
    before = mask.clone()
    vm, am = normalize_av_masks(mask, video, audio).unbind()
    expected = comfy.utils.reshape_mask(mask, video.shape)
    torch.testing.assert_close(vm, expected, atol=0, rtol=0)
    assert torch.equal(am, torch.ones_like(audio))
    assert torch.equal(mask, before)


def test_temporal_and_stereo_masks_do_not_mix_time_or_stereo_rows():
    video, audio = source().unbind()
    vm = torch.tensor([0., 1.]).reshape(2, 1, 1)
    am = torch.linspace(0, 1, 16).reshape(1, 2, 8)
    video_mask, audio_mask = normalize_av_masks(nested((vm, am)), video, audio).unbind()
    assert torch.count_nonzero(video_mask[:, :, 0]) == 0
    assert torch.equal(video_mask[:, :, 1], torch.ones_like(video[:, :, 1]))
    for channel in range(32):
        assert torch.equal(audio_mask[:, channel], am)
    normalized = normalize_av_masks(nested((video_mask, audio_mask)), video, audio)
    for a, b in zip(normalized.unbind(), (video_mask, audio_mask)):
        assert torch.equal(a, b)


@pytest.mark.parametrize('bad', [torch.ones(3, 4, 8), torch.ones(2, 2, 4, 8),
    torch.arange(24.).reshape(1, 24, 1, 1, 1) / 24, torch.ones(1), torch.tensor([[float('nan')]]),
    torch.tensor([[-.01]]), torch.tensor([[1.01]]), torch.ones(2, 2, dtype=torch.int64)])
def test_bad_masks_rejected(bad):
    with pytest.raises(ValueError):
        normalize_av_masks(bad, *source().unbind())


def test_audio_time_interpolation_and_extra_streams_rejected():
    video, audio = source().unbind()
    for parts in ((torch.ones(2, 2), torch.ones(1, 1, 2, 7)),
                  (torch.ones(2, 2), torch.ones(1, 1, 2, 8), torch.ones(1))):
        with pytest.raises(ValueError):
            normalize_av_masks(nested(parts), video, audio)


def test_video_source_resize_preserves_distinct_frame_values_and_dtype():
    video, _ = source().unbind()
    video[:, :, 0] = 2.
    video[:, :, 1] = -3.
    low = resize_video_source(video.to(torch.float64), 2, 4)
    assert low.shape == (1, 24, 2, 2, 4)
    assert low.dtype == torch.float64
    assert torch.equal(low[:, :, 0], torch.full_like(low[:, :, 0], 2.))
    assert torch.equal(low[:, :, 1], torch.full_like(low[:, :, 1], -3.))


@pytest.mark.parametrize('kind', ['zero', 'one', 'mixed', 'fractional'])
def test_explicit_anchor_matches_unchanged_native_core_for_fresh_sampling(kind):
    model = fixtures.tiny_model()
    clean = source()
    video, audio = clean.unbind()
    generator = torch.Generator().manual_seed(5)
    vm, am = (torch.rand(video.shape, generator=generator)[:, :1],
              torch.rand(audio.shape, generator=generator)[:, :1])
    if kind in ('zero', 'one'):
        vm.fill_(float(kind == 'one'))
        am.fill_(float(kind == 'one'))
    elif kind == 'mixed':
        vm = (vm > .5).float()
        am = (am > .5).float()
    masks = normalize_av_masks(nested((vm, am)), video, audio)
    noise = comfy.sample.prepare_noise(clean, 8)
    sampler = comfy.samplers.ksampler('euler')
    expected = run(model, sampler, clean, noise, masks)
    actual = run(model, sampler_with_clean_anchor(sampler, clean, noise), clean, noise, masks)
    for a, b in zip(actual.unbind(), expected.unbind()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['zero', 'one', 'mixed', 'fractional'])
def test_restart_keeps_clean_anchor_not_the_noisy_restart_and_passes_native_masks(kind):
    model = fixtures.tiny_model()
    clean = source()
    video, audio = clean.unbind()
    vm, am = torch.ones(1, 1, 2, 4, 8), torch.ones(1, 1, 2, 8)
    if kind == 'zero':
        vm.zero_()
        am.zero_()
    elif kind in ('mixed', 'fractional'):
        vm[..., :2] = 0.
        am[..., :3] = 0.
        if kind == 'fractional':
            vm[..., 2:4] = .123
            am[..., 3:5] = .321
    masks = normalize_av_masks(nested((vm, am)), video, audio)
    initial = nested([torch.full_like(p, 5.) for p in clean.unbind()])
    solver_noise = nested([torch.zeros_like(p) for p in clean.unbind()])
    cond_noise = comfy.sample.prepare_noise(clean, 9)
    seen = []

    def capture(fn, args):
        seen.append((args['input'].clone(), args['timestep'].clone(), args['c']))
        return fn(args['input'], args['timestep'], **args['c'])

    model.set_model_unet_function_wrapper(capture)
    original_options = copy.deepcopy(model.model_options)
    sampler = comfy.samplers.ksampler('euler')
    result = run(model, sampler_with_clean_anchor(sampler, clean, cond_noise), initial, solver_noise, masks)
    assert model.model_options == original_options
    assert len(seen) == 4
    for actual, expected, mask in zip(result.unbind(), clean.unbind(), masks.unbind()):
        if kind != 'one':
            torch.testing.assert_close(actual[mask == 0], expected[mask == 0], atol=1e-6, rtol=1e-6)
        if bool((mask == 1).any()):
            assert not torch.equal(actual[mask == 1], expected[mask == 1])
    # Native input injection is stronger than output pinning, including the
    # audio sigma conversion and per-patch quantization of fractional masks.
    base = model.model
    packed_mask = comfy.utils.pack_latents(masks.unbind())[0]
    packed_anchor = base.process_latent_in(comfy.utils.pack_latents(clean.unbind())[0])
    packed_initial = base.process_latent_in(comfy.utils.pack_latents(initial.unbind())[0])
    packed_noise = comfy.utils.pack_latents(cond_noise.unbind())[0]
    sigma = native_flow_sigmas(8, 12.)[4:5]
    state = base.model_sampling.noise_scaling(sigma, torch.zeros_like(packed_initial), packed_initial)
    expected_input = state * packed_mask + base.scale_latent_inpaint(
        sigma=sigma, noise=packed_noise, latent_image=packed_anchor, x=state,
        denoise_mask=packed_mask) * (1. - packed_mask)
    torch.testing.assert_close(seen[0][0], expected_input, atol=0, rtol=0)
    for name, value in base._denoise_mask_values(packed_mask, base.latent_shapes).items():
        torch.testing.assert_close(seen[0][2][name], value, atol=0, rtol=0)
    if kind == 'one':
        baseline = run(model, sampler, initial, solver_noise, masks)
        for a, b in zip(result.unbind(), baseline.unbind()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(torch.equal(p, torch.full_like(p, 5.)) for p in initial.unbind())


def test_cancel_restores_native_inpaint_fields_and_allows_reuse(monkeypatch):
    model, clean = fixtures.tiny_model(), source()
    masks = normalize_av_masks(torch.zeros(2, 2), *clean.unbind())
    initial = nested([torch.full_like(p, 2.) for p in clean.unbind()])
    solver_noise = nested([torch.zeros_like(p) for p in clean.unbind()])
    cond_noise = comfy.sample.prepare_noise(clean, 8)
    sampler = sampler_with_clean_anchor(comfy.samplers.ksampler('euler'), clean, cond_noise)
    instances = []
    original_init = comfy.samplers.KSamplerX0Inpaint.__init__

    def track(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    monkeypatch.setattr(comfy.samplers.KSamplerX0Inpaint, '__init__', track)

    def cancel(*args):
        raise RuntimeError('cancel test')

    with pytest.raises(RuntimeError, match='cancel test'):
        run(model, sampler, initial, solver_noise, masks, callback=cancel)
    assert torch.count_nonzero(instances[0].noise) == 0
    expected = model.model.process_latent_in(comfy.utils.pack_latents(initial.unbind())[0])
    torch.testing.assert_close(instances[0].latent_image, expected, atol=0, rtol=0)
    result = run(model, sampler, initial, solver_noise, masks)
    torch.testing.assert_close(result.unbind()[0], clean.unbind()[0], atol=1e-6, rtol=1e-6)
    mm.unload_all_models()  # This isolated CPU test process only.


def test_adapter_rejects_unqualified_sampler_missing_mask_and_wrong_shapes():
    clean = source()
    noise = comfy.sample.prepare_noise(clean, 3)
    for sampler in (comfy.samplers.ksampler('heun'), comfy.samplers.ksampler('euler', {'s_churn': 1.})):
        with pytest.raises(ValueError, match='native Euler'):
            sampler_with_clean_anchor(sampler, clean, noise)
    sampler = sampler_with_clean_anchor(comfy.samplers.ksampler('euler'), clean, noise)
    with pytest.raises(RuntimeError, match='denoise_mask'):
        run(fixtures.tiny_model(), sampler, clean, noise, None)
    v, a = clean.unbind()
    wrong = nested((v[..., :4], a))
    with pytest.raises(ValueError, match='match both'):
        sampler_with_clean_anchor(comfy.samplers.ksampler('euler'), clean, wrong)


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('mask_kind', ['zero', 'one', 'prefix', 'fractional', 'video_only', 'none'])
def test_progressive_initialized_av_runs_exact_native_stages(stub_lifter, partial, mask_kind):
    model, clean = fixtures.tiny_model(), source()
    video, audio = clean.unbind()
    vm, am = torch.ones(1, 1, 2, 4, 8), torch.ones(1, 1, 2, 8)
    if mask_kind == 'zero':
        vm.zero_()
        am.zero_()
    elif mask_kind in ('prefix', 'fractional', 'video_only'):
        vm[:, :, :1] = 0.
        am[..., :3] = 0.
        if mask_kind == 'fractional':
            vm[..., 1:3] = .123
            am[..., 3:5] = .321
    mask = vm if mask_kind == 'video_only' else nested((vm, am))
    if mask_kind == 'none':
        mask = None
    source_before = [p.clone() for p in clean.unbind()]
    callbacks = []
    schedule = native_flow_sigmas(8, 12.)[2:] if partial else native_flow_sigmas(8, 12.)
    output, text = runtime.sample_progressive_h3(
        model, fixtures.conditioning(), fixtures.conditioning(), {'samples': clean, 'noise_mask': mask},
        comfy.samplers.ksampler('euler'), schedule, upscaler_model='test', seed=6, low_evaluations=4,
        input_mode='initialized_av_exp', callback=lambda i, x0, x, total: callbacks.append((i, x0, total)))
    report = json.loads(text)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': len(schedule) - 5}
    assert len(callbacks) == len(schedule) - 1
    assert len(stub_lifter) == 1
    masks = normalize_av_masks(mask, video, audio)
    if masks is not None:
        for actual, expected, part_mask in zip(output['samples'].unbind(), clean.unbind(), masks.unbind()):
            if bool((part_mask == 0).any()):
                torch.testing.assert_close(actual[part_mask == 0], expected[part_mask == 0], atol=1e-6, rtol=1e-6)
    for actual, before in zip(clean.unbind(), source_before):
        assert torch.equal(actual, before)
    assert not model.wrappers
    assert report['initialization']['continuation_and_resume_qualified'] is False


def test_initialized_opt_in_does_not_change_empty_default(stub_lifter):
    model = fixtures.tiny_model()
    args = (model, fixtures.conditioning(), fixtures.conditioning(), fixtures.latent(),
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.))
    original, _ = runtime.sample_progressive_h3(*args, upscaler_model='test', seed=5, low_evaluations=4)
    initialized, _ = runtime.sample_progressive_h3(*args, upscaler_model='test', seed=5,
        low_evaluations=4, input_mode='initialized_av_exp')
    for a, b in zip(original['samples'].unbind(), initialized['samples'].unbind()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_partial_start_retains_unmasked_source_but_sigma1_erases_it(stub_lifter):
    model = fixtures.tiny_model()

    def execute(samples, sigmas):
        return runtime.sample_progressive_h3(model, fixtures.conditioning(), fixtures.conditioning(),
            {'samples': samples}, comfy.samplers.ksampler('euler'), sigmas,
            upscaler_model='test', seed=4, low_evaluations=3, input_mode='initialized_av_exp')[0]['samples']

    for partial in (False, True):
        schedule = native_flow_sigmas(8, 12.)[2:] if partial else native_flow_sigmas(8, 12.)
        initialized = execute(source(), schedule)
        empty = execute(fixtures.latent()['samples'], schedule)
        assert any(not torch.equal(a, b) for a, b in zip(initialized.unbind(), empty.unbind())) == partial


def test_masked_progressive_relay_preserves_native_anchor_and_actual_routing(stub_lifter):
    from test_progressive_relay import paired

    model, positive, _ = paired(fixtures.tiny_model(), 't2va')
    clean = source()
    vm = torch.ones(1, 1, 2, 4, 8)
    vm[:, :, 0] = 0.
    am = torch.ones(1, 1, 2, 8)
    am[..., :3] = 0.
    masks = normalize_av_masks(nested((vm, am)), *clean.unbind())
    output, text = runtime.sample_progressive_h3(model, positive, fixtures.conditioning(),
        {'samples': clean, 'noise_mask': nested((vm, am))}, comfy.samplers.ksampler('euler'),
        native_flow_sigmas(8, 12.), upscaler_model='test', seed=4, low_evaluations=4,
        input_mode='initialized_av_exp')
    for a, b, mask in zip(output['samples'].unbind(), clean.unbind(), masks.unbind()):
        torch.testing.assert_close(a[mask == 0], b[mask == 0], atol=1e-6, rtol=1e-6)
    report = json.loads(text)
    for phase in ('low', 'high'):
        assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 4}


@pytest.mark.parametrize('values', [[1.1, .5, 0.], [0., -.1, 0.], [.5, .2, .1], [.5, .5, 0.],
                                  [float('nan'), .2, 0.]])
def test_bad_initialized_ladder_rejected(values):
    with pytest.raises(ValueError):
        plan_progressive_initialized_sample(*source().unbind(), torch.tensor(values), low_evaluations=1)


def test_initialized_eav_rejects_non_native_cfg_before_sampling(stub_lifter):
    with pytest.raises(ValueError, match='CFG1'):
        runtime.sample_progressive_h3(fixtures.tiny_model(), fixtures.conditioning(), fixtures.conditioning(),
            {'samples': source()}, comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
            upscaler_model='test', seed=3, low_evaluations=4, input_mode='initialized_av_exp', eav_mode='apply_exp', cfg=2.)
    assert not stub_lifter


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
@pytest.mark.parametrize('relay', [False, True])
@pytest.mark.parametrize('eav_mode', ['disabled', 'apply_exp'])
def test_masked_distinct_models_execute_real_lora_stacks_and_keep_backends(
        stub_lifter, memory_nodes, installed_sol, monkeypatch, backend, relay, eav_mode):
    from comfy.weight_adapter.lora import LoRAAdapter
    from test_progressive_relay import paired

    low, high = fixtures.tiny_model(), fixtures.tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes

        def patch(model):
            model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
            return lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
    elif backend == 'sol':
        def patch(model):
            return installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
    else:
        def patch(model):
            return model
    low, high = patch(low), patch(high)
    if relay:
        low, positive, _ = paired(low)
    else:
        positive = fixtures.conditioning()
    source_av = source()
    vm, am = torch.ones(1, 1, 2, 4, 8), torch.ones(1, 1, 2, 8)
    vm[:, :, 0] = 0.
    am[..., :3] = 0.
    av = {'samples': source_av, 'noise_mask': nested((vm, am))}

    def execute():
        return runtime.sample_progressive_h3(low, positive, fixtures.conditioning(), av,
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
            model_hires=high, upscaler_model='test', seed=4, low_evaluations=4, input_mode='initialized_av_exp',
            eav_mode=eav_mode, eav_tau=.2)

    baseline, _ = execute()
    adapters, executed = [], []
    native_calculate = LoRAAdapter.calculate_weight

    def calculate(self, *args, **kwargs):
        result = native_calculate(self, *args, **kwargs)
        executed.append(self)
        return result

    monkeypatch.setattr(LoRAAdapter, 'calculate_weight', calculate)
    for index, model in enumerate((low, high)):
        key, weight = next((k, w) for k, w in model.model.named_parameters() if w.ndim == 2)
        for magnitude, strength in ((.1, .7), (.2, .2)):
            adapter = LoRAAdapter(set(), (
                torch.full((weight.shape[0], 2), magnitude * (index + 1)),
                torch.full((2, weight.shape[1]), .1), 2., None, None, None))
            assert model.add_patches({key: adapter}, strength_patch=strength)
            adapters.append(adapter)
    output, text = execute()
    assert all(any(a is e for e in executed) for a in adapters)
    assert any(not torch.equal(a, b) for a, b in zip(output['samples'].unbind(), baseline['samples'].unbind()))
    masks = normalize_av_masks(av['noise_mask'], *source_av.unbind())
    for actual, expected, mask in zip(output['samples'].unbind(), source_av.unbind(), masks.unbind()):
        torch.testing.assert_close(actual[mask == 0], expected[mask == 0], atol=1e-6, rtol=1e-6)
    report = json.loads(text)
    for phase in ('low', 'high'):
        assert report['stage_models'][phase]['weight_patch_entries'] == 2
        if backend != 'plain':
            assert sum(report['attention'][phase]['completed_calls'].values()) > 0
        if backend == 'sol':
            assert report['attention'][phase]['completed_calls'].get('sol:completed', 0) == 0
