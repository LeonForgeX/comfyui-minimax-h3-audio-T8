"""Execution, delegation and cache isolation, not universal GPU qualification."""
import json
from types import MethodType

import pytest
import torch
import comfy.samplers
from comfy.ldm.modules import attention

from h3_audio_t8_pkg.patch_stack_policy import compose_dit_hook, model_identity_matches
from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
from h3_audio_t8_pkg.relay_sol_backend import capture_composed_backend
from h3_audio_t8_pkg.h3_memory_advanced import (
    configure_low_vram_attention, configure_chunk_feed_forward, inspect_t8_memory_composition,
)
from h3_audio_t8_pkg.progressive_checkpoint import native_model_identity
from test_h3_memory_advanced import _small_model
from test_tst_model import sample
from test_progressive_sampling_runtime import tiny_model
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def test_unknown_attention_delegate_retains_bias_kwargs_and_errors(caplog):
    calls = []
    def prior(original, q, k, v, heads, **kwargs):
        calls.append((original, kwargs))
        return attention.attention_pytorch(q, k, v, heads,
            **{**kwargs, '_inside_attn_wrapper': True})
    backend = capture_composed_backend(prior)
    q = torch.randn(1, 3, 7, 8)
    mask = torch.randn(7, 7)
    options = {'custom': object()}
    kwargs = dict(mask=mask, skip_reshape=True, transformer_options=options)
    actual = backend.attention(q, q, q, 3, **kwargs)
    expected = attention.attention_pytorch(q, q, q, 3,
        **{**kwargs, '_inside_attn_wrapper': True})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls[0][1]['mask'] is mask
    assert calls[0][1]['transformer_options'] is options
    assert backend.report()['completed_calls'] == {'delegate:completed': 1}
    assert backend.report()['kernel_verified'] is False
    assert 'advisory' in caplog.text
    def broken(*args, **kwargs):
        raise RuntimeError('actual kernel failure')
    broken_backend = capture_composed_backend(broken)
    with pytest.raises(RuntimeError, match='actual kernel failure'):
        broken_backend.attention(q, q, q, 3, **kwargs)
    assert broken_backend.report()['completed_calls'] == {}


def test_dit_composer_preserves_prior_owner_and_native_equation():
    calls = []
    def current(args, extra):
        calls.append('current')
        return extra['original_block'](args)
    def prior(args, extra):
        calls.append('prior')
        return extra['original_block']({**args, 'x': args['x'] + 2})
    original = lambda args: args['x'] * 3
    assert compose_dit_hook(None, current, 'test') is current
    composed = compose_dit_hook(prior, current, 'test')
    assert composed({'x': 1}, {'original_block': original}) == 9
    assert calls == ['prior', 'current']
    with pytest.raises(TypeError, match='callable'):
        compose_dit_hook(object(), current, 'test')


def test_unknown_stack_executes_real_tiny_sampler_and_never_shares_cache_identity():
    model = tiny_model()
    calls = []
    def prior(original, *args, **kwargs):
        calls.append('attention')
        return original(*args, **{**kwargs, '_inside_attn_wrapper': True})
    model.model_options['transformer_options']['optimized_attention_override'] = prior
    first = stage_model_identity(model)
    second = stage_model_identity(model)
    assert first['portable_cache_reuse'] is False
    assert first['sha256'] != second['sha256']
    assert model_identity_matches(json.loads(json.dumps(first)), second)
    output = sample(model)
    assert calls and all(torch.isfinite(item).all() for item in output)
    third = stage_model_identity(model)
    assert model_identity_matches(first, third)
    model.model_options['transformer_options']['optimized_attention_override'] = lambda *a, **k: None
    assert not model_identity_matches(first, stage_model_identity(model))


def test_unknown_stack_lora_bytes_and_nan_still_checked():
    model = _small_model(1)
    model.add_wrapper_with_key('diffusion_model', 'foreign', lambda executor, *a, **k: executor(*a, **k))
    weights = torch.ones(1)
    model.patches['diffusion_model.test'] = [(1., ('diff', (weights,)), 1., None, None)]
    first = stage_model_identity(model)
    weights[0] = 2.
    assert not model_identity_matches(first, stage_model_identity(model))
    weights[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        stage_model_identity(model)


def test_memory_attention_keeps_foreign_owner_instead_of_replacing_or_blocking():
    model = _small_model(1)
    owner = model.model.diffusion_model.blocks[0].attn
    calls = []
    def prior(self, x, **kwargs):
        calls.append(len(x))
        return x * 2
    method = MethodType(prior, owner)
    path = 'diffusion_model.blocks.0.attn.forward'
    model.add_object_patch(path, method)
    patched, report = configure_low_vram_attention(model, 2)
    assert patched.object_patches[path] is method
    assert report['preserved_foreign_blocks'] == 1
    assert report['installed_attention_blocks'] == 0
    patched.patch_model(load_weights=False)
    try:
        x = torch.randn(9, 24)
        torch.testing.assert_close(owner(x), x * 2, rtol=0, atol=0)
    finally:
        patched.unpatch_model(unpatch_weights=False)
    assert calls == [9]
    assert inspect_t8_memory_composition(patched) is None


def test_later_callable_memory_owner_is_allowed_but_not_cache_authenticated():
    model, _ = configure_chunk_feed_forward(_small_model(1), 2, 256)
    path = 'diffusion_model.blocks.0.mlp.forward'
    model.object_patches[path] = lambda x: x
    assert inspect_t8_memory_composition(model) is None
    assert stage_model_identity(model)['portable_cache_reuse'] is False
    model.object_patches[path] = object()
    with pytest.raises(TypeError, match='not callable'):
        inspect_t8_memory_composition(model)


def test_progressive_checkpoint_unknown_stack_within_run_check_keeps_nonce_out_of_reuse():
    model = tiny_model()
    model.add_wrapper_with_key('diffusion_model', 'foreign', lambda executor, *a, **k: executor(*a, **k))
    sampler = comfy.samplers.ksampler('euler')
    first = native_model_identity(model, sampler)
    second = native_model_identity(model, sampler)
    assert first != second
    assert model_identity_matches(first, second)


def test_progressive_actual_prior_wrapper_is_delegated_and_default_math_unchanged(request):
    request.getfixturevalue('stub_lifter')
    from test_progressive_sampling_runtime import conditioning, latent
    from h3_audio_t8_pkg.progressive_sampling_runtime import sample_progressive_h3
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    model = tiny_model()
    sampler = comfy.samplers.ksampler('euler')
    kwargs = dict(upscaler_model='test', seed=1, low_evaluations=2)
    baseline, _ = sample_progressive_h3(model, conditioning(), conditioning(), latent(), sampler,
                                      native_flow_sigmas(4, 12.), **kwargs)
    calls = []
    def prior(apply, options):
        calls.append(tuple(options['input'].shape))
        return apply(options['input'], options['timestep'], **options['c'])
    model.model_options['model_function_wrapper'] = prior
    output, text = sample_progressive_h3(model, conditioning(), conditioning(), latent(), sampler,
                                        native_flow_sigmas(4, 12.), **kwargs)
    assert len(calls) == 4
    assert json.loads(text)['counts']['apply_model_calls'] == {'low': 2, 'high': 2}
    assert model.model_options['model_function_wrapper'] is prior
    assert all(torch.equal(a, b) for a, b in zip(output['samples'].unbind(), baseline['samples'].unbind()))


def test_dynamic_wrapper_counts_real_delegation_and_keeps_kernel_errors():
    from h3_audio_t8_pkg.dynamic_guidance_advanced import DynamicGuidanceRuntime
    runtime = DynamicGuidanceRuntime({})
    calls = []
    def prior(apply, options):
        calls.append('prior')
        return apply(options['input'], options['timestep'], **options['c'])
    runtime.prior_model_wrapper = prior
    options = dict(input=torch.tensor(2.), timestep=torch.tensor(1.), c={}, cond_or_uncond=[0])
    assert runtime.model_function_wrapper(lambda x, t: x + t, options).item() == 3.
    assert calls == ['prior'] and runtime.physical_model_forward_calls == 1
    def broken(*a, **k):
        raise RuntimeError('real kernel error')
    with pytest.raises(RuntimeError, match='real kernel error'):
        runtime.model_function_wrapper(broken, options)


def test_tst_detach_preserves_later_attention_hooks_wrappers_and_real_sampler():
    from h3_audio_t8_pkg.tst_model import build_tst_model, detach_tst_model, TST_MODEL_KEY
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    source = tiny_model()
    model, _ = build_tst_model(source, native_flow_sigmas(8, 12.), mode='apply_exp')
    calls = []
    def prior(original, *args, **kwargs):
        calls.append('attention')
        return original(*args, **{**kwargs, '_inside_attn_wrapper': True})
    def wrapper(executor, *args, **kwargs):
        calls.append('wrapper')
        return executor(*args, **kwargs)
    model.model_options['transformer_options']['optimized_attention_override'] = prior
    model.add_wrapper_with_key('diffusion_model', 'later-user', wrapper)
    detached, _ = detach_tst_model(model)
    assert detached.get_attachment(TST_MODEL_KEY) is None
    assert detached.model_options['transformer_options']['optimized_attention_override'] is prior
    assert detached.get_wrappers('diffusion_model', 'later-user') == [wrapper]
    result = sample(detached)
    assert 'attention' in calls and 'wrapper' in calls
    assert all(torch.isfinite(value).all() for value in result)


@pytest.mark.parametrize('boolean', [False, True])
def test_relay_preserves_prior_mask_and_temporal_bias_using_real_sdpa(boolean):
    from h3_audio_t8_pkg.prompt_relay_advanced import route_prompt_relay_attention, PROMPT_RELAY_RUNTIME_KEY, make_prompt_relay_bias
    from h3_audio_t8_pkg.patch_stack_policy import merge_attention_bias
    q = torch.randn(1, 2, 8, 4)
    times = torch.tensor([0., 1., 2., 3.])
    # Use an empty event list to measure preservation exactly, not a mocked kernel.
    route = {'seq_len': 8, 'events': [], 'query_segments': [{'start': 4, 'end': 8, 'query_times': times}]}
    mask = torch.zeros(8, 8)
    mask[:, 1] = -1e3
    mask = mask >= 0 if boolean else mask
    calls = []
    def prior(original, *args, **kwargs):
        calls.append(kwargs['mask'])
        return attention.attention_pytorch(*args, **{**kwargs, '_inside_attn_wrapper': True})
    delegate = capture_composed_backend(prior)
    actual = route_prompt_relay_attention(q, q, q, 2, mask=mask, skip_reshape=True,
        transformer_options={PROMPT_RELAY_RUNTIME_KEY: route}, query_chunk_rows=2, relay_backend=delegate)
    expected = attention.attention_pytorch(q, q, q, 2, mask=mask, skip_reshape=True, _inside_attn_wrapper=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert len(calls) == 3
    assert torch.equal(calls[-1], merge_attention_bias(make_prompt_relay_bias(times[-2:], 8, [], dtype=q.dtype), mask[-2:]))


def test_outpaint_keeps_live_conditioning_without_serializing_execution_objects(tmp_path):
    from h3_audio_t8_pkg.video_outpaint_conditioning import prepare_outpaint_conditioning, OutpaintConditioningProvider
    from test_video_outpaint_execution import TinyClip, _plan
    hook = object()
    class Clip(TinyClip):
        def encode_from_tokens_scheduled(self, text):
            result = super().encode_from_tokens_scheduled(text)
            result[0][1]['hooks'] = hook
            return result
    provider = prepare_outpaint_conditioning(Clip(), 'scene', _plan(), tmp_path)
    assert provider(0, 0)[0][1]['hooks'] is hook
    assert provider.portable_cache_reuse is False
    with pytest.raises(FileNotFoundError, match='Live conditioning object'):
        OutpaintConditioningProvider(tmp_path, _plan())


def test_regional_prior_attention_delegate_reaches_biased_and_nonvideo_queries(tmp_path):
    from test_video_outpaint_regional import FakePatcher, _provider
    from h3_audio_t8_pkg.video_outpaint_regional import patch_outpaint_regional_model, REGIONAL_RUNTIME_KEY
    model = FakePatcher()
    calls = []
    def prior(original, *args, **kwargs):
        calls.append(kwargs['mask'])
        return attention.attention_pytorch(*args, **{**kwargs, '_inside_attn_wrapper': True})
    model.model_options['transformer_options']['optimized_attention_override'] = prior
    patched, contract = patch_outpaint_regional_model(model, _provider(tmp_path), 128)
    assert contract['portable_cache_reuse'] is False
    override = patched.model_options['transformer_options']['optimized_attention_override']
    route = dict(seq_len=8, video_start=4, video_end=8, frame_rows=2,
        regions=[dict(text_key_start=1, text_key_end=3, allowed=torch.tensor([True, False]))])
    q = torch.randn(1, 2, 8, 4)
    result = override(attention.optimized_attention, q, q, q, 2, skip_reshape=True,
                      transformer_options={REGIONAL_RUNTIME_KEY: route})
    assert result.shape == (1, 8, 8) and torch.isfinite(result).all()
    assert len(calls) == 2 and calls[0] is None and calls[1] is not None
    assert model.model_options['transformer_options']['optimized_attention_override'] is prior
