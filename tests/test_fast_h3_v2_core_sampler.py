"""Real tiny Core AV plumbing; random weights do not qualify trained media/VSA."""
import json

import pytest
import torch

import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.ops
import comfy.supported_models_base
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg.h3_memory_advanced import configure_chunk_feed_forward, configure_low_vram_attention
from test_progressive_sampling_runtime import conditioning, latent


def model():
    class Config(comfy.supported_models_base.BASE):
        latent_format = comfy.latent_formats.MiniMaxH3AV
        unet_extra_config = {}
        sampling_settings = {'shift': 12., 'audio_shift': 3.}
        custom_operations = comfy.ops.disable_weight_init
    config = Config(dict(hidden_size=24, num_layers=1, token_refiner_num_layers=0,
        num_attention_heads=3, attention_head_dim=128, ffn_hidden_size=32,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=24,
        time_embed_dim=24, rope_inv_freq_len=16, gate_compress=True, dtype=torch.float32))
    base = comfy.model_base.MiniMaxH3(config, device=torch.device('cpu'))
    generator = torch.Generator().manual_seed(26091603)
    with torch.no_grad():
        for value in base.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * .03)
        base.diffusion_model.rope.inv_freq.fill_(1.)
    return comfy.model_patcher.ModelPatcher(base, torch.device('cpu'), torch.device('cpu'))


def sample(prepared, sampler, sigmas, source):
    noise = RandomNoise.execute(26091604).result[0]
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result


@pytest.mark.parametrize('profile', v2.PROFILES)
@pytest.mark.parametrize('memory', [False, True])
def test_real_core_eight_step_av_runs_and_does_not_mutate_bare_model(profile, memory):
    bare, source = model(), latent()
    branch = bare
    if memory:
        branch = configure_low_vram_attention(branch, 2)[0]
        branch = configure_chunk_feed_forward(branch, 2, 256)[0]
    prepared, sampler, sigmas, report = v2.build_fast_h3_v2_setup(branch, source, profile)
    output = sample(prepared, sampler, sigmas, source)[0]
    assert all(bool(torch.isfinite(value).all()) for value in output['samples'].unbind())
    assert [list(value.shape) for value in output['samples'].unbind()] == [list(value.shape) for value in source['samples'].unbind()]
    assert json.loads(report)['nfe'] == 8
    receipt = v2.capture_fast_h3_v2_owner(prepared)
    if profile != 'dense_compat_exp':
        assert receipt.runtime.counts['dense'] == 8
        assert receipt.runtime.counts['vsa'] == 0  # CPU never claims sparse CUDA dispatch.
    assert not bare.wrappers and not bare.object_patches
    assert v2.capture_fast_h3_v2_owner(bare) is None
    assert not torch.cuda.is_initialized()


def test_real_core_partial_four_then_four_continues_unfinished_audio_and_terminal_zero():
    bare, source = model(), latent()
    low, low_sampler, ladder, _ = v2.build_fast_h3_v2_setup(bare, source, 'dense_compat_exp')
    low_sampler.extra_options.update(stage_start=0, stage_end=4)
    noisy, low_x0 = sample(low, low_sampler, ladder[:5], source)
    assert not torch.equal(noisy['samples'].unbind()[1], low_x0['samples'].unbind()[1])
    high, high_sampler, high_ladder, _ = v2.build_fast_h3_v2_setup(bare, low_x0, 'dense_compat_exp')
    high_sampler.extra_options.update(stage_start=4, stage_end=8)
    final = sample(high, high_sampler, high_ladder[4:], low_x0)[0]
    audio = final['samples'].unbind()[1]
    assert bool(torch.isfinite(audio).all())
    assert not torch.equal(audio, low_x0['samples'].unbind()[1])
    assert high_ladder[-1] == 0
    assert not bare.object_patches


@pytest.mark.parametrize('profile', ['trained_vsa_exp', 'dense_compat_exp'])
def test_actual_core_eight_network_forward_observer_with_v2_owner(monkeypatch, profile):
    from tools import progressive_probe_extension as extension
    from tools import fast_h3_v2_backend_audit
    from h3_audio_t8_pkg import sampling
    from h3_audio_t8_pkg import relay_sol_backend
    modules = {'fast_h3_v2_advanced': v2, 'sampling': sampling,
        'tools.fast_h3_v2_backend_audit': fast_h3_v2_backend_audit,
        'relay_sol_backend': relay_sol_backend}
    monkeypatch.setattr(extension, 'project_module', modules.__getitem__)
    bare, source = model(), latent()
    prepared, sampler, sigmas, _ = v2.build_fast_h3_v2_setup(bare, source, profile)
    output, raw = extension.FastH3V2SamplerProbe.execute(prepared, conditioning(), source, sampler, sigmas, 31).result
    assert json.loads(raw)['completed_network_forwards'] == 8
    assert json.loads(raw)['attempted_network_forwards'] == 8
    assert json.loads(raw)['output_finite'] is True
    assert json.loads(raw)['backend_calls']['status'] == 'not_observed_by_selector_probe'
    assert bool(torch.isfinite(output['samples'].unbind()[1]).all())
    assert not bare.wrappers
