from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg import sampling


class _Injected:
    def __init__(self, noise, scale=1.0):
        self.noise = noise
        self.latent_image = torch.ones_like(noise) * 3.0
        self.inputs, self.clocks = [], []
        self.inner_model = SimpleNamespace(inner_model=SimpleNamespace(
            audio_scale=lambda: 1.0, model_sampling=SimpleNamespace(noise_scale=scale)))

    def __call__(self, x, sigma, **kwargs):
        self.inputs.append(x.clone())
        self.clocks.append((sigma.item(), sampling.time_shift_sigma(sigma, 10., 3.).item()))
        video_v = 1.0 + len(self.inputs) * 0.1
        audio_v = 2.0 + len(self.inputs) * 0.2
        velocity = torch.tensor([video_v, video_v, audio_v, audio_v], dtype=x.dtype).view(1, 1, 4)
        return x - velocity * sigma.view(1, 1, 1)


def test_exact_dmd_contract_and_no_uniform_substitution():
    assert v2.RUNG_STEPS == (999, 874, 749, 624, 500, 375, 250, 125)
    assert len(v2.dmd_sigmas()) == 9 and v2.dmd_sigmas()[-1].item() == 0
    assert v2.dmd_sigmas()[0] < 1
    assert not torch.equal(v2.dmd_sigmas(), sampling.native_flow_sigmas(8, 10))
    av = sampling.time_shift_sigma(v2.dmd_sigmas(), 10., 3.)
    assert torch.allclose(av, v2.dmd_sigmas(3), rtol=0, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1., 1.7])
def test_first_raw_noise_and_eight_step_independent_av_oracle(dtype, scale):
    noise = torch.tensor([1., -2., 3., -4.]).view(1, 1, 4).to(dtype)
    model = _Injected(noise, scale)
    callbacks = []
    output = v2.sample_v2_euler(model, torch.full_like(noise, 77), v2.dmd_sigmas(),
        callback=lambda row: callbacks.append(row), disable=True, video_values=2, packed_values=4)
    assert len(model.inputs) == len(callbacks) == 8
    expected = noise.float() * scale
    assert torch.equal(model.inputs[0], expected)
    assert model.inputs[0].dtype == torch.float32
    sv, sa = v2.dmd_sigmas(), v2.dmd_sigmas(3)
    for i in range(8):
        assert torch.allclose(model.inputs[i], expected, rtol=0, atol=5e-6)
        expected = expected.clone()
        expected[..., :2] += (1.1 + i * 0.1) * (sv[i+1] - sv[i])
        expected[..., 2:] += (2.2 + i * 0.2) * (sa[i+1] - sa[i])
    assert torch.allclose(output, expected, rtol=0, atol=5e-6)
    assert torch.allclose(callbacks[-1]["denoised"], output, rtol=0, atol=5e-6)


@pytest.mark.parametrize("mask", [0., .5, 1.])
def test_first_window_fractional_condition_rows_keep_native_init(mask):
    noise = torch.ones(1, 1, 4)
    model = _Injected(noise)
    x = torch.full_like(noise, 5.)
    masks = torch.tensor([1., 1., mask, mask]).view(1, 1, 4)
    v2.sample_v2_euler(model, x, v2.dmd_sigmas(), extra_args={"denoise_mask": masks},
                      disable=True, video_values=2, packed_values=4)
    assert torch.equal(model.inputs[0][..., :2], noise[..., :2])
    assert torch.equal(model.inputs[0][..., 2:], noise[..., 2:] if mask == 1 else x[..., 2:])


def test_partial_restart_is_not_raw_first_window_initialization():
    noise = torch.ones(1, 1, 4)
    model = _Injected(noise)
    sigmas = v2.dmd_sigmas()[4:]
    x = sigmas[0] * noise + (1 - sigmas[0]) * model.latent_image
    v2.sample_v2_euler(model, x, sigmas, disable=True, video_values=2, packed_values=4, stage_start=4)
    audio_sigma = v2.dmd_sigmas(3)[4]
    assert torch.allclose(model.inputs[0][..., 2:],
                          audio_sigma * noise[..., 2:] + (1-audio_sigma) * model.latent_image[..., 2:], atol=1e-6)
    assert len(model.inputs) == 4


@pytest.mark.parametrize("grid", [sampling.native_flow_sigmas(8, 10), torch.ones(9),
                                   v2.dmd_sigmas()[:-1], torch.full((9,), float("nan"))])
def test_wrong_schedule_is_rejected_before_any_network_call(grid):
    model = _Injected(torch.ones(1, 1, 4))
    with pytest.raises(ValueError, match="exact DMD"):
        v2.sample_v2_euler(model, model.noise, grid, video_values=2, packed_values=4)
    assert not model.inputs


def test_missing_injected_noise_never_guesses_ksampler_initial_state():
    with pytest.raises(ValueError, match="Gaussian"):
        v2.sample_v2_euler(lambda *args: None, torch.ones(1, 1, 4), v2.dmd_sigmas(),
                          video_values=2, packed_values=4)


def test_head_projection_keeps_qkv_and_gate_head_order_without_mutating_source():
    value = torch.arange(48.).view(2, 24)
    heads, dim = 4, 2
    def source(x):
        return value
    outputs = [v2._HeadProjection(source, heads, dim, s, s+2)(None) for s in (0, 2)]
    restored = torch.cat([torch.cat([o.chunk(3, -1)[i] for o in outputs], -1) for i in range(3)], -1)
    assert torch.equal(restored, value)
    gate = torch.arange(16.).view(2, 8)
    gates = [v2._HeadProjection(lambda x: gate, heads, dim, s, s+2, gate=True)(None) for s in (0, 2)]
    assert torch.equal(torch.cat(gates, -1), gate)


def test_native_sparse_capability_probe_has_required_semantics():
    sparse = v2._core_sparse()
    assert sparse.VSA_CUBE == (4, 4, 4)


def test_recipe_parameter_mutation_is_detected_not_silently_accepted():
    sparse = v2._core_sparse()
    patch = sparse.SparseAttnPatch(1., .2, True, 1., 0., 0, set(), "exact_kv_and_rows", 0, False)
    runtime = v2._V2Runtime("trained_vsa_exp", sparse, patch)
    options = {"optimized_attention_override": None}
    runtime.validate_options(options)
    patch.topk_ratio = .1
    with pytest.raises(RuntimeError, match="frozen"):
        runtime.validate_options(options)
