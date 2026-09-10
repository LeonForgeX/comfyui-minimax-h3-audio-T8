import pytest
import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from h3_audio_t8_pkg.trt_vae_encode import encode_output_shape, encode_rgb, required_encode_tiles


def fake_moments(x):
    # Tile-local terms detect wrong neighbor blending, time padding or batching.
    return x[:, :1, ::4, ::16, ::16].repeat(1, 48, 1, 1, 1) + x.mean(dim=(1, 2, 3, 4), keepdim=True) * .01


def shell():
    core = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(core)
    for name, value in dict(vae_ratio=16, clip_length=17, token_drop=3,
                            tile_size=256, tile_overlap_min=64, tiling=True).items():
        setattr(core, name, value)
    core.pixel_mean = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1, 1)
    core.pixel_std = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1, 1)
    core.latents_mean, core.latents_std = torch.zeros(24), torch.ones(24)
    core._encode_moments = fake_moments
    return core


@pytest.mark.parametrize("frames", [1, 2, 16, 17, 18, 33, 34, 35])
def test_temporal_encoder_matches_actual_core(frames):
    source = torch.rand((2, 3, frames, 32, 48), generator=torch.Generator().manual_seed(frames)) * 2 - 1
    original = source.clone()
    expected = shell().encode(source)
    actual = encode_rgb(source, fake_moments, latents_mean=[0]*24, latents_std=[1]*24, max_output_bytes=1024**2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert tuple(actual.shape) == encode_output_shape(source.shape)
    assert torch.equal(original, source)


@pytest.mark.parametrize("hw", [(256, 256), (272, 304), (512, 1024), (320, 256)])
def test_spatial_encoder_matches_actual_core_including_corners(hw):
    source = torch.rand((1, 3, 1, *hw), generator=torch.Generator().manual_seed(7)) * 2 - 1
    actual = encode_rgb(source, fake_moments, latents_mean=[0]*24, latents_std=[1]*24, max_output_bytes=1024**2)
    torch.testing.assert_close(actual, shell().encode(source), atol=1e-6, rtol=1e-6)


def test_single_frame_static_mode_is_explicit_and_first_not_last():
    source = torch.zeros(1, 3, 1, 256, 256)
    calls = []
    def backend(value):
        calls.append(value.shape)
        assert not bool(value[:, :, 1:].any())
        return torch.arange(5).float().view(1, 1, 5, 1, 1).expand(1, 48, 5, 16, 16)
    result = encode_rgb(source, backend, latents_mean=[0]*24, latents_std=[1]*24,
                        max_output_bytes=1024**2, single_frame_mode="causal_zero17_first")
    assert not bool(result.any()) and calls == [(1, 3, 17, 256, 256)]
    assert required_encode_tiles(source.shape) == ((1, 3, 1, 256, 256),)
    assert required_encode_tiles((1, 3, 17, 128, 256)) == ((1, 3, 17, 128, 256),)


@pytest.mark.parametrize("bad", ["shape", "budget", "nan", "range", "std", "backend", "cancel"])
def test_bad_input_or_backend_never_returns_partial_latent(bad):
    source = torch.zeros(1, 3, 1, 32, 32)
    options = dict(latents_mean=[0]*24, latents_std=[1]*24, max_output_bytes=1024**2)
    backend = fake_moments
    if bad == "shape":
        source = source[..., :31]
    if bad == "budget":
        options["max_output_bytes"] = 1
    if bad == "nan":
        source.fill_(float("nan"))
    if bad == "range":
        source.fill_(2)
    if bad == "std":
        options["latents_std"] = [0]*24
    if bad == "backend":
        def backend(_):
            return torch.zeros(1)
    if bad == "cancel":
        def cancelled():
            raise InterruptedError("cancel")
        options["check"] = cancelled
    with pytest.raises((ValueError, InterruptedError)):
        encode_rgb(source, backend, **options)
