from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.trt_vae_interface import H3VAEInterface
from test_trt_vae_contract import core_shell, fake_tile, native_rgb
from test_trt_vae_encode import shell, fake_moments


def batch_independent_tile(z):
    # Native attention/norm do not mix independent videos. The shared synthetic
    # fake_tile's global mean would otherwise introduce artificial cross-batch coupling.
    return torch.cat([fake_tile(z[i:i+1]) for i in range(z.shape[0])])


def interface(*, encode_backend="trt", fail=False, budget=8*1024**2):
    events = []
    native = SimpleNamespace(first_stage_model=shell(), encode=lambda x: x + 4,
                             vae_encode_crop_pixels=lambda x: x, marker="native")
    @contextmanager
    def lease(kind, shapes):
        events.append(("open", kind, shapes))
        try:
            if fail:
                raise RuntimeError("simulated backend failure")
            yield batch_independent_tile if kind == "decoder" else fake_moments
        finally:
            events.append(("closed", kind))
    return H3VAEInterface(native, lease, max_output_bytes=budget, encode_backend=encode_backend), events


@pytest.mark.parametrize("tokens", [1, 2, 7, 8, 22])
def test_decode_comfy_layout_matches_actual_core_and_original_unchanged(tokens):
    vae, events = interface()
    z = torch.rand(2, 24, tokens, 2, 3)
    source = z.clone()
    raw = z.half()
    core = core_shell()
    core._decode_pixels = batch_independent_tile
    expected = native_rgb(core, raw)
    actual = vae.decode(z)
    expected = expected.movedim(1, -1).float()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.equal(z, source) and actual.device.type == "cpu" and actual.dtype == torch.float32
    assert events[0][0] == "open" and events[-1] == ("closed", "decoder")
    assert vae.last_report["status"] == "complete" and vae.marker == "native"


@pytest.mark.parametrize("frames", [1, 2, 17, 18, 35])
def test_encode_comfy_pixels_matches_core_and_requires_real_t1(frames):
    vae, events = interface()
    pixels = torch.rand(frames, 32, 48, 3)
    raw = (pixels.movedim(-1, 0).unsqueeze(0) * 2 - 1).half()
    expected = shell().encode(raw)
    actual = vae.encode(pixels)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert events[0][2][0][2] == (1 if frames == 1 else 17)
    assert events[-1] == ("closed", "encoder")


def test_raw_decode_is_preclamp_and_native_encoder_never_opens_trt():
    vae, events = interface(encode_backend="native")
    z = torch.full((1, 24, 7, 2, 3), 20.)
    raw = vae.first_stage_model._adaptive_decode(z)
    assert raw.shape == (1, 3, 28, 32, 48) and raw.min() > 1
    events.clear()
    pixels = torch.zeros(1, 32, 48, 3)
    assert torch.equal(vae.encode(pixels), pixels + 4) and not events
    assert vae.last_report["operation"] == "native_encode"


def test_backend_failure_closes_and_does_not_fallback_or_leave_lock():
    vae, events = interface(fail=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="simulated"):
            vae.decode(torch.zeros(1, 24, 7, 2, 3))
        assert events[-1] == ("closed", "decoder") and vae.last_report["status"] == "failed"


def test_budget_and_bad_pixels_rejected_before_engine():
    vae, events = interface(budget=1)
    with pytest.raises(ValueError, match="budget"):
        vae.decode(torch.zeros(1, 24, 7, 2, 3))
    assert not events
    vae, events = interface()
    with pytest.raises(ValueError, match="finite IMAGE"):
        vae.encode(torch.full((1, 32, 48, 3), 2.))
    assert not events


def test_tiled_aliases_use_native_h3_fixed_tile_semantics():
    vae, _ = interface()
    z = torch.rand(1, 24, 7, 2, 3)
    torch.testing.assert_close(vae.decode_tiled(z, tile_x=16, tile_t=7), vae.decode(z), atol=0, rtol=0)
    pixels = torch.rand(1, 32, 48, 3)
    torch.testing.assert_close(vae.encode_tiled(pixels), vae.encode(pixels), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_decode_preserves_actual_core_pixel_buffer_precision(dtype):
    vae, _ = interface()
    core = core_shell()
    core.pixel_mean = core.pixel_mean.to(dtype)
    core.pixel_std = core.pixel_std.to(dtype)
    vae.native_vae.first_stage_model.pixel_mean = core.pixel_mean
    vae.native_vae.first_stage_model.pixel_std = core.pixel_std
    z = torch.full((1,24,7,2,3),.375)
    expected = native_rgb(core,z.half()).movedim(1,-1)
    actual = vae.decode(z)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-7)


@pytest.mark.parametrize("invalid", [(float('nan'),.4,.5), (.4,.5), (.4,.5,0)])
def test_invalid_pixel_statistics_rejected_before_backend(invalid):
    vae, events = interface()
    vae.native_vae.first_stage_model.pixel_std = torch.tensor(invalid)
    with pytest.raises(ValueError,match="Pixel normalization"):
        vae.decode(torch.zeros(1,24,7,2,3))
    assert not events


def test_raw_outpaint_and_standard_use_same_half_pixel_buffers():
    from h3_audio_t8_pkg.video_outpaint_decode import iter_decode_outpaint_shot
    from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
    from contextlib import contextmanager
    core = core_shell()
    core.pixel_mean = core.pixel_mean.half()
    core.pixel_std = core.pixel_std.half()
    native = SimpleNamespace(first_stage_model=core,device=torch.device('cpu'),throw_exception_if_invalid=lambda:None)
    @contextmanager
    def backend(kind,shapes):
        yield fake_tile
    vae = H3VAEInterface(native,backend,max_output_bytes=16*1024**2)
    plan = build_outpaint_plan(source_sha256='a'*64,width=32,height=64,frame_count=39,
                              aspect='source',generation_megapixels=0,window_frames=39)
    z = torch.full((1,24,12,4,2),.375)
    parts = list(iter_decode_outpaint_shot(vae,lambda a,b:z[:,:,a:b],plan))
    actual = torch.cat([frames for _,frames,_ in parts])
    expected = vae.decode(z)[0]
    assert actual.shape == expected.shape == (39,64,32,3)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_preflight_allocates_nothing_and_reentrant_call_is_rejected():
    vae, events = interface()
    assert vae.prepare_decode((1, 24, 7, 2, 3)) == 1
    assert not events and vae.vae_output_dtype() == torch.float32
    with vae._operation("held"):
        with pytest.raises(RuntimeError, match="already executing"):
            vae.decode(torch.zeros(1, 24, 7, 2, 3))


@pytest.mark.parametrize("frames", [1, 2, 17, 18])
def test_batched_encode_never_crops_time_or_mixes_videos(frames):
    from comfy.sd import VAE
    vae, events = interface()
    crop_owner = SimpleNamespace(crop_input=True, spacial_compression_encode=lambda: 16,
                                 output_channels=3, pad_channel_value=None)
    vae.native_vae.vae_encode_crop_pixels = lambda x: VAE.vae_encode_crop_pixels(crop_owner, x)
    pixels = torch.rand(2, frames, 35, 51, 3)
    actual = vae.encode(pixels)
    expected = torch.cat([vae.encode(video) for video in pixels])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (2, 24, 1 if frames == 1 else ((frames+16)//17)*5-3, 2, 3)


def test_actual_core_vae_decode_returns_unflattened_video_batches(monkeypatch):
    from comfy.sd import VAE
    import comfy.model_management as mm
    from contextlib import nullcontext
    vae, _ = interface()
    core = core_shell()
    core._decode_pixels = batch_independent_tile
    core.latents_mean, core.latents_std = torch.zeros(24), torch.ones(24)
    native = VAE.__new__(VAE)
    native.first_stage_model = core
    native.device = native.output_device = torch.device("cpu")
    native.vae_dtype = torch.float16
    native.latent_dim, native.disable_offload = 3, False
    native.memory_used_decode = lambda *args: 1
    native.patcher = SimpleNamespace(get_free_memory=lambda _: 1)
    native.process_output = ((lambda x: x) if hasattr(core, "_finalize_pixels")
                             else (lambda x: x.add_(1).div_(2).clamp_(0, 1)))
    monkeypatch.setattr(mm, "load_models_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm, "cuda_device_context", lambda _: nullcontext())
    z = torch.rand(2, 24, 7, 2, 3)
    expected = native.decode(z)
    actual = vae.decode(z, vae_options={})
    assert actual.shape == expected.shape == (2, 22, 32, 48, 3)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    with pytest.raises(ValueError, match="extra native"):
        vae.decode(z, vae_options={"unqualified_new_feature": True})
