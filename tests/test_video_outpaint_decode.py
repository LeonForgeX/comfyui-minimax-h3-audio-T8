from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from h3_audio_t8_pkg.video_outpaint_decode import iter_decode_outpaint_shot
from test_video_outpaint_prepare import _plan


class DecoderOracle:
    clip_length, tokens_chunk_size, token_drop, token_overlap = 17, 5, 3, 2
    frame_pre_padding, frame_overlap, vae_ratio, vae_ratio_t = 3, 5, 16, 4
    decoder = SimpleNamespace(out_channels=3)
    latents_mean = torch.zeros(24)
    latents_std = torch.ones(24)
    pixel_mean = torch.full((1, 3, 1, 1, 1), 0.5)
    pixel_std = torch.full((1, 3, 1, 1, 1), 0.25)
    blend = MiniMaxH3VideoVAE.blend
    decode = MiniMaxH3VideoVAE.decode
    decode_temporal = MiniMaxH3VideoVAE.decode_temporal
    decode_output_shape = MiniMaxH3VideoVAE.decode_output_shape
    _decode_temporal_pad_frames = MiniMaxH3VideoVAE._decode_temporal_pad_frames
    _decode_temporal_frame_plan = MiniMaxH3VideoVAE._decode_temporal_frame_plan
    _decode_temporal_chunks = MiniMaxH3VideoVAE._decode_temporal_chunks
    _finalize_pixels = MiniMaxH3VideoVAE._finalize_pixels

    def _adaptive_decode(self, z):
        # Deliberately exceed RGB gamut so post-clamp blending would fail parity.
        # Keep the fake reduction layout-independent: global slices are strided,
        # while bounded reads are compact (otherwise mean() differs by one FP32 ULP).
        z = z.contiguous()
        pixels = z[:, :3].repeat_interleave(4, dim=2).repeat_interleave(16, dim=3).repeat_interleave(16, dim=4)
        return pixels + z.mean()*3


class VAE:
    first_stage_model = DecoderOracle()
    device = torch.device("cpu")
    vae_dtype = torch.float32

    def __init__(self):
        self.shapes = []

    def throw_exception_if_invalid(self):
        pass

    def prepare_decode(self, shape):
        self.shapes.append(shape)


@pytest.mark.parametrize("count", [1, 5, 16, 17, 18, 22, 34, 39, 73, 90, 768])
def test_bounded_decode_matches_native_global_decode_before_clamp(count):
    plan = _plan(count)
    total = (plan["shots"][0]["aligned_frames"]-5)//17*5+2
    generator = torch.Generator().manual_seed(3)
    z = torch.randn((1, 24, total, 2, 6), generator=generator)*4
    vae = VAE()
    reads = []

    def read(a, b):
        reads.append((a, b))
        return z[:, :, a:b]

    parts = list(iter_decode_outpaint_shot(vae, read, plan))
    actual = torch.cat([frames for _, frames, _ in parts])
    expected = vae.first_stage_model.decode(z)[0].movedim(0, -1)[:count]
    assert torch.equal(actual, expected)
    assert actual.shape == (count, 32, 96, 3)
    assert all(b-a <= 7 for a, b in reads)
    assert all(shape[2] == 7 for shape in vae.shapes)
    assert all(frames.shape[0] <= 17 for _, frames, _ in parts)
    assert sum(frames.shape[0] for _, frames, _ in parts) == count


def test_wrong_decoder_contract_and_trajectory_are_rejected():
    vae = VAE()
    vae.first_stage_model = SimpleNamespace()
    with pytest.raises(ValueError, match="native H3"):
        list(iter_decode_outpaint_shot(vae, None, _plan(5)))
    with pytest.raises(ValueError, match="trajectory"):
        list(iter_decode_outpaint_shot(VAE(), lambda a, b: torch.zeros(1), _plan(5)))
