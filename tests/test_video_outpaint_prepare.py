from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_prepare import (
    iter_encode_outpaint_source, source_to_sampling_canvas, sampling_to_output_canvas,
    SequentialOutpaintFrameReader,
)


class NativeTemporalOracle:
    """Real native temporal orchestration; cheap deterministic fake spatial encoder."""
    clip_length, token_drop, tokens_chunk_size, vae_ratio = 17, 3, 5, 16
    encode_temporal = MiniMaxH3VideoVAE.encode_temporal
    encode = MiniMaxH3VideoVAE.encode
    latents_mean = torch.linspace(-0.5, 0.5, 24)
    latents_std = torch.linspace(0.5, 1.5, 24)

    def _normalize_pixels(self, frames):
        return frames

    def _adaptive_encode(self, frames):
        sampled = frames[:, :1, [0, 4, 8, 12, 16], ::16, ::16]
        return sampled.expand(-1, 48, -1, -1, -1).clone()


class PublicVAE:
    def __init__(self):
        self.first_stage_model = NativeTemporalOracle()
        self.calls = []

    def encode(self, frames):
        self.calls.append(tuple(frames.shape))
        return self.first_stage_model.encode(frames.permute(3, 0, 1, 2)[None] * 2 - 1)


def _plan(count, cuts=(), **kwargs):
    return build_outpaint_plan(source_sha256="a" * 64, width=32, height=32, frame_count=count,
                              aspect="custom", left=32, right=32, cut_frames=cuts, **kwargs)


@pytest.mark.parametrize("count", [1, 5, 16, 17, 18, 22, 34, 39, 73, 90, 768])
def test_bounded_encode_matches_native_full_temporal_oracle(count):
    plan = _plan(count)
    pixels = torch.linspace(0, 1, count)[:, None, None, None].expand(count, 32, 32, 3).clone()
    reads = []

    def read(start, stop):
        reads.append((start, stop))
        return pixels[start:stop]

    vae = PublicVAE()
    chunks = list(iter_encode_outpaint_source(vae, read, plan))
    actual = torch.cat([chunk for _, chunk, _ in chunks], dim=2)
    aligned = plan["shots"][0]["aligned_frames"]
    held = torch.cat((pixels, pixels[-1:].expand(aligned-count, -1, -1, -1)))
    expected = NativeTemporalOracle().encode(held.permute(3, 0, 1, 2)[None] * 2 - 1)
    assert torch.equal(actual[..., 2:4], expected)
    assert torch.count_nonzero(actual[..., :2]) == 0
    assert all(stop-start <= 17 for start, stop in reads)
    assert all(shape == (22, 32, 32, 3) for shape in vae.calls)
    assert [start for start, _, _ in chunks] == list(range(0, expected.shape[2], 5))


def test_cuts_never_read_previous_scene():
    plan = _plan(40, cuts=(18,))
    pixels = torch.zeros((40, 32, 32, 3))
    pixels[18:] = 1
    reads = []

    def read(start, stop):
        reads.append((start, stop))
        return pixels[start:stop]

    list(iter_encode_outpaint_source(PublicVAE(), read, plan, shot_index=1))
    assert all(start >= 18 and stop <= 40 for start, stop in reads)


def test_identity_coordinate_transform_and_native_pixel_placement():
    plan = _plan(5)
    source = torch.rand(2, 32, 32, 3)
    canvas = source_to_sampling_canvas(source, plan)
    assert torch.allclose(canvas[:, :, 32:64], source, atol=2e-6)
    assert torch.allclose(canvas[:, :, :32], source[:, :, :1].expand(-1, -1, 32, -1), atol=2e-6)
    output = sampling_to_output_canvas(canvas, plan)
    assert torch.allclose(canvas, output, atol=3e-6)


def test_budgeted_canvas_is_explicit_and_bounded():
    plan = _plan(5, generation_megapixels=0.002)
    pixels = torch.full((5, 32, 32, 3), 0.4)
    vae = PublicVAE()
    chunks = list(iter_encode_outpaint_source(vae, lambda a, b: pixels[a:b], plan))
    assert chunks[0][2]["geometry_mode"] == "isotropic_edge_canvas"
    assert vae.calls[0][1] * vae.calls[0][2] <= 2000
    expanded = sampling_to_output_canvas(torch.full((1, plan["sampling"]["height"], plan["sampling"]["width"], 3), 0.4), plan)
    assert expanded.shape == (1, 32, 96, 3)
    assert torch.allclose(expanded, torch.full_like(expanded, 0.4))


def test_odd_delivery_maps_source_to_whole_latent_rows_without_stretch_band():
    plan = build_outpaint_plan(
        source_sha256="a" * 64,
        width=736,
        height=416,
        frame_count=56,
        aspect="custom",
        top=95,
        bottom=96,
        window_frames=56,
        generation_megapixels=0.5,
    )
    source = torch.linspace(0, 1, 416)[:, None, None].expand(416, 736, 3)[None].clone()
    canvas = source_to_sampling_canvas(source, plan)
    assert canvas.shape == (1, 608, 736, 3)
    assert plan["sampling"]["source_rect"] == [0.0, 96.0, 736.0, 512.0]
    assert plan["sampling"]["source_lock_latent_box"] == [0, 6, 46, 32]
    output = sampling_to_output_canvas(canvas, plan)
    assert output.shape == (1, 607, 736, 3)
    assert torch.allclose(output[:, 95:511], source, atol=2e-6)


def test_incompatible_vae_and_cancel_are_explicit():
    with pytest.raises(ValueError, match="native H3"):
        list(iter_encode_outpaint_source(SimpleNamespace(first_stage_model=None), None, _plan(5)))

    def cancel():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        list(iter_encode_outpaint_source(PublicVAE(), None, _plan(5), interrupt_check=cancel))


def test_resume_does_not_reencode_committed_source_chunks():
    plan = _plan(90)
    pixels = torch.rand((90, 32, 32, 3))
    reads = []

    def read(a, b):
        reads.append((a, b))
        return pixels[a:b]

    chunks = list(iter_encode_outpaint_source(PublicVAE(), read, plan, start_token=10))
    assert chunks[0][0] == 10 and reads[0][0] == 34
    with pytest.raises(ValueError, match="chunk boundary"):
        list(iter_encode_outpaint_source(PublicVAE(), read, plan, start_token=7))


def test_real_file_reader_is_bounded_supports_resume_and_held_last_frame(tmp_path):
    from test_video_outpaint_media import _clip
    import av

    path = tmp_path / "source.mp4"
    _clip(path, 32, 32, frames=39)
    plan = _plan(39)
    with av.open(str(path)) as reference:
        expected = torch.stack([torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in reference.decode(video=0)])
    with SequentialOutpaintFrameReader(path, plan) as reader:
        assert torch.equal(reader(17, 34), expected[17:34])
        assert torch.equal(reader(34, 39), expected[34:39])
        assert torch.equal(reader(38, 39), expected[38:39])
        with pytest.raises(ValueError, match="backwards"):
            reader(0, 1)
    assert reader.container is None and reader.last is None
    with pytest.raises(RuntimeError, match="not open"):
        reader(0, 1)
