"""Real Core orchestration + synthetic tile kernel: CPU contracts, not TRT quality."""

from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

from h3_audio_t8_pkg.trt_vae_contract import (
    output_bytes, output_shape, required_tile_shapes, temporal_plan, tile_axis,
    validate_declared_input,
)
from h3_audio_t8_pkg.trt_vae_decode import decode_raw_latent


def fake_tile(z):
    # Tile-local statistics intentionally expose ordering / blending differences.
    raw = z[:, :3].repeat_interleave(4, 2).repeat_interleave(16, 3).repeat_interleave(16, 4)
    return raw + z.mean() * 0.05


def core_shell():
    # Execute actual installed Core methods without constructing/loading weights.
    core = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(core)
    for name, value in dict(vae_ratio=16, vae_ratio_t=4, clip_length=17,
                            token_drop=3, frame_pre_padding=3, tokens_chunk_size=5,
                            token_overlap=2, frame_overlap=5, tile_size=256,
                            tile_overlap_min=64, tiling=True).items():
        setattr(core, name, value)
    core.decoder = SimpleNamespace(out_channels=3)
    core.pixel_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    core.pixel_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    core.latents_mean, core.latents_std = torch.zeros(24), torch.ones(24)
    core._decode_pixels = fake_tile
    return core


def native_rgb(core, z):
    # Old H3 returns [-1,1] to VAE.process_output; newer chunked I/O finalizes
    # directly to [0,1]. Compare the same public RGB domain, not raw vs final.
    value = core.decode(z)
    return value if hasattr(core, "_finalize_pixels") else value.add(1).mul(.5).clamp(0, 1)


def native_shape(core, shape):
    if hasattr(core, "decode_output_shape"):
        return core.decode_output_shape(shape)
    tokens = shape[2]
    if tokens == 1:
        frames = 1
    else:
        padding = (-(tokens + core.token_drop)) % core.tokens_chunk_size
        windows = (tokens + padding + core.token_drop) // core.tokens_chunk_size - 1
        if windows < 1:
            padding += core.tokens_chunk_size
            windows += 1
        frames = core._decode_temporal_frame_plan(tokens + padding, windows, padding)
    return (shape[0], 3, frames, shape[3] * core.vae_ratio, shape[4] * core.vae_ratio)


def test_frame_oracle_matches_actual_core_for_512_lengths():
    core = core_shell()
    for tokens in range(1, 513):
        assert output_shape((2, 24, tokens, 38, 46)) == native_shape(core, (2, 24, tokens, 38, 46))


def test_tile_lattice_matches_actual_core():
    core = core_shell()
    for pixels in range(16, 4097, 16):
        axis = tile_axis(pixels)
        expected = core.split_tiles(pixels)
        assert (axis.starts, axis.lengths, axis.overlaps) == tuple(tuple(x) for x in expected)
        assert axis.starts[-1] + axis.lengths[-1] == pixels


@pytest.mark.parametrize("tokens,frames", [(1, 1), (2, 5), (5, 17), (6, 18), (7, 22), (22, 73), (227, 770)])
def test_known_frame_counts(tokens, frames):
    assert temporal_plan(tokens).frames == frames


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 6, 7, 8, 12, 22])
def test_tensor_time_assembly_matches_actual_core(tokens):
    z = torch.rand((1, 24, tokens, 2, 3), generator=torch.Generator().manual_seed(tokens)) * 2 - 1
    before = z.clone()
    core = core_shell()
    expected = native_rgb(core, z)
    actual = decode_raw_latent(z, fake_tile, max_output_bytes=output_bytes(z.shape))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.equal(z, before)
    assert actual.device.type == "cpu" and actual.dtype == torch.float32


@pytest.mark.parametrize("height,width", [(16, 17), (17, 19), (38, 46), (46, 38)])
def test_spatial_seams_match_native_tensor_assembly(height, width):
    z = torch.rand((1, 24, 2, height, width), generator=torch.Generator().manual_seed(40)) * 2 - 1
    expected = native_rgb(core_shell(), z)
    actual = decode_raw_latent(z, fake_tile, max_output_bytes=output_bytes(z.shape))
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_batch_is_serial_and_preserves_order():
    z = torch.stack((torch.zeros(24, 2, 2, 2), torch.ones(24, 2, 2, 2)))
    calls = []

    def tile(value):
        calls.append((tuple(value.shape), value.mean().item()))
        return fake_tile(value)

    actual = decode_raw_latent(z, tile, max_output_bytes=output_bytes(z.shape))
    expected = torch.cat([native_rgb(core_shell(), z[i:i + 1]) for i in range(2)])
    torch.testing.assert_close(actual, expected)
    assert [mean for _, mean in calls] == [0, 1]
    assert all(shape[0] == 1 for shape, _ in calls)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_tokens_rejected(bad):
    with pytest.raises(ValueError):
        temporal_plan(bad)


@pytest.mark.parametrize("shape", [(1, 24, 2, 3), (1, 32, 2, 3, 3), (1, 24, 0, 3, 3)])
def test_invalid_latent_shapes(shape):
    with pytest.raises(ValueError):
        output_shape(shape)


@pytest.mark.parametrize("pixels", [0, 17, 255, True])
def test_invalid_tile_dimensions(pixels):
    with pytest.raises(ValueError):
        tile_axis(pixels)


def test_budget_rejects_before_backend_called():
    z = torch.zeros(1, 24, 2, 2, 2)
    with pytest.raises(ValueError, match="budget"):
        decode_raw_latent(z, lambda _: pytest.fail("must not execute"), max_output_bytes=1)


@pytest.mark.parametrize("failure", ["shape", "nan", "exception", "integer"])
def test_bad_backend_never_returns_partial_video(failure):
    z = torch.zeros(1, 24, 2, 2, 2)

    def tile(value):
        if failure == "exception":
            raise RuntimeError("backend failed")
        result = fake_tile(value)
        if failure == "shape":
            return result[:, :, :-1]
        if failure == "integer":
            return result.to(torch.int32)
        result[0, 0, 0, 0, 0] = float("nan")
        return result

    with pytest.raises((ValueError, RuntimeError)):
        decode_raw_latent(z, tile, max_output_bytes=output_bytes(z.shape))


def test_cancellation_propagates_between_tiles():
    z = torch.zeros(1, 24, 2, 17, 19)
    calls = []

    def tile(value):
        calls.append(True)
        return fake_tile(value)

    def check():
        if calls:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        decode_raw_latent(z, tile, max_output_bytes=output_bytes(z.shape), check=check)
    assert len(calls) == 1


def test_nonfinite_input_rejected_without_backend():
    z = torch.full((1, 24, 2, 2, 2), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        decode_raw_latent(z, lambda _: pytest.fail("must not execute"), max_output_bytes=output_bytes(z.shape))


def test_standard_video_fits_upstream_static_graph():
    shapes = required_tile_shapes((1, 24, 22, 32, 64))
    assert shapes == ((1, 24, 7, 16, 16),)
    validate_declared_input(("batch_size", 24, 7, 16, 16), shapes)


@pytest.mark.parametrize("shape", [(1, 24, 1, 32, 64), (1, 24, 22, 8, 64), (1, 24, 22, 32, 8)])
def test_small_shapes_cannot_be_silently_run_on_upstream_static_graph(shape):
    with pytest.raises(ValueError, match="matching export"):
        validate_declared_input(("batch_size", 24, 7, 16, 16), required_tile_shapes(shape))


def test_symbolic_declaration_is_only_static_check_not_engine_qualification():
    validate_declared_input(("B", 24, "T", "H", "W"), required_tile_shapes((2, 24, 1, 8, 12)))


@pytest.mark.parametrize("bad", [(1, 24, 7, 0, 16), (True, 24, 7, 16, 16), ("", 24, 7, 16, 16)])
def test_invalid_declared_shape_rejected(bad):
    with pytest.raises(ValueError):
        validate_declared_input(bad, [(1, 24, 7, 16, 16)])
