"""CPU full-reference construction and real engine profile admission checks."""

import pytest
import torch

from tools.trt_vae_video_probe_worker import check_geometry, native_video_shell, warmup_count
from tools.audit_trt_vae_video_probe import frame_metrics


def test_actual_reference_keeps_normalization_float32_and_rope_materialized():
    from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD, MiniMaxH3VideoVAE, RotaryEmbeddingND

    core = native_video_shell()
    assert type(core) is MiniMaxH3VideoVAE
    assert core.decode.__func__ is MiniMaxH3VideoVAE.decode
    assert not hasattr(core, "encoder") and not hasattr(core, "quant_conv")
    for name, values in (("latents_mean", LATENTS_MEAN), ("latents_std", LATENTS_STD)):
        assert torch.equal(getattr(core, name), torch.tensor(values))
    for name in ("latents_mean", "latents_std", "pixel_mean", "pixel_std"):
        assert getattr(core, name).dtype == torch.float32 and not getattr(core, name).is_meta
    torch.testing.assert_close(core.decoder.pos_embed.inv_freq, RotaryEmbeddingND(48, 100.0, 3).inv_freq)
    assert all(p.is_meta for p in core.parameters())
    assert core.decode_output_shape((1, 24, 37, 20, 36)) == (1, 3, 124, 320, 576)


def test_real_saved_video_geometry_and_all_tiles_counted():
    shape, calls = check_geometry((1, 24, 37, 20, 36), 1024**3)
    assert shape == (1, 3, 124, 320, 576)
    assert calls == 42  # Seven time windows, 2x3 spatial tiles each.


@pytest.mark.parametrize("shape", [(1, 24, 1, 32, 64), (1, 24, 7, 15, 32), (2, 24, 7, 16, 16)])
def test_unsupported_shapes_are_not_padded_or_silently_sliced(shape):
    with pytest.raises(ValueError):
        check_geometry(shape, 1024**3)


def test_output_budget_is_checked_before_model_or_cuda():
    with pytest.raises(ValueError, match="budget"):
        check_geometry((1, 24, 37, 20, 36), 1024)


@pytest.mark.parametrize("bad", [True, False, -1, 2, 1.0, "1", None])
def test_warmup_is_explicit_and_bounded(bad):
    with pytest.raises(ValueError, match="Warmup"):
        warmup_count({"warmup_complete_decodes": bad})


def test_default_probe_never_silently_adds_warmup_work():
    assert warmup_count({}) == 0
    assert warmup_count({"warmup_complete_decodes": 1}) == 1


def test_independent_frame_metrics_are_over_every_channel():
    reference = torch.zeros((1, 3, 2, 2))
    candidate = reference.clone()
    candidate[0, 2, 1, 1] = .5
    assert frame_metrics(reference, candidate) == {
        "count": 12, "sum_squared_error": .25, "max_error": .5, "over_2_codes": 1,
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.01, 1.01])
def test_independent_metrics_reject_invalid_rgb(bad):
    with pytest.raises(ValueError, match="RGB"):
        frame_metrics(torch.zeros(1), torch.tensor([bad]))


def test_cpu_media_encodes_all_frames_and_exact_clock(tmp_path):
    from safetensors.torch import save_file
    from tools.prepare_trt_vae_video_media import encode_rgb, inspect_video

    rgb = torch.rand((1, 3, 3, 64, 64), generator=torch.Generator().manual_seed(7))
    tensors = tmp_path / "rgb.safetensors"
    save_file({"rgb": rgb}, str(tensors))
    source = {"fps": "24", "origin": "0", "colors": {"color_primaries": 2, "color_trc": 2, "colorspace": 2, "color_range": 0}}
    target = tmp_path / "review.mp4"
    encode_rgb(target, tensors, rgb.shape, source)
    actual = inspect_video(target)
    assert (actual["frames"], actual["duration"], actual["pts"]) == (3, "1/8", ["0", "1/24", "1/12"])
    with pytest.raises(FileExistsError):
        encode_rgb(target, tensors, rgb.shape, source)
    with pytest.raises(ValueError, match="geometry"):
        encode_rgb(tmp_path / "wrong.mp4", tensors, (1, 3, 4, 64, 64), source)


def test_bad_media_is_not_accepted(tmp_path):
    import av
    from tools.prepare_trt_vae_video_media import inspect_video

    source = tmp_path / "bad.mp4"
    source.write_bytes(b"not-a-video")
    with pytest.raises(av.error.InvalidDataError):
        inspect_video(source)


def test_media_provenance_accepts_only_known_completed_formats():
    from tools.prepare_trt_vae_video_media import source_media_identity

    report = {"status": "mechanical_media_pass_human_quality_unverified", "strict_av_decode": True,
              "file": {"path": "original.mp4", "sha256": "a" * 64}}
    assert source_media_identity(report, "draft") == {"path": "original.mp4", "file_sha256": "a" * 64}
    report["strict_av_decode"] = False
    with pytest.raises(ValueError, match="strict"):
        source_media_identity(report, "draft")
    with pytest.raises(ValueError, match="Unknown"):
        source_media_identity({"media": {}}, "draft")
