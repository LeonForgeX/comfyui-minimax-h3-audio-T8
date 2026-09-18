from fractions import Fraction
import json

import pytest
import torch

from h3_audio_t8_pkg.audio_opening_fade import mute_fade_audio, mute_fade_video_components


def audio(samples=5000, channels=2, batch=2, sample_rate=32000):
    return {"waveform": torch.ones(batch, channels, samples), "sample_rate": sample_rate,
            "custom_metadata": {"untouched": True}}


@pytest.mark.parametrize("enabled,n,fade", [(False, 999, 500), (True, 0, 0)])
def test_strict_bypass_does_not_clone_or_validate_fps(enabled, n, fade):
    source = audio()
    result, report = mute_fade_audio(source, enabled, n, fade, "not-a-rate")
    assert result is source
    assert json.loads(report)["status"] == "bypass"


@pytest.mark.parametrize("rate", [8000, 32000, 44100, 48000])
@pytest.mark.parametrize("fps", ["24", "25", "30000/1001"])
def test_exact_rational_boundary_all_batches_and_channels(rate, fps):
    source = audio(10000, sample_rate=rate)
    before = source["waveform"].clone()
    result, text = mute_fade_audio(source, True, 1, 10, fps)
    report = json.loads(text)
    exact = Fraction(rate) / Fraction(fps)
    mute = -(-exact.numerator // exact.denominator)
    fade = rate // 100
    assert report["mute_samples"] == mute
    assert report["fade_samples"] == fade
    assert torch.count_nonzero(result["waveform"][..., :mute]) == 0
    gain = result["waveform"][0, 0, mute:mute + fade]
    assert gain[0] == 0 and gain[-1] == 1
    assert torch.all(gain[1:] >= gain[:-1])
    assert torch.equal(result["waveform"][..., mute + fade:], before[..., mute + fade:])
    assert torch.equal(source["waveform"], before)
    assert result["waveform"].shape == before.shape
    assert result["custom_metadata"] is source["custom_metadata"]
    assert result["sample_rate"] == rate


@pytest.mark.parametrize("samples,n,fade", [(0, 1, 10), (1, 0, 10), (50, 999, 10), (50, 0, 9999)])
def test_empty_short_audio_and_clipped_envelope(samples, n, fade):
    source = audio(samples)
    result, text = mute_fade_audio(source, True, n, fade)
    report = json.loads(text)
    assert result["waveform"].shape[-1] == samples
    assert report["affected_samples"] <= samples
    assert torch.isfinite(result["waveform"]).all()
    if samples == 1:
        assert result["waveform"].item() == 0 if result["waveform"].numel() == 1 else torch.count_nonzero(result["waveform"]) == 0
    if n:
        assert torch.count_nonzero(result["waveform"]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_and_input_views_preserved(dtype):
    source = audio()
    source["waveform"] = source["waveform"].to(dtype)[..., ::2]
    result, _ = mute_fade_audio(source, True, 1, 10)
    assert result["waveform"].dtype == dtype
    assert source["waveform"].eq(1).all()


def test_mute_only_and_fade_only():
    source = audio()
    mute, report = mute_fade_audio(source, True, 2, 0)
    end = json.loads(report)["mute_samples"]
    assert end == 2667
    assert torch.count_nonzero(mute["waveform"][..., :end]) == 0
    assert mute["waveform"][..., end:].eq(1).all()
    fade, report = mute_fade_audio(source, True, 0, 10)
    assert json.loads(report)["mute_samples"] == 0
    assert fade["waveform"][..., 0].eq(0).all()


@pytest.mark.parametrize("kwargs", [{"fps": "0"}, {"fps": "nan"}, {"fps": "-2"},
                                   {"fade_in_ms": float("inf")}, {"mute_first_frames": -1},
                                   {"mute_first_frames": 1.2}, {"mute_first_frames": True}])
def test_bad_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        mute_fade_audio(audio(), **kwargs)


def test_bad_pcm_and_rate_rejected():
    source = audio()
    source["waveform"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        mute_fade_audio(source)
    with pytest.raises(ValueError):
        mute_fade_audio(audio(sample_rate=0))


def test_native_video_components_frames_metadata_alpha_and_no_audio():
    from comfy_api.latest import InputImpl, Types
    frames = torch.rand(24, 32, 32, 3)
    alpha = torch.rand(24, 32, 32)
    source = InputImpl.VideoFromComponents(Types.VideoComponents(frames, Fraction(24), audio(32000),
                                                                {"identity": "same"}, alpha), bit_depth=10)
    result, _ = mute_fade_video_components(source)
    a, b = source.get_components(), result.get_components()
    assert b.images is a.images and b.alpha is a.alpha and b.metadata is a.metadata
    assert b.frame_rate == a.frame_rate and result.get_bit_depth() == 10
    assert result.get_color_space() == source.get_color_space()
    assert b.audio["waveform"].shape == a.audio["waveform"].shape
    no_audio = InputImpl.VideoFromComponents(Types.VideoComponents(frames, Fraction(24)))
    assert mute_fade_video_components(no_audio)[0] is no_audio


def test_file_video_never_silently_loses_vfr_or_pts():
    from comfy_api.latest import InputImpl
    source = InputImpl.VideoFromFile("not-opened.mp4")
    assert mute_fade_video_components(source, False)[0] is source
    with pytest.raises(FileNotFoundError):
        mute_fade_video_components(source)
