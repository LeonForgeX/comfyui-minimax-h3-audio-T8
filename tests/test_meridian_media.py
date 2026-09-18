from fractions import Fraction

import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.meridian_media import (
    image_u8,
    video_window,
    original_audio_window,
    selected_frames,
    display_pixels,
)


@pytest.mark.parametrize("sar", [None, Fraction(0), Fraction(1)])
def test_unspecified_sar_does_not_collapse_image(sar):
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    assert np.array_equal(display_pixels(pixels, sar, 0), pixels)


def test_positive_sar_then_rotation_preserves_display_aspect():
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    assert display_pixels(pixels, Fraction(2), 90).shape == (12, 4, 3)
    with pytest.raises(ValueError, match="Negative"):
        display_pixels(pixels, Fraction(-1), 0)
    with pytest.raises(ValueError, match="rotation"):
        display_pixels(pixels, 1, 45)


def fixture_video(path):
    from comfy_api.latest import InputImpl, Types

    n = 24
    images = torch.zeros(n, 64, 96, 3)
    for i in range(n):
        images[i] = i / 32
    t = torch.arange(16000, dtype=torch.float32) / 16000
    audio = dict(
        waveform=(0.1 * torch.sin(2 * torch.pi * 440 * t))[None, None],
        sample_rate=16000,
    )
    InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, frame_rate=Fraction(24), audio=audio)
    ).save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
    return InputImpl.VideoFromFile(str(path))


def test_image_exact_single_preserves_portrait():
    assert image_u8(torch.ones(1, 96, 64, 3)).shape == (1, 96, 64, 3)
    with pytest.raises(ValueError):
        image_u8(torch.ones(2, 96, 64, 3))
    with pytest.raises(ValueError):
        image_u8(torch.full((1, 8, 8, 3), float("nan")))


def test_bounded_video_view_trim_crop_and_real_pts(tmp_path):
    from comfy_api.latest import InputImpl

    path = tmp_path / "source.mp4"
    fixture_video(path)
    # Actual Core view, not a fake get_components object.
    view = InputImpl.VideoFromFile(
        str(path), start_time=0.25, duration=0.5, crop=(0, 0, 64, 64)
    )
    images, clock = video_window(view, 0, 8)
    assert images.shape == (8, 64, 64, 3)
    assert images[0].float().mean() == pytest.approx(6 / 32 * 255, abs=3)
    assert clock["normalized_fps"] == "24"


def test_window_longer_than_clip_not_silently_padded(tmp_path):
    path = tmp_path / "source.mp4"
    fixture_video(path)
    with pytest.raises(ValueError, match="longer"):
        selected_frames(path, 0, 30)


def test_source_identity_repeat_and_local_decoder_thread(tmp_path, monkeypatch):
    import av

    path = tmp_path / "repeat.mp4"
    fixture_video(path)
    real_open = av.open
    observed = []

    class ObservedContainer:
        def __init__(self, *args, **kwargs):
            self.container = real_open(*args, **kwargs)

        def __enter__(self):
            return self.container.__enter__()

        def __exit__(self, *args):
            observed.append(self.container.streams.video[0].codec_context.thread_count)
            return self.container.__exit__(*args)

    monkeypatch.setattr(av, "open", ObservedContainer)
    baseline, clock = selected_frames(path, 0, 24)
    for _ in range(5):
        again, again_clock = selected_frames(path, 0, 24)
        assert torch.equal(again, baseline) and again_clock == clock
    assert observed == [1] * 6


def test_cancel_selected_decode_leaves_source(tmp_path):
    path = tmp_path / "source.mp4"
    fixture_video(path)

    def check():
        raise InterruptedError("own cancellation")

    with pytest.raises(InterruptedError):
        selected_frames(path, 0, 3, check)
    assert path.is_file()


def test_original_audio_sample_length_rate_channels(tmp_path):
    path = tmp_path / "source.mp4"
    video = fixture_video(path)
    audio = original_audio_window(video, 6, 12)
    assert audio["sample_rate"] == 16000 and audio["waveform"].shape == (1, 1, 8000)
    assert audio["waveform"].square().mean().sqrt() > 0.04


def test_silent_source_original_audio_explicit_error(tmp_path):
    from comfy_api.latest import InputImpl, Types

    path = tmp_path / "silent.mp4"
    InputImpl.VideoFromComponents(
        Types.VideoComponents(images=torch.ones(8, 64, 96, 3), frame_rate=Fraction(24))
    ).save_to(str(path))
    with pytest.raises(ValueError, match="no audio"):
        original_audio_window(InputImpl.VideoFromFile(str(path)), 0, 4)
