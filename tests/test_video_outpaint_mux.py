from pathlib import Path
import shutil
import subprocess

import pytest

from h3_audio_t8_pkg.video_outpaint_media import _packet_copy
from h3_audio_t8_pkg.dlss_nr_advanced import _audio_packet_digests, _audio_pcm_digests, _validate_audio_identity
from test_video_outpaint_media import _clip


def test_real_fractional_aac_tail_keeps_all_packet_and_pcm_timestamps(tmp_path):
    # Optional real GPU input that exposed FFmpeg7's171->192 sample tail rounding.
    root = Path(__file__).parents[1] / "artifacts/outpaint-audio-partial80-input-20260906"
    source = root / "source80_cfr.mp4"
    if not source.exists():
        pytest.skip("optional exact fractional-AAC regression input not installed")
    from h3_audio_t8_pkg.video_outpaint_media import _sha256_file
    assert _sha256_file(source).upper() == "ABBE93251436159001D3168721492DB7A8E15C9984CDCD4058EDCC127316F431"
    video = tmp_path / "video.mp4"
    _clip(video, 416, 416, frames=80)
    output = tmp_path / "output.mp4"
    report = _packet_copy(video, source, output)
    assert report["packet_counts"] == [80, 106] and report["pending_packets_bound"] == 2
    checks = _validate_audio_identity(_audio_packet_digests(source), _audio_packet_digests(output),
                                      _audio_pcm_digests(source), _audio_pcm_digests(output))
    assert all(checks.values())


def test_mux_cancel_only_reaps_its_owned_child(tmp_path):
    source = tmp_path / "source.mp4"
    video = tmp_path / "video.mp4"
    _clip(source, 32, 32, frames=80, sound=True)
    _clip(video, 32, 32, frames=80)
    def cancel():
        raise RuntimeError("cancel packet mux")
    with pytest.raises(RuntimeError, match="cancel packet mux"):
        _packet_copy(video, source, tmp_path / "private.mp4", interrupt_check=cancel)
    assert source.exists() and video.exists()


def test_native_child_error_is_contained_in_parent(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"bad media")
    with pytest.raises(RuntimeError, match="isolated outpaint mux failed"):
        _packet_copy(source, source, tmp_path / "private.mp4")


def test_multiple_audio_tracks_keep_distinct_rates_and_timing(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg needed for the real multi-track mux fixture")
    video = tmp_path / "video.mp4"
    _clip(video, 32, 32, frames=80)
    source = tmp_path / "two_tracks.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-n", "-i", str(video),
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000:duration=3.333333333333",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=3.333333333333",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-c:v", "copy", "-c:a", "aac",
        "-threads", "1", str(source)], check=True, capture_output=True, timeout=30)
    output = tmp_path / "output.mp4"
    report = _packet_copy(video, source, output)
    assert report["pending_packets_bound"] == 3 and len(report["packet_counts"]) == 3
    original = _audio_packet_digests(source)
    assert len(original) == 2
    checks = _validate_audio_identity(original, _audio_packet_digests(output),
                                      _audio_pcm_digests(source), _audio_pcm_digests(output))
    assert all(checks.values())
