import sys
from types import SimpleNamespace

import av
import pytest

from h3_audio_t8_pkg import ffmpeg_utils
from h3_audio_t8_pkg import h3_av_delivery
from test_h3_av_delivery import media


def test_path_encoder_keeps_existing_priority(monkeypatch):
    monkeypatch.delenv("T8_FFMPEG_PATH", raising=False)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _: "/existing/ffmpeg")
    assert ffmpeg_utils.resolve_ffmpeg() == "/existing/ffmpeg"


def test_portable_encoder_is_found_without_path(tmp_path, monkeypatch):
    monkeypatch.delenv("T8_FFMPEG_PATH", raising=False)
    binary = tmp_path / "ffmpeg.exe"
    binary.touch()
    binary.chmod(0o755)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(ffmpeg_utils, "_portable_candidates", lambda: [binary])
    assert ffmpeg_utils.resolve_ffmpeg() == str(binary.resolve())


def test_imageio_fallback_and_absent_encoder_are_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("T8_FFMPEG_PATH", raising=False)
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda _: None)
    monkeypatch.setattr(ffmpeg_utils, "_portable_candidates", lambda: [])
    binary = tmp_path / "imageio-ffmpeg.exe"
    binary.touch()
    binary.chmod(0o755)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: str(binary)))
    assert ffmpeg_utils.resolve_ffmpeg() == str(binary.resolve())
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    with pytest.raises(RuntimeError, match="未找到 FFmpeg"):
        ffmpeg_utils.resolve_ffmpeg()


def test_invalid_explicit_encoder_does_not_silently_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("T8_FFMPEG_PATH", str(tmp_path / "missing.exe"))
    with pytest.raises(RuntimeError, match="T8_FFMPEG_PATH"):
        ffmpeg_utils.resolve_ffmpeg()


def test_real_safe_av_encoding_mux_and_decode_without_path(tmp_path, monkeypatch):
    monkeypatch.delenv("T8_FFMPEG_PATH", raising=False)
    monkeypatch.setenv("PATH", "")
    try:
        binary = ffmpeg_utils.resolve_ffmpeg()
    except RuntimeError:
        pytest.skip("Install portable FFmpeg or imageio-ffmpeg for real PATH-free media regression")
    output = tmp_path / "portable-av.mp4"
    report = h3_av_delivery.save_h3_av_safe(*media(), output)
    assert report["status"] == "pass"
    assert report["frames"] == 24
    with av.open(str(output)) as container:
        assert len(list(container.decode(video=0))) == 24
    with av.open(str(output)) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) >= 32000
    assert list(tmp_path.iterdir()) == [output]
    assert binary
