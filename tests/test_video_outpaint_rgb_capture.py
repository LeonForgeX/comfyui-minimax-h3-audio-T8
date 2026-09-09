import hashlib
import json

import pytest

from h3_audio_t8_pkg.video_outpaint_rgb_capture import EncoderRGBCapture


def capture(path, frames=2):
    return EncoderRGBCapture(path, width=2, height=1, frames=frames, provenance={"test": True})


def test_exact_stream_survives_downstream_validation_failure(tmp_path):
    path = tmp_path / "frames.rgb"
    with pytest.raises(RuntimeError, match="bad encoded video"):
        with capture(path) as recorder:
            recorder.write_frame(b"123456")
            recorder.write_frame(memoryview(b"abcdef"))
            raise RuntimeError("bad encoded video")
    value = json.loads(path.with_suffix(".rgb.json").read_text())
    assert path.read_bytes() == b"123456abcdef"
    assert value["rgb_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert value["complete_rgb_stream"] and value["status"] == "captured"
    assert not value["video_acceptance_claimed"]
    assert "bad encoded video" in value["error"]


def test_partial_capture_is_retained_but_never_claimed_complete(tmp_path):
    path = tmp_path / "partial.rgb"
    with pytest.raises(ValueError, match="before all frames"):
        with capture(path) as recorder:
            recorder.write_frame(b"123456")
    value = json.loads(path.with_suffix(".rgb.json").read_text())
    assert value["status"] == "partial" and not value["complete_rgb_stream"]
    assert value["bytes_written"] == 6 and path.read_bytes() == b"123456"


@pytest.mark.parametrize("existing", ["raw", "metadata"])
def test_no_overwrite(tmp_path, existing):
    path = tmp_path / "frame.rgb"
    target = path if existing == "raw" else path.with_suffix(".rgb.json")
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        with capture(path):
            pass
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize("payload", [b"12345", b"1234567"])
def test_wrong_frame_bytes_rejected(tmp_path, payload):
    with pytest.raises(ValueError, match="size/count"):
        with capture(tmp_path / "frame.rgb") as recorder:
            recorder.write_frame(payload)


def test_extra_frame_is_rejected_without_writing_extra_bytes(tmp_path):
    path = tmp_path / "frame.rgb"
    with pytest.raises(ValueError, match="size/count"):
        with capture(path, frames=1) as recorder:
            recorder.write_frame(b"123456")
            recorder.write_frame(b"abcdef")
    assert path.read_bytes() == b"123456"


def test_insufficient_disk_headroom_does_not_create_raw_file(tmp_path, monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_rgb_capture as module
    from types import SimpleNamespace
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="headroom"):
        with capture(tmp_path / "frame.rgb"):
            pass
    assert not (tmp_path / "frame.rgb").exists()
