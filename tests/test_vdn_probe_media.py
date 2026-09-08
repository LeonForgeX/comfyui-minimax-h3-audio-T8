import json
import shutil
import subprocess

import pytest
import torch

from tools.vdn_probe_extension.media import save_raw_av


def audio():
    t = torch.arange(32000, dtype=torch.float32) / 32000
    return {"sample_rate": 32000, "waveform": (0.05 * torch.sin(t * 440 * 6.283185))[None, None].repeat(1, 2, 1)}


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_raw_av_safe_save_real_encoder_and_no_overwrite(tmp_path):
    frames = torch.linspace(0, 1, 24 * 32 * 64 * 3).reshape(24, 32, 64, 3)
    target = tmp_path / "safe.mp4"
    report = save_raw_av(frames, audio(), target)
    assert report["status"] == "pass" and len(report["source_rgb8_sha256"]) == 64
    assert report["audio"]["encoded_samples"] == 32000
    for threads in (1, 4):
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-xerror", "-err_detect", "explode",
                        "-threads", str(threads), "-i", str(target), "-f", "null", "-"],
                       check=True, capture_output=True, timeout=40)
    streams = json.loads(subprocess.check_output(
        [shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(target)]))["streams"]
    assert streams[0]["width"] == 64 and streams[0]["height"] == 32
    assert int(streams[0]["nb_frames"]) == 24 and streams[0]["has_b_frames"] == 0
    assert streams[1]["codec_name"] == "aac"
    original = target.read_bytes()
    with pytest.raises(ValueError, match="new MP4"):
        save_raw_av(frames, audio(), target)
    assert target.read_bytes() == original


@pytest.mark.parametrize("kind", ["odd", "empty", "audio_clip", "audio_nan", "sample_rate"])
def test_safe_save_rejects_invalid_inputs_before_publication(tmp_path, kind):
    frames, sound = torch.zeros(2, 32, 64, 3), audio()
    if kind == "odd":
        frames = frames[:, :31]
    elif kind == "empty":
        frames = frames[:0]
    elif kind == "audio_clip":
        sound["waveform"][..., 0] = 2.
    elif kind == "audio_nan":
        sound["waveform"][..., 0] = float("nan")
    else:
        sound["sample_rate"] = 0
    target = tmp_path / "invalid.mp4"
    with pytest.raises(ValueError):
        save_raw_av(frames, sound, target)
    assert not target.exists()
