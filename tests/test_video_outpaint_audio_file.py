from fractions import Fraction
import shutil
import subprocess
import sys

import av
import numpy as np
import pytest
import torch
from comfy_api.input_impl import VideoFromFile

from h3_audio_t8_pkg.video_outpaint_audio import iter_encode_outpaint_audio
from h3_audio_t8_pkg.video_outpaint_audio_file import prepare_outpaint_audio_file
from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from test_video_outpaint_media import _clip
from test_video_outpaint_audio import TinyAudio


def _inputs(tmp_path, *, sound=True, delayed=False):
    path = tmp_path / "source.mp4"
    if delayed:
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-n", "-f", "lavfi", "-i",
                        "color=c=blue:s=64x48:r=24:duration=1.625", "-itsoffset", "0.25", "-f", "lavfi",
                        "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264",
                        "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                       check=True, capture_output=True, timeout=30)
        source = VideoFromFile(str(path))
    else:
        source = _clip(path, 64, 48, frames=39, sound=sound)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=64, height=48,
                              frame_count=39, aspect="custom", left=16, right=16, cut_frames=(17,))
    return inspection, plan


def _oracle(inspection):
    # Independent tiny-file in-memory timestamp reference; production is disk-backed.
    result = np.zeros((2, 52000), dtype=np.float32)
    with av.open(inspection["path"]) as container:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)
        frames = []
        for frame in container.decode(container.streams.audio[0]):
            frames.extend(resampler.resample(frame))
        frames.extend(resampler.resample(None))
        for frame in frames:
            position = round(Fraction(frame.pts)*Fraction(frame.time_base)*32000)
            first, last = max(0, -position), min(frame.samples, result.shape[-1]-position)
            if last > first:
                result[:, position+first:position+last] = frame.to_ndarray()[:, first:last]
    return torch.from_numpy(result).unsqueeze(0)


@pytest.mark.parametrize("delayed", [False, True])
def test_real_file_matches_timestamp_oracle_and_shot_partition(tmp_path, delayed):
    inspection, plan = _inputs(tmp_path, delayed=delayed)
    expected = _oracle(inspection)
    with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path) as audio:
        assert audio.sample_count == 52000 and audio.stream_position == 0
        torch.testing.assert_close(audio.read(0, 52000), expected, rtol=0, atol=0)
        a_count, a_reader = audio.shot(0)
        b_count, b_reader = audio.shot(1)
        assert a_count+b_count == 52000
        joined = torch.cat((a_reader(0, a_count), b_reader(0, b_count)), dim=-1)
        assert torch.equal(joined, expected)
        if delayed:
            assert torch.count_nonzero(joined[..., :5000]) == 0
            assert joined[..., 10000:25000].abs().mean() > 0.01
        path = audio.path
    assert not path.exists()


def test_real_source_to_bounded_native_audio_posterior(tmp_path):
    inspection, plan = _inputs(tmp_path)
    stage = TinyAudio().eval()
    with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path) as audio:
        for shot in range(2):
            count, reader = audio.shot(shot)
            with torch.inference_mode():
                expected = stage.encode(reader(0, count))
            actual = torch.cat([chunk for _, chunk in iter_encode_outpaint_audio(
                stage, reader, count, block_tokens=16, scratch_parent=tmp_path)], dim=-1)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert not list(tmp_path.glob("t8-outpaint-*"))


def test_silent_file_does_not_pretend_to_have_observed_audio(tmp_path):
    inspection, plan = _inputs(tmp_path, sound=False)
    with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path) as audio:
        assert audio is None
    assert not list(tmp_path.glob("t8-outpaint-*"))


def test_isolated_decoder_accepts_negative_fractional_video_origin(tmp_path):
    from h3_audio_t8_pkg.video_outpaint_audio_file import _decode_isolated
    inspection, _ = _inputs(tmp_path)
    target = tmp_path / "negative-origin.f32le"
    _decode_isolated(inspection["path"], target, Fraction(-1, 2), 52000, 0, None)
    actual = torch.from_numpy(np.fromfile(target, dtype="<f4").reshape(-1, 2).T).unsqueeze(0)
    expected = _oracle(inspection)
    assert torch.count_nonzero(actual[..., :16000]) == 0
    assert torch.equal(actual[..., 16000:], expected[..., :36000])


def test_canonical_audio_tamper_cancel_and_bad_track_are_explicit(tmp_path):
    inspection, plan = _inputs(tmp_path)
    with pytest.raises(ValueError, match="changed during use"):
        with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path) as audio:
            with audio.path.open("r+b") as stream:
                stream.write(b"bad!")
    assert not list(tmp_path.glob("t8-outpaint-*"))
    with pytest.raises(ValueError, match="audio_stream_position"):
        with prepare_outpaint_audio_file(inspection, plan, stream_position=1):
            pass

    def cancel():
        raise RuntimeError("cancel audio preparation")

    with pytest.raises(RuntimeError, match="cancel"):
        with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path, interrupt_check=cancel):
            pass
    assert not list(tmp_path.glob("t8-outpaint-*"))


def test_native_decoder_process_failure_is_reported_without_parent_crash(tmp_path, monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_audio_file as module
    inspection, plan = _inputs(tmp_path)
    popen = subprocess.Popen
    children = []

    def exit_child(*args, **kwargs):
        child = popen([sys.executable, "-I", "-c", "import os; os._exit(55)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", exit_child)
    with pytest.raises(RuntimeError, match="decode failed.*55"):
        with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path):
            pass
    assert len(children) == 1 and children[0].poll() == 55
    assert not list(tmp_path.glob("t8-outpaint-*"))


def test_cancel_running_decoder_reaps_only_the_owned_child(tmp_path, monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_audio_file as module
    inspection, plan = _inputs(tmp_path)
    popen = subprocess.Popen
    children = []

    def slow_child(*args, **kwargs):
        child = popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", slow_child)

    def cancel():
        if children:
            assert children[0].poll() is None
            raise RuntimeError("cancel running audio decoder")

    with pytest.raises(RuntimeError, match="cancel"):
        with prepare_outpaint_audio_file(inspection, plan, scratch_parent=tmp_path, interrupt_check=cancel):
            pass
    assert len(children) == 1 and children[0].poll() is not None
    assert not list(tmp_path.glob("t8-outpaint-*"))
