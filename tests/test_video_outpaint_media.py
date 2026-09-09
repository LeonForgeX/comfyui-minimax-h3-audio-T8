from __future__ import annotations

import shutil
import subprocess
import json

import pytest

from comfy_api.input_impl import VideoFromFile
from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source, finalize_outpaint_file, validate_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan


def _clip(path, width, height, *, frames=5, sound=False, fps=24):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg needed for real CPU file-contract tests")
    command = [ffmpeg, "-v", "error", "-n", "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:r={fps}"]
    if sound:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=32000:duration={frames/fps}", "-c:a", "aac"]
    command += ["-frames:v", str(frames), "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(path)]
    subprocess.run(command, check=True, capture_output=True, timeout=30)
    return VideoFromFile(str(path))


def _source(tmp_path, *, sound=True):
    source = _clip(tmp_path / "source.mp4", 64, 48, sound=sound)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=64, height=48, frame_count=5,
                              aspect="custom", left=16, right=16)
    return inspection, plan


@pytest.mark.parametrize("sound", [False, True])
def test_real_mp4_finalizer_preserves_source_audio_timing_and_reusable_video(tmp_path, sound):
    inspection, plan = _source(tmp_path, sound=sound)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    output, report = finalize_outpaint_file(candidate, inspection, plan, tmp_path / "result.mp4")
    assert output.get_frame_count() == 5
    assert output.get_dimensions() == (96, 48)
    assert output.get_frame_count() == 5  # Reusable, no one-shot generation hidden in save.
    assert report["media"]["audio_packet_payload_exact"]
    assert report["media"]["audio_packet_timeline_exact"]
    assert report["media"]["audio_decoded_pcm_exact"]
    assert report["media"]["audio_decoded_timeline_exact"]
    assert not report["perceptual_acceptance"]
    assert not list(tmp_path.glob(".*outpaint-*.mp4"))


@pytest.mark.parametrize("width,frames,sound", [(64, 5, False), (96, 6, False), (96, 5, True)])
def test_wrong_geometry_padding_or_candidate_audio_not_published(tmp_path, width, frames, sound):
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, width, 48, frames=frames, sound=sound)
    with pytest.raises(ValueError):
        finalize_outpaint_file(candidate, inspection, plan, tmp_path / "result.mp4")
    assert not (tmp_path / "result.mp4").exists()


def test_non24fps_has_no_silent_retiming(tmp_path):
    source = _clip(tmp_path / "source.mp4", 64, 48, fps=30)
    with pytest.raises(ValueError, match="24fps"):
        inspect_outpaint_source(source)


def test_changed_source_and_plan_cannot_reuse_inspection(tmp_path):
    inspection, plan = _source(tmp_path)
    other = build_outpaint_plan(**dict(plan["request"], source_sha256="b" * 64))
    with pytest.raises(ValueError, match="match"):
        validate_outpaint_source(inspection, other)
    with (tmp_path / "source.mp4").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_outpaint_source(inspection, plan)


def test_candidate_mutation_during_inspection_is_retained_before_mux(tmp_path, monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_media as module
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    initial = module._sha256_file(candidate)
    original = module._file_source_contract

    def mutate(video, **kwargs):
        result = original(video, **kwargs)
        with candidate.open("ab") as handle:
            handle.write(b"injected change")
        return result

    monkeypatch.setattr(module, "_file_source_contract", mutate)
    with pytest.raises(ValueError, match="changed during media inspection"):
        finalize_outpaint_file(candidate, inspection, plan, tmp_path / "result.mp4")
    diagnostic = json.loads(next(tmp_path.glob(".outpaint-failure-*/diagnostic.json")).read_text())
    assert diagnostic["candidate_sha256_before_media_inspection"] == initial
    assert diagnostic["stage"] == "candidate_changed_during_media_inspection"
    assert diagnostic["files"]["video_only"]["sha256"] != initial
    assert not (tmp_path / "result.mp4").exists()


def test_cancellation_and_existing_target_never_overwrite(tmp_path):
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    target = tmp_path / "result.mp4"
    calls = 0

    def interrupt():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("user cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        finalize_outpaint_file(candidate, inspection, plan, target, interrupt_check=interrupt)
    assert not target.exists()
    assert not list(tmp_path.glob(".*outpaint-*.mp4"))
    target.write_bytes(b"user result")
    with pytest.raises(FileExistsError):
        finalize_outpaint_file(candidate, inspection, plan, target)
    assert target.read_bytes() == b"user result"


def test_pixel_receipt_from_another_candidate_cannot_be_published(tmp_path):
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    with pytest.raises(ValueError, match="pixel receipt"):
        finalize_outpaint_file(candidate, inspection, plan, tmp_path / "result.mp4",
                               pixel_receipt={"candidate_sha256": "a"*64, "receipt_sha256": "b"*64})
    assert not (tmp_path / "result.mp4").exists()


def test_caller_metadata_cannot_override_actual_validation_result(tmp_path):
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    with pytest.raises(ValueError, match="metadata cannot replace"):
        finalize_outpaint_file(candidate, inspection, plan, tmp_path / "result.mp4",
                               delivery_metadata={"pre_encode_pixel_evidence_verified": True})
    assert not (tmp_path / "result.mp4").exists()
    assert not (tmp_path / "result.mp4.outpaint.json").exists()


@pytest.mark.parametrize("stage", ["encoded_video_before_mux", "muxed_video_and_audio"])
def test_validation_failure_retains_exact_private_media_without_publication(tmp_path, monkeypatch, stage):
    import h3_audio_t8_pkg.video_outpaint_media as module
    inspection, plan = _source(tmp_path)
    candidate = tmp_path / "candidate.mp4"
    _clip(candidate, 96, 48)
    original = candidate.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError("injected strict media failure")
    key = "_strict_validate_encoded_video" if stage == "encoded_video_before_mux" else "_validate_final_file"
    monkeypatch.setattr(module, key, fail)
    target = tmp_path / "result.mp4"
    with pytest.raises(RuntimeError, match="Diagnostic:"):
        finalize_outpaint_file(candidate, inspection, plan, target)
    assert not target.exists() and not target.with_suffix(".mp4.outpaint.json").exists()
    folders = list(tmp_path.glob(".outpaint-failure-*"))
    assert len(folders) == 1
    report = json.loads((folders[0] / "diagnostic.json").read_text(encoding="utf-8"))
    assert report["stage"] == stage and not report["final_video_published"]
    assert report["status"] == "failed_not_deliverable" and report["plan_sha256"] == plan["plan_sha256"]
    assert (folders[0] / "video_only.mp4").read_bytes() == original == candidate.read_bytes()
    assert ("muxed" in report["files"]) == (stage == "muxed_video_and_audio")
    assert not list(tmp_path.glob(".*outpaint-*.mp4"))
