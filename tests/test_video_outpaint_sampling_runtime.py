from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_noise import outpaint_window_noise
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_source_store import OutpaintSourceStore
from h3_audio_t8_pkg.video_outpaint_window_store import OutpaintWindowStore
from h3_audio_t8_pkg.video_outpaint_sampling_runtime import sample_prepared_outpaint_windows
from test_video_outpaint_sampling import _plan, _backend


def _execution(tmp_path, cuts=(), noise_algorithm="t8.outpaint.coordinate_noise/v1"):
    plan = _plan(cuts)
    source = OutpaintSourceStore(tmp_path / "source", plan, video_vae_sha256="b"*64)
    for shot, start, stop in source.expected:
        if source.position() != (shot, start, stop):
            continue
        chunk = torch.zeros((1, 24, stop-start, 6, 6))
        chunk[:, :, :, 2:4, 2:4] = 2
        source.append(shot, start, chunk)
    identity = {"model_sha256": "c"*64, "conditioning_sha256": "d"*64, "audio_source_sha256": "e"*64,
                "source_cache_sha256": hashlib.sha256(source.path.read_bytes()).hexdigest(),
                "seed": 42, "steps": 20, "sampler_name": "res_multistep", "scheduler": "simple",
                "noise_algorithm": noise_algorithm}
    windows = OutpaintWindowStore(tmp_path / "windows", plan, execution_identity=identity)

    def audio(shot, local):
        frames = plan["shots"][shot]["windows"][local]["render_frames"]
        return torch.full((1, 32, 2, round(frames/24*40)), 3.0)

    return {"model": object(), "conditioning_for_window": lambda s, w: [[torch.zeros(1), {}]],
            "audio_for_window": audio, "source_store": source, "window_store": windows,
            "verify_execution": lambda: windows.identity, "sample_function": _backend}


def test_coordinate_noise_preserves_rng_and_exact_window_overlap():
    plan = _plan()
    before = torch.random.get_rng_state().clone()
    video0, audio0 = outpaint_window_noise(plan, 0, 0, 42)
    video1, audio1 = outpaint_window_noise(plan, 0, 1, 42)
    assert torch.equal(before, torch.random.get_rng_state())
    second = plan["shots"][0]["windows"][1]
    assert torch.equal(video0[:, :, second["video_start"]:], video1[:, :, :second["context_video_latents"]])
    assert torch.equal(audio0[..., second["audio_start"]:], audio1[..., :second["context_audio_latents"]])
    smaller = build_outpaint_plan(**dict(plan["request"], window_frames=39))
    short_video, short_audio = outpaint_window_noise(smaller, 0, 0, 42)
    assert torch.equal(video0[:, :, :short_video.shape[2]], short_video)
    assert torch.equal(audio0[..., :short_audio.shape[-1]], short_audio)
    assert not torch.equal(video0, outpaint_window_noise(plan, 0, 0, 43)[0])


@pytest.mark.parametrize("algorithm", ["t8.outpaint.coordinate_noise/v1", "t8.outpaint.native_cpu_noise/v1"])
def test_serial_sampling_resume_and_context_are_identical_to_uninterrupted(tmp_path, algorithm):
    original = _execution(tmp_path / "original", noise_algorithm=algorithm)
    report = sample_prepared_outpaint_windows(**original)
    assert report["all_windows_sampled"] and not report["generated_video_complete"]
    interrupted = _execution(tmp_path / "interrupted", noise_algorithm=algorithm)

    def cancel(_):
        raise RuntimeError("cancel after commit")

    with pytest.raises(RuntimeError, match="cancel"):
        sample_prepared_outpaint_windows(**interrupted, progress=cancel)
    store = interrupted["window_store"]
    assert len(store.snapshot()["committed"]) == 1 and store.snapshot()["status"] == "interrupted"
    with pytest.raises(ValueError, match="explicit resume"):
        sample_prepared_outpaint_windows(**interrupted)
    report = sample_prepared_outpaint_windows(**interrupted, resume=True)
    assert report["resume_from"] == 1 and report["windows_sampled_this_call"] == 1
    for index in range(2):
        assert all(torch.equal(a, b) for a, b in zip(original["window_store"].load(index), store.load(index)))
    assert sample_prepared_outpaint_windows(**interrupted)["windows_sampled_this_call"] == 0


def test_noise_mode_cannot_silently_reuse_an_existing_sampling_store(tmp_path):
    _execution(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        _execution(tmp_path, noise_algorithm="t8.outpaint.native_cpu_noise/v1")


def test_scene_cut_resets_context_and_original_audio_is_never_generated(tmp_path):
    inputs = _execution(tmp_path, cuts=(45,))
    sample_prepared_outpaint_windows(**inputs)
    store = inputs["window_store"]
    assert store.context_for(1) is None
    for index in range(2):
        video, audio = store.load(index)
        # Each 45-frame shot has 13 fully observed tokens (43 frames). The
        # partial final token and hold-padding are not original observations.
        assert torch.all(video[:, :, :13, 2:4, 2:4] == 2)
        assert torch.all(video[:, :, 13:, 2:4, 2:4] == 42)
        assert torch.all(audio == 3)


def test_tampered_saved_output_or_live_identity_is_not_reused(tmp_path):
    inputs = _execution(tmp_path)
    inputs["verify_execution"] = lambda: {}
    with pytest.raises(ValueError, match="identity"):
        sample_prepared_outpaint_windows(**inputs)
    store = inputs["window_store"]
    inputs["verify_execution"] = lambda: store.identity
    sample_prepared_outpaint_windows(**inputs)
    next(store.root.glob("window-*.safetensors")).write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        sample_prepared_outpaint_windows(**inputs)


def test_failed_sampler_leaves_window_uncommitted_and_can_retry(tmp_path):
    inputs = _execution(tmp_path)

    def fail(*_, **__):
        raise RuntimeError("sampler failed")

    inputs["sample_function"] = fail
    with pytest.raises(RuntimeError, match="sampler failed"):
        sample_prepared_outpaint_windows(**inputs)
    store = inputs["window_store"]
    assert store.snapshot()["committed"] == []
    inputs["sample_function"] = _backend
    assert sample_prepared_outpaint_windows(**inputs, resume=True)["all_windows_sampled"]


def test_abrupt_process_exit_releases_lease_and_preserves_committed_window(tmp_path):
    worker = Path(__file__).with_name("outpaint_process_exit_worker.py")
    stopped = subprocess.run([sys.executable, str(worker), str(tmp_path)],
                              capture_output=True, text=True, timeout=90)
    assert stopped.returncode == 43, stopped.stdout + stopped.stderr
    inputs = _execution(tmp_path)
    store = inputs["window_store"]
    assert store.snapshot()["status"] == "running"
    assert len(store.snapshot()["committed"]) == 1
    before = store.load(0)
    report = sample_prepared_outpaint_windows(**inputs, resume=True)
    assert report["resume_from"] == 1 and report["windows_sampled_this_call"] == 1
    assert all(torch.equal(a, b) for a, b in zip(before, store.load(0)))


def test_global_decode_reads_use_committed_window_ownership(tmp_path):
    inputs = _execution(tmp_path)
    sample_prepared_outpaint_windows(**inputs)
    store = inputs["window_store"]
    first, _ = store.load(0)
    second, _ = store.load(1)
    full = torch.cat((first, second[:, :, -5:]), dim=2)
    assert torch.equal(store.read_video_range(0, 19, 26), full[:, :, 19:26])
    assert torch.equal(store.read_video_range(0, 0, 7), full[:, :, :7])


def test_masked_audio_provider_keeps_generated_context_across_resume(tmp_path):
    inputs = _execution(tmp_path)
    original = inputs["audio_for_window"]

    def provider(shot, window):
        samples = original(shot, window)
        return {"samples": samples, "noise_mask": torch.ones((1, 1, 2, samples.shape[-1]))}

    inputs["audio_for_window"] = provider

    def cancel(_):
        raise RuntimeError("cancel after first commit")

    with pytest.raises(RuntimeError, match="cancel"):
        sample_prepared_outpaint_windows(**inputs, progress=cancel)
    store = inputs["window_store"]
    assert torch.all(store.load(0)[1] == 99)
    assert sample_prepared_outpaint_windows(**inputs, resume=True)["windows_sampled_this_call"] == 1
    assert torch.all(store.load(1)[1] == 99)
