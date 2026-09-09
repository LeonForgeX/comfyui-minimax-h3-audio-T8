import hashlib

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_audio_runtime import prepare_outpaint_audio_cache, OutpaintAudioProvider
from h3_audio_t8_pkg.video_outpaint_audio_store import OutpaintAudioStore
from h3_audio_t8_pkg.video_outpaint_sampling_runtime import sample_prepared_outpaint_windows
from h3_audio_t8_pkg.video_outpaint_source_store import OutpaintSourceStore
from h3_audio_t8_pkg.video_outpaint_window_store import OutpaintWindowStore
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from test_video_outpaint_audio import ManagedAudio
from test_video_outpaint_audio_file import _inputs
from test_video_outpaint_media import _clip
from test_video_outpaint_sampling import _backend


@pytest.fixture
def managed(monkeypatch):
    import comfy.model_management as management
    monkeypatch.setattr(management, "load_models_gpu", lambda *a, **kw: None)
    return ManagedAudio()


def test_audio_cache_cancel_resume_matches_full_and_complete_cache_skips_encoder(tmp_path, managed):
    inspection, plan = _inputs(tmp_path)
    complete, _ = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "full", block_tokens=8)

    def cancel(report):
        if report["new_chunks"] == 1:
            raise RuntimeError("cancel first audio commit")

    with pytest.raises(RuntimeError, match="cancel"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "resume", block_tokens=8, progress=cancel)
    with pytest.raises(ValueError, match="explicit resume"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "resume", block_tokens=8)
    resumed, report = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "resume",
                                                  block_tokens=8, resume=True)
    assert report["resume_from"] == 1 and report["replayed_prefix_chunks"] == 1
    assert report["chunks_encoded_this_call"] == len(resumed.store.expected)-1
    for shot, total in enumerate(resumed.store.totals):
        assert torch.equal(complete.store.read_range(shot, 0, total), resumed.store.read_range(shot, 0, total))
    loads = len(managed.memory_shapes)
    same, report = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "resume", block_tokens=8)
    assert report["chunks_encoded_this_call"] == report["replayed_prefix_chunks"] == 0
    assert len(managed.memory_shapes) == loads and same.verify() == resumed.verify()
    assert not list((tmp_path / "resume").glob("t8-outpaint-*"))


def test_silent_source_needs_no_vae_and_all_audio_is_unobserved(tmp_path):
    inspection, plan = _inputs(tmp_path, sound=False)
    provider, report = prepare_outpaint_audio_cache(None, inspection, plan, tmp_path / "cache")
    assert not report["source_has_audio"] and report["source_audio_prepared"]
    for shot in range(2):
        data = provider(shot, 0)
        assert torch.count_nonzero(data["samples"]) == 0
        assert torch.all(data["noise_mask"] == 1)
    assert not list((tmp_path / "cache").glob("audio-*.safetensors"))


def test_audio_padding_and_overlaps_feed_serial_sampler_with_real_file_and_tiny_vae(tmp_path, managed):
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=80, sound=True)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=32, height=32, frame_count=80,
                              aspect="custom", left=32, right=32, window_frames=73)
    audio, _ = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "audio", block_tokens=16)
    sources = OutpaintSourceStore(tmp_path / "video", plan, video_vae_sha256="b"*64)
    for shot, start, stop in sources.expected:
        tensor = torch.zeros((1, 24, stop-start, 2, 6))
        tensor[..., 2:4] = 2
        sources.append(shot, start, tensor)
    windows = OutpaintWindowStore(tmp_path / "windows", plan, execution_identity={
        "model_sha256": "c"*64, "conditioning_sha256": "d"*64,
        "source_cache_sha256": hashlib.sha256(sources.path.read_bytes()).hexdigest(),
        "audio_source_sha256": audio.verify(), "seed": 42, "steps": 20,
        "sampler_name": "res_multistep", "scheduler": "simple", "noise_algorithm": "t8.outpaint.coordinate_noise/v1"})

    def verify():
        assert audio.verify() == windows.identity["audio_source_sha256"]
        return windows.identity

    result = sample_prepared_outpaint_windows(model=object(), conditioning_for_window=lambda s, w: [[torch.zeros(1), {}]],
        audio_for_window=audio, source_store=sources, window_store=windows, verify_execution=verify, sample_function=_backend)
    assert result["all_windows_sampled"] and not result["generated_video_complete"]
    for index, (_, local) in enumerate(windows.windows):
        provided = audio(0, local)
        _, actual = windows.load(index)
        observed = (provided["noise_mask"] == 0).expand_as(actual)
        assert torch.equal(actual[observed], provided["samples"][observed])
        assert torch.all(actual[~observed] == 99)
    assert torch.count_nonzero(audio(0, 1)["noise_mask"]) > 0


def test_changed_weights_or_corrupted_assets_cannot_be_reused(tmp_path, managed):
    inspection, plan = _inputs(tmp_path)
    provider, _ = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache")
    with torch.no_grad():
        managed.first_stage_model.latents_mean[0] += 0.1
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache")
    asset = next(provider.store.root.glob("audio-*.safetensors"))
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="truncated"):
        OutpaintAudioProvider(inspection, provider.store)


def test_mid_encode_weight_change_invalidates_committed_cache(tmp_path, managed):
    inspection, plan = _inputs(tmp_path)

    def mutate(report):
        if report["new_chunks"] == 1:
            with torch.no_grad():
                managed.first_stage_model.latents_mean[0] += 0.1

    with pytest.raises(ValueError, match="loaded audio VAE changed"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache", block_tokens=8, progress=mutate)
    assert (tmp_path / "cache/invalid_audio_preparation.json").exists()
    with pytest.raises(ValueError, match="invalidated"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache", block_tokens=8, resume=True)


def test_mid_encode_pcm_change_invalidates_even_completed_latents(tmp_path, managed):
    inspection, plan = _inputs(tmp_path)

    def mutate(report):
        if report["new_chunks"] == 1:
            path = next((tmp_path / "cache").glob("t8-outpaint-source-audio-*/*.f32le"))
            with path.open("r+b") as stream:
                stream.write(b"bad!")

    with pytest.raises(ValueError, match="conditioning audio changed"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache", progress=mutate)
    assert (tmp_path / "cache/invalid_audio_preparation.json").exists()


def test_store_rejects_order_wrong_shape_and_settings_change(tmp_path):
    inspection, plan = _inputs(tmp_path)
    settings = dict(audio_vae_sha256="a"*64, pcm_sha256="b"*64, stream_position=0,
                    encoding_device="cpu", block_tokens=8)
    store = OutpaintAudioStore(tmp_path / "cache", plan, **settings)
    store.begin()
    with pytest.raises(ValueError, match="serial order"):
        store.append(0, 8, torch.zeros(1, 32, 2, 8))
    with pytest.raises(ValueError, match="float32 shape"):
        store.append(0, 0, torch.zeros(1, 32, 2, 7))
    store.append(0, 0, torch.zeros(1, 32, 2, 8))
    with pytest.raises(ValueError, match="completely prepared"):
        store.read_range(0, 0, 10)
    with pytest.raises(ValueError, match="identity mismatch"):
        OutpaintAudioStore(tmp_path / "cache", plan, **dict(settings, block_tokens=16))


def test_asset_written_before_failed_manifest_is_reused_without_overwrite(tmp_path, managed, monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_audio_store as module
    inspection, plan = _inputs(tmp_path)
    write = module._atomic_write_bytes
    failed = False

    def fail_once(path, data):
        nonlocal failed
        if path.name == "outpaint_source_audio.json" and b'"chunks":[{' in data and not failed:
            failed = True
            raise OSError("simulated manifest commit failure")
        return write(path, data)

    monkeypatch.setattr(module, "_atomic_write_bytes", fail_once)
    with pytest.raises(OSError, match="commit failure"):
        prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache", block_tokens=8)
    asset = next((tmp_path / "cache").glob("audio-*.safetensors"))
    original = asset.read_bytes()
    timestamp = asset.stat().st_mtime_ns
    provider, report = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache",
                                                  block_tokens=8, resume=True)
    assert report["resume_from"] == 0 and provider.store.snapshot()["status"] == "prepared"
    assert asset.read_bytes() == original and asset.stat().st_mtime_ns == timestamp


def test_bound_provider_rejects_source_change(tmp_path, managed):
    inspection, plan = _inputs(tmp_path)
    provider, _ = prepare_outpaint_audio_cache(managed, inspection, plan, tmp_path / "cache")
    with (tmp_path / "source.mp4").open("ab") as stream:
        stream.write(b"source modified after provider construction")
    with pytest.raises(ValueError, match="changed since inspection"):
        provider(0, 0)
