from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_source_runtime import loaded_video_vae_identity, prepare_outpaint_source_cache
from test_video_outpaint_media import _clip
from test_video_outpaint_prepare import NativeTemporalOracle, PublicVAE


class StatefulOracle(NativeTemporalOracle):
    def __init__(self):
        self.weight = torch.ones(1)
        self.buffer = torch.zeros(1)

    def state_dict(self):
        return {"weight": self.weight, "mean": self.latents_mean, "std": self.latents_std}

    def named_buffers(self):
        return [("nonpersistent_normalizer", self.buffer)]

    def _adaptive_encode(self, frames):
        return super()._adaptive_encode(frames) * self.weight + self.buffer


class StatefulVAE(PublicVAE):
    def __init__(self):
        super().__init__()
        self.first_stage_model = StatefulOracle()


def _input(tmp_path):
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=39)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(source_sha256=inspection["sha256"], width=32, height=32, frame_count=39,
                              aspect="custom", left=32, right=32)
    return inspection, plan


def test_loaded_identity_detects_weights_buffers_and_configuration():
    vae = StatefulVAE()
    first = loaded_video_vae_identity(vae)
    assert loaded_video_vae_identity(StatefulVAE()) == first
    vae.first_stage_model.weight += 1
    assert loaded_video_vae_identity(vae)["sha256"] != first["sha256"]
    vae = StatefulVAE()
    vae.first_stage_model.buffer += 1
    assert loaded_video_vae_identity(vae)["sha256"] != first["sha256"]
    vae = StatefulVAE()
    vae.vae_dtype = torch.float16
    assert loaded_video_vae_identity(vae)["sha256"] != first["sha256"]
    vae.patcher = SimpleNamespace(patches={"pending": 1})
    from h3_audio_t8_pkg.patch_stack_policy import model_identity_matches
    selected = loaded_video_vae_identity(vae)
    next_selected = loaded_video_vae_identity(vae)
    assert selected['portable_cache_reuse'] is False and selected['sha256'] != next_selected['sha256']
    assert model_identity_matches(selected, next_selected)
    vae.patcher.patches['pending'] = 2
    assert not model_identity_matches(selected, loaded_video_vae_identity(vae))


def test_real_encode_delegates_prior_callable_and_never_reuses_opaque_cache(tmp_path):
    from h3_audio_t8_pkg.video_outpaint_source_runtime import source_vae_identity_matches
    inspection, plan = _input(tmp_path)
    vae = StatefulVAE()
    original = vae.encode
    calls = []
    def prior(frames):
        calls.append(tuple(frames.shape))
        return original(frames)
    vae.encode = prior
    store, report = prepare_outpaint_source_cache(vae, inspection, plan, tmp_path / 'cache')
    assert len(calls) == 3 and report['all_source_video_chunks_prepared']
    assert vae.encode is prior and source_vae_identity_matches(store, loaded_video_vae_identity(vae))
    again, _ = prepare_outpaint_source_cache(vae, inspection, plan, tmp_path / 'cache')
    assert again.root != store.root and len(calls) == 6
    vae.first_stage_model.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='NaN or Inf'):
        loaded_video_vae_identity(vae)


def test_real_file_preparation_resumes_and_does_not_reencode_committed_chunks(tmp_path):
    inspection, plan = _input(tmp_path)
    root = tmp_path / "cache"
    vae = StatefulVAE()

    def cancel_after_commit(_):
        raise RuntimeError("cancelled after first commit")

    with pytest.raises(RuntimeError, match="cancelled"):
        prepare_outpaint_source_cache(vae, inspection, plan, root, progress=cancel_after_commit)
    assert len(vae.calls) == 1
    store, report = prepare_outpaint_source_cache(vae, inspection, plan, root)
    assert report["resume_from"] == (0, 5, 10)
    assert report["chunks_encoded_this_call"] == 2
    assert report["all_source_video_chunks_prepared"]
    assert store.read_range(0, 0, 12).shape == (1, 24, 12, 2, 6)
    _, again = prepare_outpaint_source_cache(vae, inspection, plan, root)
    assert again["chunks_encoded_this_call"] == 0
    assert len(vae.calls) == 3
    vae.first_stage_model.weight += 1
    with pytest.raises(ValueError, match="mismatch"):
        prepare_outpaint_source_cache(vae, inspection, plan, root)


def test_midrun_model_change_invalidates_cache_even_after_cancellation(tmp_path):
    inspection, plan = _input(tmp_path)
    vae = StatefulVAE()
    root = tmp_path / "cache"

    def mutate(_):
        vae.first_stage_model.weight += 1
        raise RuntimeError("cancel")

    with pytest.raises(ValueError, match="changed during"):
        prepare_outpaint_source_cache(vae, inspection, plan, root, progress=mutate)
    assert (root / "invalid_source_preparation.json").is_file()
    vae.first_stage_model.weight -= 1
    with pytest.raises(ValueError, match="invalidated"):
        prepare_outpaint_source_cache(vae, inspection, plan, root)
