"""Failure exits must release only the failed stage and preserve its error."""
import pytest
import torch
from comfy.model_management import InterruptProcessingException
from h3_audio_t8_pkg import long_video_dual_model_stages as stages
from h3_audio_t8_pkg import long_video_dual_residency as residency
from test_prompt_relay_core_compat import model_fixture
from test_long_video_dual_model_stages import latent


@pytest.mark.parametrize("kind", ["exception", "cancel", "missing_x0"])
def test_failed_started_stage_releases_model_and_keeps_original_error(monkeypatch, kind):
    model = model_fixture()
    calls = []
    expected = InterruptProcessingException() if kind == "cancel" else RuntimeError("fixture stage failure")
    def sample(*args, **kwargs):
        if kind in ("exception", "cancel"):
            raise expected
        return latent(9.)
    monkeypatch.setattr(stages, "_sample_prepared_segment", sample)
    monkeypatch.setattr(residency, "release_stage_residency", lambda target: calls.append(target))
    with pytest.raises(BaseException) as caught:
        stages.sample_model_stage(model, [], latent(0.), sampler=None, sigmas=torch.tensor([1., 0.]), seed=0,
            output_kind="denoised_x0")
    if kind in ("exception", "cancel"):
        assert caught.value is expected
    else:
        assert isinstance(caught.value, RuntimeError)
    assert calls == [model]


def test_unobserved_forward_is_reported_without_unloading_successful_stage(monkeypatch):
    model = model_fixture()
    calls = []
    warnings = []
    monkeypatch.setattr(stages, "_sample_prepared_segment", lambda *args, **kwargs: latent(9.))
    monkeypatch.setattr(stages, "warn_patch_stack", warnings.append)
    monkeypatch.setattr(residency, "release_stage_residency", lambda target: calls.append(target))
    output, report = stages.sample_model_stage(
        model, [], latent(0.), sampler=None, sigmas=torch.tensor([1., 0.]), seed=0,
        output_kind="zero_sigma_output")
    assert output["samples"] is not None
    assert report["completed_network_forwards"] == 0
    assert report["forward_evidence_complete"] is False
    assert any("observer coverage differs" in warning for warning in warnings)
    assert calls == []


def test_cleanup_failure_does_not_replace_original_sampling_exception(monkeypatch):
    expected = RuntimeError("original sampling failure")
    def sample(*args, **kwargs):
        raise expected
    def cleanup(*args):
        raise RuntimeError("cleanup also failed")
    monkeypatch.setattr(stages, "_sample_prepared_segment", sample)
    monkeypatch.setattr(residency, "release_stage_residency", cleanup)
    with pytest.raises(RuntimeError) as caught:
        stages.sample_model_stage(model_fixture(), [], latent(0.), sampler=None,
            sigmas=torch.tensor([1., 0.]), seed=0)
    assert caught.value is expected
    assert "cleanup also failed" in " ".join(getattr(caught.value, "__notes__", []))


def test_input_rejection_does_not_unload_an_unstarted_stage(monkeypatch):
    monkeypatch.setattr(residency, "release_stage_residency", lambda *args: pytest.fail("Not started"))
    with pytest.raises(ValueError):
        stages.sample_model_stage(model_fixture(), [], latent(0.), sampler=None,
            sigmas=torch.tensor([1., 1.]), seed=0)
