"""Public stage-pair CPU oracles; trained lifter/VAEs and GPU quality NOT_RUN."""
import copy
import json

import comfy.nested_tensor
import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import long_video, progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8
from h3_audio_t8_pkg.progressive_stage_inputs import prepare_stage_inputs
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE, make_audio
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_continuation import accepted, capture  # noqa: F401


def inputs(source, *, task="t2va", audio_mode="native"):
    options = dict(audio_mode=audio_mode, add_source_as_reference=audio_mode == "reference_only",
        drive_audio=None if audio_mode == "native" else make_audio(6))
    if task == "fl2va":
        options["last_frame"] = torch.ones(1, 64, 128, 3) * .25
    if task == "ref2va":
        options["ref_images"] = {"image_1": torch.ones(1, 64, 128, 3) * .4}
    low, high, _ = source.prepare_conditions(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt="Continue the shot.", length=124, **options)
    return low, high


def run(low, high, task, callback=None):
    models = [long_video.patch_long_video_model(tiny_model()) for _ in range(2)]
    before = [copy.deepcopy(model.model_options) for model in models]
    result = runtime.sample_progressive_h3(models[0], high[0], high[0], high[1],
        comfy.samplers.ksampler("euler"), native_flow_sigmas(4, 12.),
        upscaler_model="test", seed=9, model_hires=models[1], low_evaluations=2, task=task,
        input_mode="prepared_pair_exp", av_latent_low=low[1],
        positive_low=low[0], negative_low=low[0], callback=callback)
    for model, original in zip(models, before):
        assert model.model_options == original
    return result


@pytest.mark.parametrize("task,frames,audio_mode", [
    ("t2va", 5, "native"), ("i2va", 22, "native"),
    ("fl2va", 39, "native"), ("ref2va", 22, "native"),
    ("t2va", 22, "lock_source"), ("ref2va", 22, "remix_source"),
    ("ref2va", 22, "reference_only"),
])
def test_public_pair_runs_native_motion_and_source_audio(accepted, stub_lifter, task, frames, audio_mode):  # noqa: F811
    low, high = inputs(capture(accepted, context_frames=frames), task=task, audio_mode=audio_mode)
    source_video, source_audio = high[1]["samples"].unbind()
    output, receipt = run(low, high, task)
    report = json.loads(receipt)
    assert report["counts"]["callbacks"] == {"low": 2, "high": 2}
    assert report["prepared_stage_inputs"]["stage_masks_independent"]
    assert report["prepared_stage_inputs"]["audio_time_resampled"] is False
    video, audio = output["samples"].unbind()
    steps = {5: 2, 22: 7, 39: 12}[frames]
    torch.testing.assert_close(video[:, :, :steps], source_video[:, :, :steps], rtol=0, atol=2e-6)
    if audio_mode == "lock_source":
        torch.testing.assert_close(audio, source_audio, rtol=0, atol=2e-6)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert len(stub_lifter) == 1


@pytest.mark.parametrize("fault", ["low_canvas", "low_audio_time", "low_keyframe", "mask", "missing"])
def test_invalid_public_pair_refuses_before_lift(accepted, stub_lifter, fault):  # noqa: F811
    low, high = inputs(capture(accepted))
    low = list(low)
    if fault in {"low_canvas", "low_audio_time"}:
        v, a = low[1]["samples"].unbind()
        low[1] = {**low[1], "samples": comfy.nested_tensor.NestedTensor((
            v[..., :2] if fault == "low_canvas" else v,
            a[..., :-1] if fault == "low_audio_time" else a))}
    elif fault == "low_keyframe":
        low[0] = copy.deepcopy(low[0])
        low[0][0][1]["minimax_keyframes"][0]["latent"] = torch.zeros(1, 24, 1, 4, 8)
    elif fault == "mask":
        low[1] = {**low[1], "noise_mask": torch.tensor([[float("nan")]])}
    else:
        low[0] = None
    with pytest.raises(ValueError):
        run(low, high, "t2va")
    assert not stub_lifter


def test_stage_sources_are_not_resized_or_replaced(accepted):  # noqa: F811
    low, high = inputs(capture(accepted))
    video = low[1]["samples"].unbind()[0]
    video.fill_(.37)
    before = video.clone()
    _, low_mask, high_mask = prepare_stage_inputs(high[1], low[1], high[0], high[0], low[0], low[0],
        native_flow_sigmas(4, 12.), low_evaluations=2, low_scale=.5, task="t2va")
    assert torch.equal(video, before)
    assert low_mask is None and high_mask is not None


def test_public_v3_bridge_appends_prepared_inputs():
    fields = MiniMaxH3ProgressiveSamplerEXPT8.INPUT_TYPES()
    optional = fields["optional"]
    assert list(optional)[-4:] == ["input_mode", "av_latent_low", "positive_low", "negative_low"]
    assert "prepared_pair_exp" in optional["input_mode"][1]["options"]
