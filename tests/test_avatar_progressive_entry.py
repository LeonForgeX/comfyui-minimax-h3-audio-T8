import json

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg.avatar_progressive_entry import sample_avatar_progressive
from h3_audio_t8_pkg.nodes_avatar_progressive import MiniMaxH3AvatarProgressiveEXPT8
from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8
from h3_audio_t8_pkg.sampling import native_flow_sigmas
import test_progressive_sampling_runtime as fixtures


stub_lifter = fixtures.stub_lifter


def inputs():
    av = fixtures.latent()
    video, audio = av["samples"].unbind()
    audio = torch.linspace(-.2, .2, audio.numel()).reshape_as(audio)
    av["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    av["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    return av


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("separate", [False, True])
def test_real_tiny_native_euler_source_audio_anchored_both_stages(stub_lifter, task, separate):
    model = fixtures.tiny_model()
    high = fixtures.tiny_model() if separate else None
    av = inputs()
    before = [t.clone() for t in av["samples"].unbind()]
    callbacks = []
    output, text = sample_avatar_progressive(model=model, model_hires=high,
        positive=fixtures.conditioning(task), negative=fixtures.conditioning(task), av_latent=av,
        sampler=comfy.samplers.ksampler("euler"), sigmas=native_flow_sigmas(8, 12),
        upscaler_model="labelled-test-lifter", seed=6, task=task, low_evaluations=4,
        callback=lambda *items: callbacks.append(items[0]))
    report = json.loads(text)
    assert report["counts"]["actual_forwards"] == {"low": 4, "high": 4}
    assert callbacks == list(range(8)) and len(stub_lifter) == 1
    torch.testing.assert_close(output["samples"].unbind()[1], before[1], atol=1e-6, rtol=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(av["samples"].unbind(), before))
    assert not model.wrappers and (high is None or not high.wrappers)
    assert report["avatar"]["source_audio_latent_max_abs_difference"] < 1e-6
    assert report["avatar"]["trained_model_quality_qualified"] is False
    assert report["highres_tiling"] is False
    mm.unload_all_models()  # isolated CPU test process only


@pytest.mark.parametrize("kind", ["missing", "video_only", "generated", "fractional"])
def test_wrong_audio_mode_rejected_before_model_or_lifter(kind):
    av = inputs()
    video, audio = av["samples"].unbind()
    if kind == "missing":
        av.pop("noise_mask")
    elif kind == "video_only":
        av["noise_mask"] = torch.ones_like(video)
    else:
        av["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video),
            torch.full_like(audio, 1 if kind == "generated" else .01)))
    with pytest.raises(ValueError):
        sample_avatar_progressive(av_latent=av)


def test_new_schema_default_does_not_mutate_original():
    old_before = MiniMaxH3ProgressiveSamplerEXPT8.INPUT_TYPES()
    schema = MiniMaxH3AvatarProgressiveEXPT8.define_schema()
    fields = {f.id: f for f in schema.inputs}
    assert fields["low_evaluations"].default == 4 and fields["task"].default == "i2va"
    assert MiniMaxH3ProgressiveSamplerEXPT8.INPUT_TYPES() == old_before
    assert "model_hires" in fields


def test_cancel_at_low_never_runs_high_and_next_task_recovers(stub_lifter):
    model = fixtures.tiny_model()
    settings = dict(model=model, positive=fixtures.conditioning(), negative=fixtures.conditioning(),
        av_latent=inputs(), sampler=comfy.samplers.ksampler("euler"), sigmas=native_flow_sigmas(8, 12),
        upscaler_model="labelled-test-lifter", seed=6, low_evaluations=4)

    def cancel(*args):
        raise InterruptedError("owned CPU cancel")

    with pytest.raises(InterruptedError, match="owned CPU cancel"):
        sample_avatar_progressive(**settings, callback=cancel)
    assert not model.wrappers and not stub_lifter
    _, report = sample_avatar_progressive(**settings)
    assert json.loads(report)["counts"]["actual_forwards"] == {"low": 4, "high": 4}
    mm.unload_all_models()


@pytest.mark.parametrize('with_relay', [False, True])
def test_avatar_eav_relay_each_stage_preserves_full_recording_anchor(stub_lifter, with_relay):
    from test_progressive_relay import paired
    model = fixtures.tiny_model()
    positive = fixtures.conditioning('i2va')
    if with_relay:
        model, positive, _ = paired(model, 'i2va')
    wrappers_before = {kind: {key: list(values) for key, values in group.items()}
                       for kind, group in model.wrappers.items()}
    av = inputs()
    original = [p.clone() for p in av['samples'].unbind()]
    settings = dict(model=model, positive=positive, negative=fixtures.conditioning('i2va'), av_latent=av,
        sampler=comfy.samplers.ksampler('euler'), sigmas=native_flow_sigmas(8, 12),
        upscaler_model='labelled-test-lifter', seed=6, task='i2va', low_evaluations=4,
        eav_tau=.2)
    baseline, _ = sample_avatar_progressive(**settings, eav_mode='disabled')
    control, _ = sample_avatar_progressive(**settings, eav_mode='report_only')
    applied, text = sample_avatar_progressive(**settings, eav_mode='apply_exp')
    report = json.loads(text)
    for a, b in zip(baseline['samples'].unbind(), control['samples'].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for phase in ('low', 'high'):
        stage = report['eav'][phase]
        assert stage['model_forward_count'] == stage['verified_native_mask_forwards'] == 4
        assert stage['attention_calls_per_active_forward'] == [1]*4
        if with_relay:
            assert report['prompt_relay'][phase]['completed_calls'] == {'forward': 4, 'routed_attention': 4}
    assert report['eav']['high']['forwards'][0]['progress_video'] > report['eav']['low']['forwards'][-1]['progress_video']
    torch.testing.assert_close(applied['samples'].unbind()[1], original[1], atol=1e-6, rtol=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(av['samples'].unbind(), original))
    assert not torch.equal(applied['samples'].unbind()[0], control['samples'].unbind()[0])
    assert not report['avatar']['spatial_tiling'] and model.wrappers == wrappers_before
    mm.unload_all_models()  # owned CPU test process, never the user's Core
