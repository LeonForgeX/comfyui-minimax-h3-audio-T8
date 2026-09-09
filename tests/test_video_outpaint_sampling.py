from __future__ import annotations

import pytest
import torch
import comfy.nested_tensor

from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_sampling import (
    SEAM_CONTEXT_INSET_LATENTS,
    _seam_safe_source_lock_box,
    _source_noise_weights,
    sample_outpaint_window,
)


def _plan(cuts=()):
    return build_outpaint_plan(source_sha256="a" * 64, width=32, height=32, frame_count=90,
                              aspect="custom", left=32, right=32, top=32, bottom=32,
                              window_frames=73, cut_frames=cuts)


def _inputs(plan, shot=0, index=0):
    window = plan["shots"][shot]["windows"][index]
    vt = (window["render_frames"] - 5) // 17 * 5 + 2
    at = round(window["render_frames"] / 24 * 40)
    return {"model": object(), "conditioning": [[torch.zeros(1), {"other": "kept"}]],
            "plan": plan, "shot_index": shot, "window_index": index,
            "video_latent": torch.full((1, 24, vt, 6, 6), 2.0),
            "audio_latent": torch.full((1, 32, 2, at), 3.0),
            "video_noise": torch.ones((1, 24, vt, 6, 6)),
            "audio_noise": torch.ones((1, 32, 2, at))}


def _backend(*args, **kwargs):
    video, audio = args[8].unbind()
    return comfy.nested_tensor.NestedTensor((torch.full_like(video, 42), torch.full_like(audio, 99)))


def test_source_and_audio_are_preserved_even_if_backend_overwrites_everything():
    inputs = _inputs(_plan())
    source = inputs["video_latent"].clone()
    result, context, report = sample_outpaint_window(**inputs, sample_function=_backend)
    video, audio = result["samples"].unbind()
    assert torch.all(video[:, :, :, 2:4, 2:4] == 2)
    assert torch.all(video[:, :, :, :2] == 42)
    assert torch.all(audio == 3)
    assert torch.equal(source, inputs["video_latent"])
    assert context["target_window_index"] == 1
    assert report["deliver_stop"] == 73
    assert report["sampled_audio_discarded"]


@pytest.mark.parametrize("width,height,margins", [(96, 96, (32,32,32,32)),
                                                 (736,416,(0,96,0,96))])
def test_complete_source_edges_stay_locked_at_real_sizes(width, height, margins):
    left, top, right, bottom = margins
    plan = build_outpaint_plan(
        source_sha256="a" * 64, width=width, height=height, frame_count=56,
        aspect="custom", left=left, right=right, top=top, bottom=bottom,
        window_frames=56,
    )
    original = plan["sampling"]["source_lock_latent_box"]
    safe = _seam_safe_source_lock_box(plan)
    assert SEAM_CONTEXT_INSET_LATENTS == 2
    assert safe == original
    vt = (56 - 5) // 17 * 5 + 2
    at = round(56 / 24 * 40)
    shape = (1, 24, vt, plan["sampling"]["height"] // 16, plan["sampling"]["width"] // 16)
    captured = {}

    def backend(*args, **kwargs):
        captured["mask"] = kwargs["noise_mask"].unbind()[0].clone()
        video, audio = args[8].unbind()
        return comfy.nested_tensor.NestedTensor((torch.full_like(video, 42), audio))

    result, _, report = sample_outpaint_window(
        model=object(), conditioning=[[torch.zeros(1), {}]], plan=plan,
        shot_index=0, window_index=0,
        video_latent=torch.full(shape, 2.0),
        audio_latent=torch.full((1, 32, 2, at), 3.0),
        video_noise=torch.ones(shape), audio_noise=torch.ones((1, 32, 2, at)),
        sample_function=backend, source_edge_mode="strict",
    )
    video = result["samples"].unbind()[0]
    x0, y0, x1, y1 = safe
    assert torch.all(captured["mask"][:, :, :, y0:y1, x0:x1] == 0)
    assert torch.all(video[:, :, :, y0:y1, x0:x1] == 2)
    exterior = captured["mask"].expand_as(video).bool()
    assert torch.all(video[exterior] == 42)
    assert not report["source_edge_context_regenerated"]
    assert report["seam_context_policy"] == "full_observed_source_latent_lock_v3"
    assert report["final_source_pixels_restored_by_compositor"]


def test_no_fractional_noise_ramp_can_regenerate_the_source_edge():
    plan = build_outpaint_plan(
        source_sha256="a" * 64, width=96, height=96, frame_count=56,
        aspect="custom", left=0, right=0, top=32, bottom=32, window_frames=56,
    )
    weights = _source_noise_weights(plan, device="cpu", mode="strict")
    assert torch.count_nonzero(weights) == 0
    assert torch.equal(weights[:, 0], weights[:, -1])
    assert torch.count_nonzero(_source_noise_weights(_plan(), device="cpu")) == 0


def test_context_ramp_is_projected_back_before_result_and_next_window_context():
    plan = build_outpaint_plan(source_sha256='a'*64,width=96,height=96,frame_count=90,
                              aspect='custom',left=32,top=32,right=32,bottom=32,window_frames=73)
    x0,y0,x1,y1 = plan['sampling']['source_lock_latent_box']
    captures = []
    def backend(*args, **kwargs):
        captures.append(kwargs['noise_mask'].unbind()[0].clone())
        video,audio = args[8].unbind()
        return comfy.nested_tensor.NestedTensor((torch.full_like(video,42),audio))
    context = None
    for index,window in enumerate(plan['shots'][0]['windows']):
        vt=(window['render_frames']-5)//17*5+2
        at=round(window['render_frames']/24*40)
        video=torch.full((1,24,vt,10,10),2.)
        audio=torch.full((1,32,2,at),3.)
        result,context,report=sample_outpaint_window(model=object(),conditioning=[[torch.zeros(1),{}]],
            plan=plan,shot_index=0,window_index=index,video_latent=video,audio_latent=audio,
            video_noise=torch.ones_like(video),audio_noise=torch.ones_like(audio),
            context=context,sample_function=backend)
        actual=result['samples'].unbind()[0]
        assert torch.all(actual[:,:,:,y0:y1,x0:x1]==2)
        assert torch.all(actual[:,:,:,:y0]==42)
        assert report['source_edge_latents_projected_before_decode']
        if context is not None:
            assert torch.all(context['video_tail'][:,:,:,y0:y1,x0:x1]==2)
    assert torch.any((captures[0]>0)&(captures[0]<1))
    assert torch.all(captures[0][:,:,:,y0+2:y1-2,x0+2:x1-2]==0)


def test_next_window_uses_only_owned_previous_context_and_native_keyframes():
    plan = _plan()
    _, context, _ = sample_outpaint_window(**_inputs(plan), sample_function=_backend)
    inputs = _inputs(plan, index=1)
    captured = {}

    def backend(*args, **kwargs):
        captured["conditions"] = args[6]
        captured["masks"] = kwargs["noise_mask"]
        video, audio = args[8].unbind()
        return comfy.nested_tensor.NestedTensor((torch.full_like(video, 88), audio))

    result, tail, report = sample_outpaint_window(**inputs, context=context, sample_function=backend)
    overlap = plan["shots"][0]["windows"][1]["context_video_latents"]
    video, _ = result["samples"].unbind()
    assert torch.equal(video[:, :, :overlap], context["video_tail"])
    assert torch.all(video[:, :, overlap:, :2] == 88)
    assert captured["conditions"][0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    assert "minimax_keyframes" not in inputs["conditioning"][0][1]
    assert captured["conditions"][0][1]["other"] == "kept"
    assert torch.count_nonzero(captured["masks"].unbind()[1]) == 0
    assert tail is None and report["deliver_start"] == 73 and report["deliver_stop"] == 90


def test_cross_source_cross_shot_missing_or_changed_audio_context_is_rejected():
    plan = _plan()
    _, context, _ = sample_outpaint_window(**_inputs(plan), sample_function=_backend)
    inputs = _inputs(plan, index=1)
    for bad in (None, dict(context, plan_sha256="b" * 64), dict(context, shot_index=1)):
        with pytest.raises(ValueError, match="bound"):
            sample_outpaint_window(**inputs, context=bad, sample_function=_backend)
    context["audio_tail"].zero_()
    with pytest.raises(ValueError, match="audio context"):
        sample_outpaint_window(**inputs, context=context, sample_function=_backend)
    with pytest.raises(ValueError, match="new shot"):
        sample_outpaint_window(**_inputs(_plan((45,)), shot=1), context=context, sample_function=_backend)


def test_backend_shape_and_nonfinite_failures_are_explicit():
    inputs = _inputs(_plan())
    with pytest.raises(ValueError, match="joint AV"):
        sample_outpaint_window(**inputs, sample_function=lambda *a, **kw: torch.zeros(1))
    inputs["video_noise"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sample_outpaint_window(**inputs, sample_function=_backend)


def test_source_video_overlap_and_audio_dtype_must_match():
    plan = _plan()
    _, context, _ = sample_outpaint_window(**_inputs(plan), sample_function=_backend)
    inputs = _inputs(plan, index=1)
    inputs["video_latent"][:, :, :, 2:4, 2:4] += 0.1
    with pytest.raises(ValueError, match="source video context"):
        sample_outpaint_window(**inputs, context=context, sample_function=_backend)
    inputs = _inputs(plan, index=1)
    context["audio_tail"] = context["audio_tail"].half()
    with pytest.raises(ValueError, match="audio dtype"):
        sample_outpaint_window(**inputs, context=context, sample_function=_backend)


def test_unknown_audio_is_generated_as_internal_context_but_observed_audio_stays_locked():
    plan = _plan()
    first = _inputs(plan)
    mask = torch.ones((1, 1, 2, first["audio_latent"].shape[-1]))
    mask[..., :50] = 0
    result, context, report = sample_outpaint_window(**first, audio_noise_mask=mask, sample_function=_backend)
    audio = result["samples"].unbind()[1]
    assert torch.all(audio[..., :50] == 3) and torch.all(audio[..., 50:] == 99)
    assert not report["sampled_audio_discarded"]
    assert report["audio_output_policy"] == "original_file_track_only"
    second = _inputs(plan, index=1)
    window = plan["shots"][0]["windows"][1]
    next_mask = torch.ones((1, 1, 2, second["audio_latent"].shape[-1]))
    next_mask[..., :max(0, 50-window["audio_start"])] = 0
    result, _, _ = sample_outpaint_window(**second, context=context, audio_noise_mask=next_mask, sample_function=_backend)
    audio = result["samples"].unbind()[1]
    assert torch.equal(audio[..., :window["context_audio_latents"]], context["audio_tail"])


def test_fractional_or_invalid_audio_mask_is_rejected():
    inputs = _inputs(_plan())
    with pytest.raises(ValueError, match="binary"):
        sample_outpaint_window(**inputs, audio_noise_mask=torch.full((1, 1, 2, 122), 0.5), sample_function=_backend)


def test_backend_cannot_mutate_ownership_masks():
    def backend(*args, **kwargs):
        for mask in kwargs["noise_mask"].unbind():
            mask.fill_(1)
        return _backend(*args, **kwargs)

    result, _, _ = sample_outpaint_window(**_inputs(_plan()), sample_function=backend)
    video, audio = result["samples"].unbind()
    assert torch.all(audio == 3) and torch.all(video[:, :, :, 2:4, 2:4] == 2)
