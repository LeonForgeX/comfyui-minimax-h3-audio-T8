from __future__ import annotations

import copy

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_color import color_match_outpaint_frames
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan


def _data(cuts=(), dtype=torch.float32, channels=4):
    plan = build_outpaint_plan(source_sha256="a" * 64, width=8, height=8, frame_count=6,
                              aspect="custom", left=8, right=8, top=8, bottom=8, cut_frames=cuts)
    source = torch.full((6, 8, 8, channels), 0.6)
    candidate = torch.full((6, 24, 24, channels), 0.5)
    source[..., 3:] = 0.8
    candidate[..., 3:] = 0.3
    if dtype == torch.uint8:
        return (source * 255).round().to(dtype), (candidate * 255).round().to(dtype), plan
    return source.to(dtype), candidate.to(dtype), plan


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.uint8])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_only_expanded_rgb_near_boundary_changes(dtype, channels):
    source, candidate, plan = _data(dtype=dtype, channels=channels)
    before = candidate.clone()
    result, state, report = color_match_outpaint_frames(source, candidate, plan, fade_pixels=3)
    assert torch.equal(result[:, 8:16, 8:16], source)
    assert torch.equal(result[:, :5, :5], candidate[:, :5, :5])
    assert torch.all(result[:, 8:16, 7, :3] > candidate[:, 8:16, 7, :3])
    assert torch.equal(result[:, :8, :, 3:], candidate[:, :8, :, 3:])
    assert torch.equal(candidate, before)
    assert state["next_frame"] == 6
    assert report["source_exact_before_encoding"] and report["color_match_applied"]
    assert not report["perceptual_acceptance"]


def test_stream_chunking_is_bit_exact_and_cuts_reset_tint():
    source, candidate, plan = _data(cuts=(3,))
    candidate[3:, ..., :3] = 0.7
    whole, final, _ = color_match_outpaint_frames(source, candidate, plan)
    state = None
    pieces = []
    for start, stop in ((0, 2), (2, 4), (4, 6)):
        frames, state, _ = color_match_outpaint_frames(source[start:stop], candidate[start:stop], plan,
                                                       start_frame=start, state=state)
        pieces.append(frames)
    assert torch.equal(whole, torch.cat(pieces))
    assert state == final
    isolated, _, report = color_match_outpaint_frames(source[3:], candidate[3:], plan, start_frame=3)
    assert torch.equal(isolated, whole[3:])
    assert report["shot_resets"] == 1


@pytest.mark.parametrize("settings", [{"enabled": False}, {"strength": 0}])
def test_disabled_preserves_candidate_outside_source(settings):
    source, candidate, plan = _data()
    result, _, report = color_match_outpaint_frames(source, candidate, plan, **settings)
    assert torch.equal(result[:, :8], candidate[:, :8])
    assert torch.equal(result[:, 8:16, 8:16], source)
    assert not report["color_match_applied"]


def test_state_rejects_gaps_tamper_other_sources_and_settings():
    source, candidate, plan = _data()
    _, state, _ = color_match_outpaint_frames(source[:2], candidate[:2], plan)
    with pytest.raises(ValueError, match="requires state"):
        color_match_outpaint_frames(source[2:3], candidate[2:3], plan, start_frame=2)
    for start, extra in ((3, {}), (2, {"strength": 0.5})):
        with pytest.raises(ValueError, match="mismatch"):
            color_match_outpaint_frames(source[start:start+1], candidate[start:start+1], plan,
                                        start_frame=start, state=state, **extra)
    changed = copy.deepcopy(state)
    changed["offsets"][0][0] += 0.01
    with pytest.raises(ValueError, match="mismatch"):
        color_match_outpaint_frames(source[2:3], candidate[2:3], plan, start_frame=2, state=changed)
    different_plan = build_outpaint_plan(**dict(plan["request"], source_sha256="b" * 64))
    with pytest.raises(ValueError, match="mismatch"):
        color_match_outpaint_frames(source[2:3], candidate[2:3], different_plan, start_frame=2, state=state)


@pytest.mark.parametrize("settings", [{"enabled": 1}, {"strip_pixels": 0}, {"fade_pixels": 0},
                                     {"strength": float("nan")}, {"temporal_alpha": 0}])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        color_match_outpaint_frames(*_data(), **settings)


def test_non_normalized_float_rgb_rejected():
    source, candidate, plan = _data()
    candidate[0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="normalized"):
        color_match_outpaint_frames(source, candidate, plan)
