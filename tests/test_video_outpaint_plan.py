from __future__ import annotations

import hashlib

import pytest

from h3_audio_t8_pkg.video_outpaint_plan import (
    ASPECTS, WINDOW_FRAMES, build_outpaint_plan, canonical, validate_outpaint_plan,
)


def _plan(**kwargs):
    return build_outpaint_plan(**({"source_sha256": "a" * 64, "width": 736, "height": 416,
                                 "frame_count": 768} | kwargs))


@pytest.mark.parametrize("aspect", ASPECTS)
def test_all_aspects_preserve_entire_unscaled_source(aspect):
    plan = _plan(aspect=aspect)
    out = plan["output"]
    x0, y0, x1, y1 = out["source_rect"]
    assert x1 - x0 == 736 and y1 - y0 == 416
    assert 0 <= x0 < x1 <= out["width"]
    assert 0 <= y0 < y1 <= out["height"]
    assert out["frames"] == 768
    assert not out["source_cropped"] and not out["source_resized"]
    if aspect not in {"source", "custom"}:
        a, b = map(int, aspect.split(":"))
        assert out["width"] * b == out["height"] * a
    assert validate_outpaint_plan(plan) == plan


def test_custom_margins_and_anchor_are_not_silently_quantized():
    plan = _plan(width=739, height=419, aspect="custom", left=13, top=27, right=31, bottom=49)
    assert plan["output"]["margins"] == [13, 27, 31, 49]
    assert plan["output"]["source_rect"] == [13, 27, 752, 446]
    assert plan["output"]["width"] == 783
    assert _plan(anchor_y=0)["output"]["source_rect"][1] == 0
    bottom = _plan(anchor_y=1)["output"]
    assert bottom["source_rect"][3] == bottom["height"]


def test_odd_exact_delivery_uses_padding_slack_for_latent_aligned_source():
    plan = _plan(frame_count=56, aspect="custom", top=95, bottom=96, window_frames=56)
    assert plan["output"]["height"] == 607
    assert plan["output"]["source_rect"] == [0, 95, 736, 511]
    assert plan["sampling"]["height"] == 608
    assert plan["sampling"]["offset"] == [0.0, 1.0]
    assert plan["sampling"]["source_rect"] == [0.0, 96.0, 736.0, 512.0]
    assert plan["sampling"]["source_lock_latent_box"] == [0, 6, 46, 32]


@pytest.mark.parametrize("budget", [0, 0.001024, 0.02, 0.24, 0.5, 1.0, 4.0])
def test_budget_is_actual_model_area_and_transform_is_isotropic(budget):
    plan = _plan(width=1920, height=1080, generation_megapixels=budget)
    model = plan["sampling"]
    assert model["width"] % 32 == model["height"] % 32 == 0
    assert model["pixels"] == model["width"] * model["height"] <= model["budget_pixels"]
    if budget:
        assert model["pixels"] <= int(budget * 1_000_000)
    x0, y0, x1, y1 = model["source_rect"]
    assert (x1 - x0) / 1920 == pytest.approx((y1 - y0) / 1080)
    lx0, ly0, lx1, ly1 = model["source_lock_latent_box"]
    assert lx0 * 16 <= x0 <= x1 <= lx1 * 16
    assert ly0 * 16 <= y0 <= y1 <= ly1 * 16


@pytest.mark.parametrize("frames", [1, 5, 6, 22, 23, 89, 124, 634, 768, 3000])
@pytest.mark.parametrize("window", WINDOW_FRAMES)
def test_windows_have_no_gaps_duplicate_delivery_or_padding_leak(frames, window):
    plan = _plan(frame_count=frames, window_frames=window)
    shot = plan["shots"][0]
    delivered = []
    for i, item in enumerate(shot["windows"]):
        assert item["source_start"] % 17 == 0
        assert item["render_frames"] % 17 == 5
        assert item["source_frames"] <= item["render_frames"]
        assert item["context_video_latents"] == 0 if i == 0 else item["context_video_latents"] > 0
        delivered.extend(range(item["deliver_start"], item["deliver_stop"]))
    assert delivered == list(range(frames))
    assert plan["policies"]["padding_delivered"] is False
    assert plan["policies"]["vram_safe"] is None


def test_shot_cuts_reset_context_and_do_not_sample_next_shot():
    plan = _plan(cut_frames=(23, 300, 767))
    delivered = []
    for shot in plan["shots"]:
        first = shot["windows"][0]
        assert first["context_video_latents"] == first["context_audio_latents"] == 0
        for item in shot["windows"]:
            assert item["source_start"] + item["source_frames"] <= shot["stop"]
            delivered.extend(range(item["deliver_start"], item["deliver_stop"]))
    assert delivered == list(range(768))


@pytest.mark.parametrize("kwargs", [
    {"width": True}, {"height": 0}, {"frame_count": 0}, {"source_fps": "30"},
    {"source_fps": "0/0"}, {"generation_megapixels": float("nan")},
    {"generation_megapixels": 0.0001}, {"anchor_x": 2}, {"left": 4},
    {"aspect": "other"}, {"source_sha256": "x" * 64}, {"window_frames": 40},
    {"cut_frames": (300, 23)}, {"cut_frames": (23, 23)}, {"cut_frames": (768,)},
])
def test_invalid_requests_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        _plan(**kwargs)


def test_rehashed_geometry_tampering_is_not_accepted():
    plan = _plan()
    plan["output"]["source_rect"][0] = -32
    with pytest.raises(ValueError, match="integrity"):
        validate_outpaint_plan(plan)
    plan.pop("plan_sha256")
    plan["plan_sha256"] = hashlib.sha256(canonical(plan).encode()).hexdigest()
    with pytest.raises(ValueError, match="coordinates"):
        validate_outpaint_plan(plan)
