import copy
import json

import numpy as np
import pytest

from h3_audio_t8_pkg import meridian_plan as p


@pytest.fixture
def material():
    return dict(
        identity="geometry-content-id",
        start=12,
        end=100,
        kind="video",
        pivot=[0.0, 0.0, 1.0],
    )


@pytest.mark.parametrize("preset", p.PRESETS)
def test_presets_canonical_and_roundtrip(material, preset):
    plan = p.preset_plan(material, preset=preset)
    assert p.canonical_plan(p.plan_json(plan), material) == plan
    assert plan["time_keys"] == [
        dict(t=0, src=12),
        dict(t=72, src=12 if preset == "freeze_orbit" else 84),
    ]
    assert p.identity(plan) == p.identity(json.loads(p.plan_json(plan)))


@pytest.mark.parametrize(
    "field,value",
    [("frames", True), ("roll", False), ("origin_frame", 12.0), ("window_end", True)],
)
def test_noninteger_fields_rejected(material, field, value):
    plan = p.preset_plan(material)
    plan[field] = value
    with pytest.raises(ValueError):
        p.canonical_plan(plan, material)


@pytest.mark.parametrize(
    "change",
    [
        "geometry",
        "key_missing_t",
        "key_not_object",
        "reverse",
        "out_of_window",
        "focal",
        "coincident",
        "duplicate_t",
        "nan",
    ],
)
def test_invalid_plans(material, change):
    plan = p.preset_plan(material)
    if change == "geometry":
        plan["geometry_id"] = "old"
    elif change == "key_missing_t":
        del plan["camera_keys"][0]["t"]
    elif change == "key_not_object":
        plan["time_keys"][0] = 3
    elif change == "reverse":
        plan["time_keys"] = [dict(t=0, src=99), dict(t=72, src=12)]
    elif change == "out_of_window":
        plan["time_keys"][1]["src"] = 101
    elif change == "focal":
        plan["camera_keys"][0]["focal"] = 0
    elif change == "coincident":
        plan["camera_keys"][0]["look"] = [0.0, 0.0, 0.0]
    elif change == "duplicate_t":
        plan["camera_keys"][1]["t"] = 0
    elif change == "nan":
        plan["camera_keys"][0]["pos"][0] = float("nan")
    with pytest.raises(ValueError):
        p.canonical_plan(plan, material)


def test_duplicate_json_key_rejected(material):
    raw = p.plan_json(p.preset_plan(material)).replace(
        '"frames":73', '"frames":73,"frames":73'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        p.canonical_plan(raw, material)


def test_backend_rounding_ties_even_and_audio(material):
    plan = p.preset_plan(material, frames=5)
    plan["time_keys"] = [dict(t=0, src=12), dict(t=4, src=14)]
    assert p.source_map(plan) == [12, 12, 13, 14, 14]
    with pytest.raises(ValueError, match="1:1"):
        p.audio_policy(plan, "source_1to1")
    assert p.audio_policy(plan, "silent")["duration"] == 5 / 24
    plan["time_keys"][1]["src"] = 16
    assert p.audio_policy(plan, "source_1to1")["one_to_one"]


def test_still_and_freeze_not_original_sound(material):
    still = dict(material, kind="image", start=0, end=0)
    plan = p.preset_plan(still)
    assert p.source_map(plan) == [0] * 73
    with pytest.raises(ValueError):
        p.audio_policy(plan, "source_1to1")


@pytest.mark.parametrize(
    "shape", [(1024, 1536), (3027, 1531), (1920, 1080), (1080, 1920)]
)
def test_aspect_bucket_not_stretch(shape):
    canvas, condition = p.bucket(*shape)
    assert canvas in p.TARGETS and condition in p.CONDITIONS
    assert all(n % 16 == 0 for n in (*canvas, *condition))


def test_fov_roundtrip_and_bad_dimensions():
    degrees = p.horizontal_fov(450, 512, 1.3)
    assert p.fov_multiplier(degrees, 450, 512) == pytest.approx(1.3)
    with pytest.raises(ValueError):
        p.fov_multiplier(40, 0, 512)


def test_tracks_do_not_mutate_caller_or_couple_source_breakpoints(material):
    plan = p.preset_plan(material)
    before = copy.deepcopy(plan)
    result = p.canonical_plan(plan, material)
    result["time_keys"].insert(1, dict(t=20, src=20))
    assert plan == before
    assert result["camera_keys"] == before["camera_keys"]
    assert not np.array_equal(p.source_map(result), p.source_map(before))


@pytest.mark.parametrize("strength", [-0.12, 0.0, 0.12])
def test_slide_translates_target_with_camera_without_compensating_pan(material, strength):
    material = dict(material, pivot=[0.2, -0.1, 1.0])
    plan = p.preset_plan(material, preset="slide", strength=strength)
    keys = plan["camera_keys"]
    assert keys[-1]["pos"] == [strength, 0.0, 0.0]
    assert keys[-1]["look"] == [0.2 + strength, -0.1, 1.0]
    assert np.allclose(
        np.asarray(keys[0]["look"]) - keys[0]["pos"],
        np.asarray(keys[-1]["look"]) - keys[-1]["pos"],
    )
    assert plan["time_keys"] == [dict(t=0, src=12), dict(t=72, src=84)]
    assert material["pivot"] == [0.2, -0.1, 1.0]


def test_fixing_slide_does_not_rewrite_authored_or_orbit_plan(material):
    slide = p.preset_plan(material, preset="slide", strength=0.03)
    slide["camera_keys"][-1]["look"] = material["pivot"].copy()
    assert p.canonical_plan(slide, material) == slide
    for preset in ("orbit", "freeze_orbit", "source_camera"):
        plan = p.preset_plan(material, preset=preset, strength=0.03)
        assert all(key["look"] == material["pivot"] for key in plan["camera_keys"])
