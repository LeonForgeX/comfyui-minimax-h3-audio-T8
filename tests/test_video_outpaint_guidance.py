from __future__ import annotations

import copy
import json

import pytest

from h3_audio_t8_pkg.video_outpaint_guidance import (
    build_outpaint_guidance,
    validate_outpaint_guidance,
)
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan


def _plan(**kwargs):
    return build_outpaint_plan(**({
        "source_sha256": "a" * 64, "width": 320, "height": 240,
        "frame_count": 80, "aspect": "custom", "left": 64,
        "top": 64, "right": 64, "bottom": 64,
        "generation_megapixels": 0.5, "cut_frames": (40,),
    } | kwargs))


def test_guidance_compiles_per_shot_spatial_rows_and_people():
    plan = _plan()
    regions = [
        {"shot": "all", "region": "top", "prompt": "clouds"},
        {"shot": 1, "region": "bottom", "prompt": "floor"},
    ]
    people = [
        {"shot": 0, "label": "lead", "box": [0, 20, 80, 120]},
        {"shot": "all", "label": "support", "box": [100, 40, 60, 100]},
    ]
    guidance = build_outpaint_guidance(plan, json.dumps(regions), json.dumps(people))
    assert [len(shot["regions"]) for shot in guidance["shots"]] == [1, 2]
    assert [len(shot["people"]) for shot in guidance["shots"]] == [2, 1]
    assert guidance["shots"][0]["people"][0]["touches_source_boundary"] == ["left"]
    assert guidance["shots"][0]["people"][0]["source_pixels_locked"] is True
    assert guidance["policies"]["expanded_area_person_pixels_guaranteed"] is False
    assert validate_outpaint_guidance(guidance, plan) == guidance


def test_expanded_region_is_only_outside_source_token_centres():
    plan = _plan()
    guidance = build_outpaint_guidance(
        plan, [{"shot": "all", "region": "expanded", "prompt": "continue scene"}], [])
    rows = guidance["shots"][0]["regions"][0]["spatial_rows"]
    columns = guidance["sampling_grid"]["columns"]
    source = guidance["sampling_grid"]["source_rect"]
    assert rows
    for row in rows:
        x, y = row % columns, row // columns
        px, py = x * 32 + 16, y * 32 + 16
        assert not (source[0] <= px < source[2] and source[1] <= py < source[3])


def test_custom_bbox_cannot_claim_source_owned_pixels():
    plan = _plan()
    with pytest.raises(ValueError, match="overlaps"):
        build_outpaint_guidance(
            plan, [{"shot": 0, "region": "bbox", "box": [64, 64, 40, 40], "prompt": "bad"}], [])


def test_guidance_tampering_is_rejected_even_when_shape_is_valid():
    plan = _plan()
    guidance = build_outpaint_guidance(
        plan, [{"shot": "all", "region": "left", "prompt": "wall"}], [])
    changed = copy.deepcopy(guidance)
    changed["shots"][0]["regions"][0]["spatial_rows"].append(999)
    with pytest.raises(ValueError, match="integrity"):
        validate_outpaint_guidance(changed, plan)


@pytest.mark.parametrize("value", [
    [{"shot": 9, "region": "top", "prompt": "x"}],
    [{"shot": 0, "region": "unknown", "prompt": "x"}],
    [{"shot": 0, "region": "top", "prompt": ""}],
    [{"shot": 0, "region": "top", "prompt": "x", "other": 1}],
])
def test_invalid_region_requests_fail_closed(value):
    with pytest.raises(ValueError):
        build_outpaint_guidance(_plan(), value, [])
