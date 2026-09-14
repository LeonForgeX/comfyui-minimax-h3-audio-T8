from __future__ import annotations

import json
from pathlib import Path

from tools.package_progressive_candidate import SELF_LIFT_WORKFLOWS


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples" / "workflows" / "33-selflift-taomate"
PENDING_TOKENS = ("UNREVIEWED", "未人审", "尚待人审", "仍需CPU/UI复核", "未通过不晋级")


def _long_node(workflow: dict) -> dict:
    return next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3ProgressiveLongVideoEXPT8"
    )


def test_promoted_workflows_have_no_pending_review_markers():
    paths = sorted(DESTINATION.glob("*.json"))
    assert len(paths) == 9
    assert {path.relative_to(ROOT).as_posix() for path in paths} == set(SELF_LIFT_WORKFLOWS)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in PENDING_TOKENS:
            assert token not in text
        workflow = json.loads(text)
        review = workflow["extra"]["t8_bound_review"]
        assert review["status"] == "accepted_in_this_review_scope"
        assert review["not_universal_quality_claim"] is True


def test_both_eav_workflows_ship_the_reviewed_recipe_and_user_guidance():
    names = (
        "2026-09-14_H3_SelfLift_I2VA_EAV_4plus4_EXP.json",
        "2026-09-14_H3_SelfLift_I2VA_KJ_FFN_TST_EAV_Relay_4plus4_EXP.json",
    )
    for name in names:
        workflow = json.loads((DESTINATION / name).read_text(encoding="utf-8"))
        node = _long_node(workflow)
        named = node["widgets_values_named"]
        assert named["eav_mode"] == "apply_exp"
        assert named["eav_tau"] == 8.0
        assert named["eav_start_video_progress"] == 0.15
        assert named["eav_end_video_progress"] == 0.90
        assert named["eav_g_hard_limit"] == 1.5
        note = next(node for node in workflow["nodes"] if node["type"] in {"MarkdownNote", "Note"})
        text = note["widgets_values"][0]
        assert "想要更高动态时，用户可自行小幅逐步提高 eav_tau" in text
        assert "不要误调 TST 的 tau" in text
        assert "身份漂移、局部形变、闪烁" in text
