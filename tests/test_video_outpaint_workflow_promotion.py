from copy import deepcopy

from tools.promote_video_outpaint_workflows import (
    BASE_FILES,
    CANDIDATE_FILES,
    PREVIEW_FILES,
    execution_contract,
    normalize_release_workflow,
    promote_workflow,
)


def _workflow(note: str = "隔离测试草稿，尚未正式安装/发布。") -> dict:
    return {
        "id": "fixed-id",
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 1,
                "type": "MiniMaxH3VideoOutpaintPlanT8",
                "mode": 0,
                "inputs": [{"name": "aspect", "link": None, "widget": {"name": "aspect"}}],
                "outputs": [{"name": "plan", "type": "T8_H3_OUTPAINT_PLAN", "links": []}],
                "widgets_values": ["custom"],
                "widgets_values_named": {"aspect": "custom"},
                "title": "H3 Video Outpaint · Plan (EXP)",
            },
            {
                "id": 2,
                "type": "MarkdownNote",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [note],
                "title": "Read first · Draft",
            },
        ],
        "links": [],
        "extra": {
            "workflow_title": "H3 Video Outpaint · Draft",
            "outpaint_delivery_status": "draft_not_registered_not_human_accepted",
        },
    }


def test_promotion_changes_only_release_labels_and_status():
    source = _workflow()
    before = deepcopy(source)
    promoted = promote_workflow(source)
    assert source == before
    assert execution_contract(promoted) == execution_contract(source)
    assert promoted["extra"]["workflow_title"].endswith(" · EXP")
    assert promoted["extra"]["outpaint_delivery_status"] == (
        "released_exp_explicit_human_review_required"
    )
    note = promoted["nodes"][1]["widgets_values"][0]
    assert "正式 EXP 工作流" in note and "草稿" not in note


def test_preview_has_non_generation_release_status():
    promoted = promote_workflow(
        _workflow("这是隔离测试草稿，尚未发布。请勿把本图当成生成首帧的选择或验收。"),
        preview=True,
    )
    assert promoted["extra"]["outpaint_delivery_status"] == (
        "released_exp_geometry_preview_no_generation"
    )
    assert "不会生成视频" in promoted["nodes"][1]["widgets_values"][0]


def test_release_map_contains_exactly_twelve_distinct_workflows():
    destinations = [*BASE_FILES.values(), *PREVIEW_FILES.values(), *CANDIDATE_FILES.values()]
    assert len(destinations) == 12
    assert len(set(destinations)) == 12
    assert all(name.startswith("2026-09-07_H3_Video_Outpaint_") for name in destinations)
    assert all(name.endswith("_EXP.json") for name in destinations)


def test_release_normalization_removes_only_unlinked_widget_inputs():
    source = _workflow()
    source["nodes"][0]["inputs"].extend(
        [
            {"name": "plan", "type": "T8_PLAN", "link": 1},
            {"name": "top", "link": None, "widget": {"name": "top"}},
        ]
    )
    source["nodes"][0]["widgets_values"] = ["custom", 96]
    source["nodes"].insert(
        0,
        {
            "id": 3,
            "type": "Source",
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "plan", "type": "T8_PLAN", "links": [1]}],
            "widgets_values": [],
        },
    )
    source["links"] = [[1, 3, 0, 1, 1, "T8_PLAN"]]
    normalized = normalize_release_workflow(source)["workflow"]
    target = next(node for node in normalized["nodes"] if node["id"] == 1)
    assert [item["name"] for item in target["inputs"]] == ["plan"]
    assert normalized["links"] == [[1, 3, 0, 1, 0, "T8_PLAN"]]
    assert target["widgets_values"] == ["custom", 96]
