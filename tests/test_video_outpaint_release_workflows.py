from __future__ import annotations

import asyncio
import json
from pathlib import Path

import h3_audio_t8_pkg


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows" / "27-video-outpaint"


def _workflows() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIR.glob("*.json"))
    }


def test_release_contains_twelve_exp_workflows_without_changing_v174_baseline():
    workflows = _workflows()
    assert len(workflows) == 12
    assert all(name.startswith("2026-09-07_H3_Video_Outpaint_") for name in workflows)
    assert all(name.endswith("_EXP.json") for name in workflows)
    assert sum(len(workflow["nodes"]) for workflow in workflows.values()) == 107
    assert sum(
        node["type"] != "MarkdownNote"
        for workflow in workflows.values()
        for node in workflow["nodes"]
    ) == 95
    assert sum(len(workflow["links"]) for workflow in workflows.values()) == 89
    encoded = json.dumps(workflows, ensure_ascii=False)
    assert " · Draft" not in encoded
    assert "(Draft)" not in encoded
    assert "隔离测试草稿" not in encoded
    assert "尚未正式安装/发布" not in encoded


def test_every_outpaint_node_in_release_workflows_is_registered_and_enabled():
    schemas = [
        node.define_schema()
        for node in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ]
    registered = {schema.node_id for schema in schemas}
    used = {
        node["type"]
        for workflow in _workflows().values()
        for node in workflow["nodes"]
        if node["type"].startswith("MiniMaxH3VideoOutpaint")
    }
    assert used <= registered
    assert len(used) == 17
    assert all(
        node.get("mode", 0) == 0
        for workflow in _workflows().values()
        for node in workflow["nodes"]
    )


def test_candidate_confirmation_remains_opt_in_and_dlss_has_no_removed_license_widget():
    workflows = _workflows()
    confirm = next(value for name, value in workflows.items() if "03_Confirm" in name)
    select = next(
        node
        for node in confirm["nodes"]
        if node["type"] == "MiniMaxH3VideoOutpaintSelectCandidateT8"
    )
    assert select["widgets_values_named"]["confirm_selection"] is False

    dlss = next(value for name, value in workflows.items() if "08_Save_Completed" in name)
    audit = next(
        node
        for node in dlss["nodes"]
        if node["type"] == "MiniMaxH3DLSSNRRuntimeAuditT8Advanced"
    )
    assert "accept_external_runtime_license" not in audit["widgets_values_named"]
    assert audit["widgets_values"] == ["1.3", "feature_probe_1_frame", 0, 0]


def test_generation_workflows_keep_fixed_seed_and_default_color_match():
    workflows = _workflows()
    candidates = [
        node
        for workflow in workflows.values()
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3VideoOutpaintCandidateT8"
    ]
    assert len(candidates) == 2
    for node in candidates:
        assert node["widgets_values_named"]["control_after_generate"] == "fixed"
        assert node["widgets_values_named"]["steps"] == 20
        assert node["widgets_values_named"]["resume"] is False
        assert node["widgets_values_named"]["color_match"] is True

    four_stage = next(value for name, value in workflows.items() if "Four_Stages" in name)
    compose = next(
        node
        for node in four_stage["nodes"]
        if node["type"] == "MiniMaxH3VideoOutpaintComposeT8"
    )
    assert compose["widgets_values"] == ["outpaint", True, False, "joint_decode"]
    assert compose["widgets_values_named"]["source_mode"] == "joint_decode"
    assert all(node["widgets_values_named"]["source_mode"] == "joint_decode" for node in candidates)
    assert not any(
        node["type"] == "SaveVideo"
        for workflow in workflows.values()
        for node in workflow["nodes"]
    )
