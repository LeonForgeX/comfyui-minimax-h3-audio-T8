"""Promote browser-saved H3 Video Outpaint drafts to released EXP workflows.

The promotion is intentionally narrow: workflow topology, node modes, widget
values, native subgraph definitions and IDs are preserved. Only the filename,
human-facing draft labels and delivery status change. Source and user workflow
copies are written from the same bytes so their hashes must match.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

try:
    from .frontend_workflow_compat import normalize_native_widget_inputs
except ImportError:
    from frontend_workflow_compat import normalize_native_widget_inputs


BASE_FILES = {
    "2026-09-06_H3_Video_Outpaint_Four_Stages_DRAFT.json":
        "2026-09-07_H3_Video_Outpaint_Four_Stages_EXP.json",
    "2026-09-06_H3_Video_Outpaint_Compact_DRAFT.json":
        "2026-09-07_H3_Video_Outpaint_Compact_EXP.json",
}
PREVIEW_FILES = {
    "2026-09-06_H3_Video_Outpaint_Geometry_Preview_DRAFT.json":
        "2026-09-07_H3_Video_Outpaint_Geometry_Preview_EXP.json",
}
CANDIDATE_FILES = {
    f"H3_Outpaint_{index:02d}_{route}_DRAFT.json":
        f"2026-09-07_H3_Video_Outpaint_{index:02d}_{route}_EXP.json"
    for index, route in enumerate(
        (
            "Generate_Candidate",
            "Review_Candidate",
            "Confirm_Candidate",
            "Continue_Selection",
            "Save_Completed_Selection",
            "Generate_Guided_Candidate",
            "Continue_Guided_Selection",
            "Save_Completed_Selection_DLSS_2x",
            "Model_Compatibility_Audit",
        ),
        start=1,
    )
}


def _replace_labels(value):
    if isinstance(value, dict):
        return {key: _replace_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_labels(item) for item in value]
    if not isinstance(value, str):
        return value
    value = value.replace(" · Draft", " · EXP").replace("(Draft)", "(EXP)")
    value = value.replace(
        "Video Outpaint · 独立草稿 / Independent draft",
        "Video Outpaint · 正式 EXP 工作流 / Released EXP workflow",
    )
    value = value.replace(
        "这是未发布草稿，仅隔离测试环境已注册；正式安装、预览和完整人审还需完成。",
        "这是正式 EXP 工作流；生成前先预览范围，成片仍需人工检查扩画边界、动作和声音。",
    )
    value = value.replace(
        "这是隔离测试草稿，尚未发布。请勿把本图当成生成首帧的选择或验收。",
        "这是正式 EXP 范围预览，不会生成视频。请勿把本图当成生成候选或成片验收。",
    )
    value = value.replace(
        "隔离测试草稿，尚未正式安装/发布。",
        "正式 EXP 工作流。扩画会改变画面，请先审候选，再明确确认并接续。",
    )
    return value


def execution_contract(workflow: dict) -> dict:
    """Return the fields promotion is forbidden to change."""
    return {
        "id": workflow.get("id"),
        "last_node_id": workflow.get("last_node_id"),
        "last_link_id": workflow.get("last_link_id"),
        "links": deepcopy(workflow.get("links", [])),
        "nodes": [
            {
                key: deepcopy(node.get(key))
                for key in (
                    "id",
                    "type",
                    "mode",
                    "inputs",
                    "outputs",
                    "widgets_values",
                    "widgets_values_named",
                )
                if key in node
            }
            for node in workflow.get("nodes", [])
            if node.get("type") != "MarkdownNote"
        ],
        "definitions": deepcopy(workflow.get("definitions")),
    }


def promote_workflow(workflow: dict, *, preview: bool = False) -> dict:
    before = execution_contract(workflow)
    result = _replace_labels(deepcopy(workflow))
    extra = result.setdefault("extra", {})
    extra["outpaint_delivery_status"] = (
        "released_exp_geometry_preview_no_generation"
        if preview
        else "released_exp_explicit_human_review_required"
    )
    if execution_contract(result) != before:
        raise ValueError("promotion changed the executable workflow contract")
    encoded = json.dumps(result, ensure_ascii=False)
    for stale in (" · Draft", "(Draft)", "未发布草稿", "隔离测试草稿"):
        if stale in encoded:
            raise ValueError(f"stale draft label remains: {stale}")
    return result


def normalize_release_workflow(workflow: dict) -> dict:
    """Apply the project's strict-importer format without changing semantics."""
    result = deepcopy(workflow)
    expected = deepcopy(result)
    normalize_native_widget_inputs(expected)
    report = normalize_native_widget_inputs(result)
    if execution_contract(result) != execution_contract(expected):
        raise ValueError("widget normalization changed the executable workflow contract")
    return {"workflow": result, "report": report}


def _write_pair(source_path: Path, destination: Path, mirror: Path, *, preview: bool) -> dict:
    workflow = json.loads(source_path.read_text(encoding="utf-8"))
    promoted = promote_workflow(workflow, preview=preview)
    normalized = normalize_release_workflow(promoted)
    promoted = normalized["workflow"]
    payload = (json.dumps(promoted, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    destination.write_bytes(payload)
    mirror.write_bytes(payload)
    source_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    mirror_sha = hashlib.sha256(mirror.read_bytes()).hexdigest()
    if source_sha != mirror_sha:
        raise ValueError(f"workflow mirror mismatch: {destination.name}")
    return {
        "name": destination.name,
        "nodes": len(promoted.get("nodes", [])),
        "links": len(promoted.get("links", [])),
        "normalized_nodes": len(normalized["report"]["normalized"]),
        "repaired_link_slots": len(normalized["report"]["repaired_links"]),
        "sha256": source_sha,
    }


def promote_all(
    *,
    base_dir: Path,
    preview_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
    user_dir: Path,
) -> list[dict]:
    if output_dir.exists() or user_dir.exists():
        raise FileExistsError("promotion destinations must be new directories")
    output_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)
    reports = []
    groups = (
        (base_dir, BASE_FILES, False),
        (preview_dir, PREVIEW_FILES, True),
        (candidate_dir, CANDIDATE_FILES, False),
    )
    for source_dir, mapping, preview in groups:
        for source_name, destination_name in mapping.items():
            reports.append(
                _write_pair(
                    source_dir / source_name,
                    output_dir / destination_name,
                    user_dir / destination_name,
                    preview=preview,
                )
            )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--user-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reports = promote_all(
        base_dir=args.base_dir,
        preview_dir=args.preview_dir,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        user_dir=args.user_dir,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "status": "promoted_from_browser_saved_workflows",
                "workflow_count": len(reports),
                "source_user_hashes_equal": True,
                "workflows": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()
