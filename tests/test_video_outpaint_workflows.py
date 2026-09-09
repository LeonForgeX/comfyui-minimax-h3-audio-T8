from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.build_video_outpaint_workflows import build_workflow, build_compact, execution_signature, STAGES
from tools.build_video_outpaint_workflows import build_preview_workflow, PREVIEW_NODE, PREVIEW_TYPES


def fixture():
    from h3_audio_t8_pkg.nodes_video_outpaint import VIDEO_OUTPAINT_DRAFT_NODE_CLASSES
    data = json.loads((Path(__file__).parent / "fixtures/outpaint_frontend_contract.json").read_text(encoding="utf-8"))
    # Keep the historical capture unchanged, but build new workflows against
    # the actual current native schemas, including optional source-mode widgets.
    for cls in VIDEO_OUTPAINT_DRAFT_NODE_CLASSES:
        data["object_info"][cls.define_schema().node_id] = json.loads(json.dumps(cls.GET_NODE_INFO_V1()))
    return data


def test_independent_frontend_is_deterministic_and_preserves_native_noise_color_audio():
    data = fixture()
    original = deepcopy(data)
    first = build_workflow(data["prompt"], data["object_info"])
    assert first == build_workflow(data["prompt"], data["object_info"])
    assert data == original
    assert first["version"] == 0.4
    nodes = {node["type"]: node for node in first["nodes"]}
    assert nodes[STAGES[2]]["widgets_values"] == [20260808, "fixed", 20, False, "t8.outpaint.native_cpu_noise/v1"]
    assert nodes[STAGES[3]]["widgets_values"] == ["outpaint", True, False, "joint_decode"]
    assert {i["name"] for i in nodes[STAGES[1]]["inputs"]} == {"plan", "clip", "video_vae", "audio_vae"}
    assert "SaveVideo" not in nodes and "MarkdownNote" in nodes
    assert not any("widget" in i for node in first["nodes"] for i in node["inputs"])
    assert execution_signature(first)


def test_compact_expand_has_identical_nodes_widgets_and_named_edges():
    data = fixture()
    source = build_workflow(data["prompt"], data["object_info"])
    original = deepcopy(source)
    compact = build_compact(source, data["object_info"])
    assert source == original and compact == build_compact(source, data["object_info"])
    assert execution_signature(source) == execution_signature(compact)
    definition = compact["definitions"]["subgraphs"][0]
    assert {n["type"] for n in definition["nodes"]} == set(STAGES)
    names = [i["name"] for i in definition["inputs"]]
    assert names.count("video_vae") == 1  # One external VAE pin still feeds Prepare and Compose.
    video_vae = next(i for i in definition["inputs"] if i["name"] == "video_vae")
    assert len(video_vae["linkIds"]) == 2
    assert {"resume_audio", "resume", "source_mode", "color_match", "prompt", "generation_megapixels"} <= set(names)
    assert "cut_frames_json" not in names  # Advanced controls remain inside, not removed.
    assert any(i["name"] == "cut_frames_json" for n in definition["nodes"] for i in n["inputs"])
    assert {o["type"] for o in definition["outputs"]} == {"VIDEO", "STRING"}


@pytest.mark.parametrize("damage", ["inner_target", "outer_origin", "proxy", "duplicate"])
def test_compact_audit_rejects_corrupt_connections(damage):
    data = fixture()
    compact = build_compact(build_workflow(data["prompt"], data["object_info"]), data["object_info"])
    definition = compact["definitions"]["subgraphs"][0]
    if damage == "inner_target":
        link = next(item for item in definition["links"] if item["target_id"] > 0)
        node = next(n for n in definition["nodes"] if n["id"] == link["target_id"])
        node["inputs"][link["target_slot"]]["link"] = None
    elif damage == "outer_origin":
        origin = next(n for n in compact["nodes"] if n["id"] == compact["links"][0][1])
        origin["outputs"][compact["links"][0][2]]["links"] = []
    elif damage == "proxy":
        top = next(n for n in compact["nodes"] if n["type"] == definition["id"])
        top["properties"]["proxyWidgets"][0][1] = "missing_control"
    else:
        definition["links"].append(deepcopy(definition["links"][0]))
    with pytest.raises(ValueError):
        execution_signature(compact)


def test_wrong_graph_and_extra_save_sink_are_not_silently_packaged():
    data = fixture()
    graph = data["prompt"]
    stage_id = next(key for key, node in graph.items() if node["class_type"] == STAGES[0])
    graph.pop(stage_id)
    with pytest.raises(ValueError, match="exactly one"):
        build_workflow(graph, data["object_info"])
    graph = fixture()["prompt"]
    graph["99"] = {"class_type": "SaveVideo", "inputs": {}}
    with pytest.raises(ValueError, match="already saves"):
        build_workflow(graph, data["object_info"])


def _preview_info():
    from h3_audio_t8_pkg.nodes_video_outpaint_preview import MiniMaxH3VideoOutpaintGeometryPreviewT8
    info = fixture()["object_info"]
    # Actual V3 schema serialization, not a hand-written approximation of the
    # new preview node's input order. Existing fixture schemas came from a server.
    info[PREVIEW_NODE] = json.loads(json.dumps(MiniMaxH3VideoOutpaintGeometryPreviewT8.GET_NODE_INFO_V1()))
    return info


def test_preview_workflow_has_only_source_plan_and_output_without_model_loading():
    info = _preview_info()
    original = deepcopy(info)
    workflow, graph = build_preview_workflow(info)
    assert build_preview_workflow(info) == (workflow, graph) and info == original
    assert {node["class_type"] for node in graph.values()} == set(PREVIEW_TYPES)
    assert len(graph) == 3 and len(workflow["links"]) == 2
    assert graph["3"]["inputs"]["plan"] == ["2", 0]
    assert {node["type"] for node in workflow["nodes"]} == {*PREVIEW_TYPES, "MarkdownNote"}
    preview = next(node for node in workflow["nodes"] if node["type"] == PREVIEW_NODE)
    assert preview["widgets_values"] == [0, 768]
    assert preview["inputs"] == [{"name": "plan", "type": "T8_H3_OUTPAINT_PLAN", "link": 2}]
    assert execution_signature(workflow)


@pytest.mark.parametrize("damage", ["missing", "not_output"])
def test_preview_builder_needs_available_executable_preview_schema(damage):
    info = _preview_info()
    if damage == "missing":
        info.pop(PREVIEW_NODE)
    else:
        info[PREVIEW_NODE]["output_node"] = False
    with pytest.raises(ValueError, match="preview"):
        build_preview_workflow(info)
