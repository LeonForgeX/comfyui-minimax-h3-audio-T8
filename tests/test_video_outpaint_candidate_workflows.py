from copy import deepcopy
import json

import pytest

from tools.build_video_outpaint_candidate_workflows import (
    build_candidate_workflows, ROUTES, CANDIDATE, READ, SELECT, RELOAD, RESTORE,
    COMPLETED, CONTINUE, COMPOSE,
    GUIDANCE, GUIDED_PREPARE, REGIONAL_MODEL,
    DLSS_AUDIT, DLSS_VIDEO,
    COMPATIBILITY,
)
from test_video_outpaint_workflows import fixture
from tools.audit_outpaint_candidate_workflow_roundtrip import audit_roundtrip


def contract():
    from h3_audio_t8_pkg.nodes_video_outpaint_candidates import VIDEO_OUTPAINT_CANDIDATE_DRAFT_NODE_CLASSES
    from h3_audio_t8_pkg.nodes_video_outpaint_reload import MiniMaxH3VideoOutpaintLoadPreparedT8
    from h3_audio_t8_pkg.nodes_video_outpaint_guidance import VIDEO_OUTPAINT_GUIDANCE_DRAFT_NODE_CLASSES
    from h3_audio_t8_pkg.nodes_dlss_nr_advanced import (
        MiniMaxH3DLSSNRRuntimeAuditT8Advanced, MiniMaxH3DLSSNRVideoFileT8Advanced)
    from comfy_extras.nodes_preview_any import PreviewAny
    data = fixture()
    for cls in [*VIDEO_OUTPAINT_CANDIDATE_DRAFT_NODE_CLASSES,
                *VIDEO_OUTPAINT_GUIDANCE_DRAFT_NODE_CLASSES,
                MiniMaxH3DLSSNRRuntimeAuditT8Advanced,
                MiniMaxH3DLSSNRVideoFileT8Advanced,
                MiniMaxH3VideoOutpaintLoadPreparedT8]:
        data["object_info"][cls.define_schema().node_id] = json.loads(json.dumps(cls.GET_NODE_INFO_V1()))
    data["object_info"]["PreviewAny"] = json.loads(json.dumps({"input": PreviewAny.INPUT_TYPES(),
        "output": PreviewAny.RETURN_TYPES, "output_name": PreviewAny.RETURN_TYPES,
        "output_node": PreviewAny.OUTPUT_NODE, "display_name": "Preview as Text"}))
    return data


def test_nine_independent_routes_are_deterministic_and_do_not_modify_source():
    data = contract()
    before = deepcopy(data)
    result = build_candidate_workflows(data["prompt"], data["object_info"])
    assert list(result) == list(ROUTES)
    assert result == build_candidate_workflows(data["prompt"], data["object_info"])
    assert data == before
    for item in result.values():
        assert item["workflow"]["extra"]["outpaint_delivery_status"].endswith("not_human_accepted")
        assert not any(n["type"] == "SaveVideo" for n in item["workflow"]["nodes"])


def test_generate_stops_at_candidate_and_displays_exact_id_without_shifted_seed_widgets():
    data = contract()
    item = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[0]]
    graph = item["prompt"]
    kinds = {n["class_type"] for n in graph.values()}
    assert CANDIDATE in kinds and not kinds & {READ, SELECT, RESTORE, COMPLETED, CONTINUE, COMPOSE, RELOAD}
    assert graph["id_text"]["inputs"]["source"] == ["candidate", 3]
    candidate = next(n for n in item["workflow"]["nodes"] if n["type"] == CANDIDATE)
    assert candidate["widgets_values"] == ["candidate_01", 20260808, "fixed", 20, False, True, False, "joint_decode"]
    assert [i["name"] for i in candidate["inputs"]] == ["model", "prepared", "video_vae"]


def test_read_and_confirm_have_no_models_no_generation_and_no_implicit_selection():
    data = contract()
    result = build_candidate_workflows(data["prompt"], data["object_info"])
    review = result[ROUTES[1]]["prompt"]
    confirm = result[ROUTES[2]]["prompt"]
    assert {n["class_type"] for n in review.values()} == {"LoadVideo", "MiniMaxH3VideoOutpaintPlanT8", RELOAD, READ}
    assert review["read"]["inputs"]["candidate_id"] == ""
    assert confirm["select"]["inputs"]["confirm_selection"] is False
    assert confirm["id_text"]["inputs"]["source"] == ["select", 2]
    for graph in (review, confirm):
        assert all(not any(s in n["class_type"] for s in ("VAELoader", "CLIPLoader", "UNETLoader", "PrepareT8", "Continue", "Compose", "Sample")) for n in graph.values())


def test_continue_uses_explicit_selection_and_stored_settings_not_regeneration():
    data = contract()
    item = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[3]]
    graph = item["prompt"]
    assert graph["restore"]["inputs"]["selection_id"] == ""
    assert graph["continue"]["inputs"]["selected"] == ["restore", 0]
    assert graph["save"]["inputs"]["sampled"] == ["continue", 0]
    assert not any(n["class_type"] in {CANDIDATE, READ, SELECT, "CLIPLoader", "MiniMaxH3VideoOutpaintPrepareT8"} for n in graph.values())
    assert not any(k in graph["continue"]["inputs"] for k in ("seed", "steps", "color_match"))
    assert "color_match" not in graph["save"]["inputs"]


def test_completed_selection_save_has_video_vae_but_no_diffusion_model_or_sampler():
    data = contract()
    graph = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[4]]["prompt"]
    kinds = {node["class_type"] for node in graph.values()}
    assert graph["completed_restore"]["inputs"]["selection_id"] == ""
    assert graph["save_completed"]["inputs"]["sampled"] == ["completed_restore", 0]
    assert COMPLETED in kinds and COMPOSE in kinds and "VAELoader" in kinds
    assert not kinds & {"UNETLoader", "CLIPLoader", CANDIDATE, READ, SELECT, RESTORE, CONTINUE,
                        "MiniMaxH3VideoOutpaintPrepareT8", "MiniMaxH3VideoOutpaintSampleT8"}


def test_guided_routes_are_independent_and_rebind_the_same_prepared_cache():
    data = contract()
    result = build_candidate_workflows(data["prompt"], data["object_info"])
    generate = result[ROUTES[5]]["prompt"]
    assert generate["guided_prepare"]["inputs"]["guidance"] == ["guidance", 0]
    assert generate["regional_model"]["inputs"]["prepared"] == ["guided_prepare", 0]
    assert generate["guided_candidate"]["inputs"]["model"] == ["regional_model", 0]
    assert generate["guided_candidate"]["inputs"]["prepared"] == ["regional_model", 1]
    assert {GUIDANCE, GUIDED_PREPARE, REGIONAL_MODEL, CANDIDATE} <= {
        node["class_type"] for node in generate.values()}
    continuation = result[ROUTES[6]]["prompt"]
    assert continuation["guided_model"]["inputs"]["prepared"] == ["guided_prepared_reload", 0]
    assert continuation["guided_restore"]["inputs"]["prepared"] == ["guided_model", 1]
    assert continuation["guided_continue"]["inputs"]["model"] == ["guided_model", 0]
    assert GUIDED_PREPARE not in {node["class_type"] for node in continuation.values()}


def test_dlss_route_is_postprocess_only_and_matches_current_runtime_schema():
    data = contract()
    graph = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[7]]["prompt"]
    kinds = {node["class_type"] for node in graph.values()}
    assert {RELOAD, COMPLETED, COMPOSE, DLSS_AUDIT, DLSS_VIDEO} <= kinds
    assert not kinds & {"UNETLoader", "CLIPLoader", CANDIDATE, CONTINUE, GUIDED_PREPARE, REGIONAL_MODEL}
    assert graph["dlss_video"]["inputs"]["source_video"] == ["dlss_compose", 0]
    assert graph["dlss_video"]["inputs"]["scale"] == "2.0"
    assert "accept_external_runtime_license" not in graph["dlss_runtime"]["inputs"]
    assert graph["dlss_runtime"]["inputs"]["probe_mode"] == "feature_probe_1_frame"


def test_compatibility_route_has_no_media_or_sampler():
    data = contract()
    graph = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[8]]["prompt"]
    kinds = {node["class_type"] for node in graph.values()}
    assert COMPATIBILITY in kinds and "UNETLoader" in kinds
    assert not kinds & {"LoadVideo", CANDIDATE, CONTINUE, COMPOSE, DLSS_VIDEO, GUIDED_PREPARE}


@pytest.mark.parametrize("damage", ["schema", "stage", "duplicate_stage", "reserved_id", "extra_sink"])
def test_missing_schema_or_ambiguous_source_graph_fails_before_publication(damage):
    data = contract()
    if damage == "schema":
        data["object_info"].pop(READ)
    elif damage == "stage":
        data["prompt"] = {k: n for k, n in data["prompt"].items() if n["class_type"] != "MiniMaxH3VideoOutpaintPlanT8"}
    elif damage == "duplicate_stage":
        data["prompt"]["duplicate"] = deepcopy(next(n for n in data["prompt"].values() if n["class_type"] == "MiniMaxH3VideoOutpaintPlanT8"))
    elif damage == "reserved_id":
        data["prompt"]["candidate"] = {"class_type": "unused", "inputs": {}}
    else:
        data["prompt"]["sink"] = {"class_type": "SaveVideo", "inputs": {}}
    with pytest.raises(ValueError):
        build_candidate_workflows(data["prompt"], data["object_info"])


def _saved_fixture(item):
    saved = deepcopy(item["workflow"])
    for index, source in enumerate(item["prompt"].values()):
        node = next(n for n in saved["nodes"] if n["id"] == index + 1)
        node["widgets_values_named"] = {k: v for k, v in source["inputs"].items() if not isinstance(v, list)}
        for name in node["widgets_values_named"]:
            if not any(pin["name"] == name for pin in node["inputs"]):
                node["inputs"].append({"name": name, "link": None, "widget": {"name": name}})
        if node["type"] == CANDIDATE:
            node["widgets_values_named"]["control_after_generate"] = "fixed"
    return saved


@pytest.mark.parametrize("damage", [None, "shifted_seed", "steps_bool", "missing_named", "edge", "mode"])
def test_roundtrip_auditor_rejects_shifted_widgets_and_rewired_or_disabled_nodes(damage):
    data = contract()
    item = build_candidate_workflows(data["prompt"], data["object_info"])[ROUTES[0]]
    saved = _saved_fixture(item)
    node = next(n for n in saved["nodes"] if n["type"] == CANDIDATE)
    if damage == "shifted_seed":
        node["widgets_values_named"].update(control_after_generate=20, steps=False, resume=True)
    elif damage == "steps_bool":
        node["widgets_values_named"]["steps"] = False
    elif damage == "missing_named":
        node.pop("widgets_values_named")
    elif damage == "edge":
        saved["links"][0][2] += 1
    elif damage == "mode":
        node["mode"] = 2
    if damage is None:
        assert audit_roundtrip(item["prompt"], saved)["all_execution_inputs_equal"]
    else:
        with pytest.raises(ValueError):
            audit_roundtrip(item["prompt"], saved)
