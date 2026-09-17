"""Candidate route semantics, independent of expensive model execution."""
import asyncio
from copy import deepcopy

import pytest
import h3_audio_t8_pkg

from tools.build_semantic_bridge_workflows import DUAL_REVIEW, recipes, release_name
from tools.run_semantic_bridge_workflow_probe import instrument, graph_assets


@pytest.fixture(scope="module")
def graphs():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    info = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}
    return recipes(info)


def test_relay_candidate_routes_the_actual_patched_model_and_latent(graphs):
    graph = graphs["H3_SemanticBridge_Relay_EXP"]
    assert graph["9"]["inputs"]["execution_mode"] == "apply_exp"
    assert graph["9"]["inputs"]["model"] == ["1", 0]
    assert graph["2"]["inputs"]["model"] == ["9", 0]
    assert graph["10"]["inputs"]["model"] == ["2", 0]
    assert graph["10"]["inputs"]["av_latent"] == ["9", 2]
    assert graph["12"]["inputs"]["conditioning"] == ["9", 1]
    assert graph["13"]["inputs"]["latent_image"] == ["9", 2]
    assert "length" not in graph["9"]["inputs"]
    assert graph["36"]["inputs"]["length"] == 73
    assert len(graph["36"]["inputs"]["local_prompts"].splitlines()) == 2
    assert "40" not in graph  # No second external apply after the Relay binding.


def test_external_candidate_applies_exactly_once(graphs):
    graph = graphs["H3_BUNNY_External_EXP"]
    assert "semantic_bridge" not in graph["9"]["inputs"]
    assert graph["40"]["inputs"]["conditioning"] == ["9", 0]
    assert graph["12"]["inputs"]["conditioning"] == ["40", 0]


@pytest.mark.parametrize("key", ["H3_SemanticBridge_SingleLoop_8s_EXP", "H3_SemanticBridge_DualIndependent_8s_EXP"])
def test_loop_candidates_keep_two_segment_recipe_and_relay(graphs, key):
    graph = graphs[key]
    values = graph["8"]["inputs"]
    assert values["total_duration_seconds"] == 8
    assert values["render_window_frames"] == 124 and values["context_frames"] == 22
    assert values["prompt_relay_mode"] == "apply_exp"
    assert values["eav_mode"] == "disabled"
    assert graph["7"]["inputs"]["length"] == 193
    if "Dual" in key:
        assert values["semantic_bridge_pass1"] != values["semantic_bridge_pass2"]
        assert values["coarse_steps"] == values["refine_steps"] == 4
        assert values["low_context_source"] == "accepted_picture_low_context_v1"


@pytest.mark.parametrize("key, expected_recipe", [
    ("H3_BUNNY_External_EXP", "external"), ("H3_SemanticBridge_Relay_EXP", "relay"),
    ("H3_SemanticBridge_SingleLoop_8s_EXP", "loop"),
    ("H3_SemanticBridge_DualIndependent_8s_EXP", "loop")])
def test_probe_instrumentation_only_adds_report_reader(graphs, key, expected_recipe):
    graph = deepcopy(graphs[key])
    recipe, canvas = instrument(graph)
    assert recipe == expected_recipe
    assert canvas[2:] == ([192, "24/1"] if recipe == "loop" else [73, "24/1"])
    assert graph.pop("90")["class_type"] == "PreviewAny"
    assert graph == graphs[key]


def test_probe_refuses_silent_report_only_relay(graphs):
    graph = deepcopy(graphs["H3_SemanticBridge_Relay_EXP"])
    graph["9"]["inputs"]["execution_mode"] = "report_only"
    with pytest.raises(ValueError, match="actually apply"):
        instrument(graph)


def test_probe_hashes_upscaler_and_both_distinct_bridge_weights(graphs, tmp_path, monkeypatch):
    from tools import run_semantic_bridge_workflow_probe as probe
    monkeypatch.setattr(probe, "file_identity", lambda path: {"path": str(path)})
    paths = [item["path"].replace("\\", "/") for item in graph_assets(
        graphs["H3_SemanticBridge_DualIndependent_8s_EXP"], tmp_path)]
    assert len(paths) == len(set(paths)) == 8
    assert sum("/semantic_bridge/" in path for path in paths) == 2
    assert sum("/latent_upscale_models/" in path for path in paths) == 1


def test_dual_spatial_candidate_preserves_both_bridges_and_second_sampling(graphs):
    graph = graphs["H3_SemanticBridge_DualIndependent_8s_EXP"]
    values = graph["8"]["inputs"]
    assert (values["low_width"], values["low_height"], values["width"], values["height"]) == (640, 320, 896, 448)
    assert values["coarse_steps"] == values["refine_steps"] == 4
    assert values["model_pass1"] == ["2", 0] and values["model_pass2"] == ["3", 0]
    assert values["upscaler_model"] == "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    for key in ("33", "34"):
        assert graph[key]["inputs"]["enabled"] is True
        assert graph[key]["inputs"]["alpha"] == .10
    assert graph["33"]["inputs"]["model_name"] != graph["34"]["inputs"]["model_name"]
    # A diagnostic Bridge bypass must not remove MODEL2 or its four forwards.
    control = deepcopy(graph)
    control["34"]["inputs"]["enabled"] = False
    assert control["8"] == graph["8"]
    assert control["3"] == graph["3"]


def test_release_review_only_promotes_the_bound_repaired_dual_recipe(graphs):
    assert release_name("H3_SemanticBridge_DualIndependent_8s_EXP") == "H3_SemanticBridge_DualIndependent_8s_Advanced"
    for name in graphs:
        if "DualIndependent" not in name:
            assert release_name(name) == name
    assert DUAL_REVIEW["status"] == "accepted_in_this_review_scope"
    assert DUAL_REVIEW["not_universal_quality_claim"] is True
    assert DUAL_REVIEW["media_sha256"] == "6da418003510e040d2b3ccc7d9961dcb8d6b652395c0b50425f326159348ce9c"
