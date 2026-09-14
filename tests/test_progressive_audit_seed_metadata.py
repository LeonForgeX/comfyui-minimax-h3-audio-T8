"""Independent workflow audit must honor actual widget schema metadata."""
from copy import deepcopy
import json

import pytest

from tools.api_to_frontend_workflow import convert, node_properties
from tools.audit_progressive_workflows import audit_candidate
from tools.frontend_workflow_compat import normalize_native_widget_inputs


@pytest.mark.parametrize("name,module,expected", [
    ("MiniMaxH3MemoryEfficientSageAttentionPatch", "custom_nodes.ComfyUI-KJNodes", None),
    ("SolAttentionPatch", "custom_nodes.ComfyUI-sol-attn", None),
    ("UNETLoader", "nodes", "comfy-core"),
    ("SamplerCustomAdvanced", "comfy_extras.nodes_custom_sampler", "comfy-core"),
    ("MiniMaxH3ProgressiveLongVideoEXPT8", "custom_nodes.minimax-h3-audio-T8", "minimax-h3-audio-t8"),
    ("Unknown", "", None),
    ("Unknown", None, None),
    ("Unknown", 7, None),
    ("Unknown", [], None),
    ("Unknown", {}, None),
])
def test_registry_owner_uses_schema_not_class_prefix(name, module, expected):
    result = node_properties(name, {"python_module": module})
    assert result.get("cnr_id") == expected
    assert result["Node name for S&R"] == name


def test_explicit_registry_metadata_is_preserved():
    assert node_properties("External", {"python_module": "custom_nodes.external", "cnr_id": "verified-owner"}) == {
        "Node name for S&R": "External", "cnr_id": "verified-owner"}


def test_null_module_keeps_explicit_owner_and_graph_inputs():
    info = {"Example": {"python_module": None, "cnr_id": "verified-owner",
        "input": {"required": {"enabled": ["BOOLEAN", {"default": False}]}},
        "output": [], "output_name": []}}
    graph = {"1": {"class_type": "Example", "inputs": {"enabled": True}}}
    node = convert(graph, info, "Native unregistered schema")["nodes"][0]
    assert node["properties"]["cnr_id"] == "verified-owner"
    assert node["widgets_values"] == [True]


def test_native_tuple_schema_preserves_sol_style_widget_values():
    info = {"Patch": {"input": {"required": {
        "enabled": ("BOOLEAN", {"default": True}),
        "tau": ("FLOAT", {"default": 1.3}),
        "thresh_type": (["diag", "exact"], {"default": "diag"}),
    }}, "output": ("MODEL",), "output_name": ("model",)}}
    graph = {"21": {"class_type": "Patch", "inputs": {
        "enabled": True, "tau": 0.5, "thresh_type": "exact"}}}
    native = convert(graph, info, "Native INPUT_TYPES")
    serialized = convert(graph, json.loads(json.dumps(info)), "Native INPUT_TYPES")
    assert native["nodes"][0]["widgets_values"] == [True, 0.5, "exact"]
    assert native["nodes"] == serialized["nodes"]
    assert native["links"] == serialized["links"]
    normalize_native_widget_inputs(native)
    assert audit_candidate(graph, native, info)["explicit_widget_values"] == 3


@pytest.mark.parametrize("name,options,control", [
    ("base_seed", {"control_after_generate": True}, True),
    ("seed", {"control_after_generate": False}, False),
    ("noise_seed", {"control_after_generate": False}, False),
    ("seed", {"control_after_generate": True}, True),
    ("seed", {}, True),
    ("noise_seed", {}, True),
    ("base_seed", {}, False),
])
def test_audit_declared_and_legacy_control(name, options, control):
    info = {"Example": {"input": {"required": {
        name: ["INT", {"default": 7, **options}],
        "seed_policy": [["increment", "fixed"], {"default": "increment"}],
    }}, "output": ["STRING"], "output_name": ["status"]}}
    graph = {"11": {"class_type": "Example", "inputs": {name: 7, "seed_policy": "increment"}}}
    workflow = convert(graph, info, "Audit metadata contract")
    normalize_native_widget_inputs(workflow)
    values = workflow["nodes"][0]["widgets_values"]
    assert values == ([7, "fixed", "increment"] if control else [7, "increment"])
    assert audit_candidate(graph, workflow, info)["explicit_widget_values"] == 2

    changed = deepcopy(workflow)
    changed["nodes"][0]["widgets_values"][-1] = "fixed"
    with pytest.raises(ValueError, match="Widget value changed"):
        audit_candidate(graph, changed, info)

    extra = deepcopy(workflow)
    extra["nodes"][0]["widgets_values"].append("fixed")
    with pytest.raises(ValueError, match="Unaccounted serialized widgets"):
        audit_candidate(graph, extra, info)

    if control:
        for replacement in ("increment", "randomize", 0, False):
            changed = deepcopy(workflow)
            changed["nodes"][0]["widgets_values"][1] = replacement
            with pytest.raises(ValueError, match="Seed control must stay fixed"):
                audit_candidate(graph, changed, info)
