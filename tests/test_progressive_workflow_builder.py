from copy import deepcopy

import pytest

from tools.build_progressive_workflows import build_workflow


def fixture(task):
    prompt = {
        "6": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {"task_type": task.upper()}},
        "7": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "sampler_name": "euler", "scheduler": "native_flow", "steps": 8}},
        "10": {"class_type": "MiniMaxH3ProgressiveSamplerEXPT8", "inputs": {
            "task": task, "model": ["7", 0], "sampler": ["7", 1], "sigmas": ["7", 2],
            "av_latent": ["6", 1], "seed": 123, "low_evaluations": 6}},
    }
    schema = {
        "MiniMaxH3AudioConditioningT8": {"input": {"required": {"task_type": [["T2VA", "I2VA"], {}]}},
            "output": ["CONDITIONING", "LATENT"], "output_name": ["positive", "av_latent"]},
        "MiniMaxH3DualClockSamplerT8": {"input": {"required": {
            "sampler_name": [["euler"], {}], "scheduler": [["native_flow"], {}], "steps": ["INT", {"default": 8}]}},
            "output": ["MODEL", "SAMPLER", "SIGMAS"], "output_name": ["model", "sampler", "sigmas"]},
        "MiniMaxH3ProgressiveSamplerEXPT8": {"input": {"required": {
            "model": ["MODEL", {}], "sampler": ["SAMPLER", {}], "sigmas": ["SIGMAS", {}],
            "av_latent": ["LATENT", {}], "task": [["t2va", "i2va"], {}],
            "seed": ["INT", {"default": 123, "control_after_generate": True}],
            "low_evaluations": ["INT", {"default": 6}]}},
            "output": ["LATENT", "STRING"], "output_name": ["av_latent", "report_json"]},
    }
    return prompt, schema


@pytest.mark.parametrize("task", ["t2va", "i2va"])
def test_frontend_uses_real_converter_preserves_recipe_and_explicit_seed_control(task):
    prompt, schema = fixture(task)
    before = deepcopy(prompt)
    workflow = build_workflow(prompt, schema)
    assert prompt == before
    assert workflow["last_node_id"] == 4 and len(workflow["links"]) == 4
    sampler = workflow["nodes"][2]
    assert sampler["type"] == "MiniMaxH3ProgressiveSamplerEXPT8"
    assert sampler["widgets_values"] == [task, 123, "fixed", 6]
    assert len(sampler["inputs"]) == 4
    assert workflow["nodes"][-1]["type"] == "MarkdownNote"
    note = workflow["nodes"][-1]["widgets_values"][0]
    assert "1.77.0 EXP" in note and "latent_upscale_models" in note and "不是VDN的8+4" in note
    assert workflow["extra"]["progressive_delivery_status"] == "exp_release_1_77_0_short_clips_only_audio_limits_documented"
    assert build_workflow(prompt, schema)["id"] == workflow["id"]


@pytest.mark.parametrize("fault", ["task", "sampler", "schedule", "type"])
def test_unqualified_routes_do_not_silently_become_frontend_candidates(fault):
    prompt, schema = fixture("i2va")
    if fault == "task":
        prompt["6"]["inputs"]["task_type"] = "T2VA"
    elif fault == "sampler":
        prompt["7"]["inputs"]["sampler_name"] = "dual_clock_euler"
    elif fault == "schedule":
        prompt["7"]["inputs"]["scheduler"] = "other"
    else:
        prompt["10"]["class_type"] = "OtherSampler"
    with pytest.raises(ValueError):
        build_workflow(prompt, schema)


def test_qa_runtime_paths_are_bound_to_the_actual_project():
    from tools import run_progressive_workflow_qa as qa
    config = qa.probe_resource_config(qa.CORE, qa.PROJECT)
    assert qa.PROJECT.name == "minimax-h3-audio-T8"
    assert config["t8_probe_nodes"]["custom_nodes"] == str(qa.PROJECT / "tools")


def test_independent_candidate_audit_checks_source_values_and_physical_links():
    from tools.audit_progressive_workflows import audit_candidate
    prompt, schema = fixture("t2va")
    result = audit_candidate(prompt, build_workflow(prompt, schema), schema)
    assert result["nodes"] == 3 and result["edges"] == 4
    assert result["explicit_widget_values"] == 7
    assert "not_browser" in result["scope"]


@pytest.mark.parametrize("fault", ["seed", "low_steps", "edge", "extra_edge", "node", "mode", "extra_widget"])
def test_candidate_audit_rejects_real_serialization_mutations(fault):
    from tools.audit_progressive_workflows import audit_candidate
    prompt, schema = fixture("t2va")
    workflow = build_workflow(prompt, schema)
    if fault == "seed":
        workflow["nodes"][2]["widgets_values"][2] = "randomize"
    elif fault == "low_steps":
        workflow["nodes"][2]["widgets_values"][3] = 4
    elif fault == "edge":
        workflow["links"][0][2] += 1
    elif fault == "extra_edge":
        workflow["links"].append([99, 1, 0, 2, 0, "LATENT"])
    elif fault == "node":
        workflow["nodes"].pop(0)
    elif fault == "mode":
        workflow["nodes"][2]["mode"] = 4
    else:
        workflow["nodes"][2]["widgets_values"].append(123)
    with pytest.raises(ValueError):
        audit_candidate(prompt, workflow, schema)


def test_api_comparison_rejects_changes_and_allows_only_verified_default():
    from tools.audit_progressive_workflows import audit_api
    prompt, schema = fixture("i2va")
    api = deepcopy(prompt)
    ids = {k: str(i+1) for i, k in enumerate(prompt)}
    api = {ids[k]: node for k, node in api.items()}
    for node in api.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list):
                node["inputs"][key] = [ids[value[0]], value[1]]
    assert "not_proof" in audit_api(prompt, api, schema)["scope"]
    schema["MiniMaxH3AudioConditioningT8"]["input"]["optional"] = {"allow": ["BOOLEAN", {"default": True}]}
    api["1"]["inputs"]["allow"] = True
    assert audit_api(prompt, api, schema)["schema_defaults_added"] == ["6.allow"]
    api["1"]["inputs"]["allow"] = 1
    with pytest.raises(ValueError):
        audit_api(prompt, api, schema)
