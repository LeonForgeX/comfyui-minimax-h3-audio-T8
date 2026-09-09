from copy import deepcopy

import pytest

from tools.run_video_outpaint_candidate_probe import build_candidate_prompt
from tools.run_video_outpaint_regional_candidate_probe import build_regional_candidate_prompt
from test_video_outpaint_workflows import fixture


def test_learned_candidate_probe_uses_prior_plan_models_and_prepared_cache_without_selection():
    prior = {"prompt": fixture()["prompt"]}
    before = deepcopy(prior)
    graph = build_candidate_prompt(prior)
    assert prior == before
    kinds = {node["class_type"] for node in graph.values()}
    assert {"MiniMaxH3VideoOutpaintCandidateT8", "MiniMaxH3VideoOutpaintLoadPreparedT8", "PreviewAny"} <= kinds
    assert not kinds & {"CLIPLoader", "MiniMaxH3VideoOutpaintPrepareT8", "MiniMaxH3VideoOutpaintSelectCandidateT8",
        "MiniMaxH3VideoOutpaintContinueCandidateT8", "MiniMaxH3VideoOutpaintComposeT8", "SaveVideo"}
    candidate = graph["generated_candidate"]["inputs"]
    sample = next(n["inputs"] for n in prior["prompt"].values() if n["class_type"] == "MiniMaxH3VideoOutpaintSampleT8")
    assert candidate["seed"] == sample["seed"] and candidate["steps"] == sample["steps"]
    assert candidate["model"] == sample["model"] and candidate["resume"] is False
    assert graph["candidate_identifier"]["inputs"]["source"] == ["generated_candidate", 3]
    assert sum(node["class_type"] == "VAELoader" for node in graph.values()) == 1


def test_missing_prior_stage_is_not_guessed():
    prior = {"prompt": fixture()["prompt"]}
    prior["prompt"] = {k: n for k, n in prior["prompt"].items() if n["class_type"] != "MiniMaxH3VideoOutpaintSampleT8"}
    with pytest.raises(ValueError, match="exactly one"):
        build_candidate_prompt(prior)


@pytest.mark.parametrize("builder", [build_candidate_prompt, build_regional_candidate_prompt])
@pytest.mark.parametrize("mode", ["joint_decode", "preserve_source", "legacy_absent"])
def test_candidate_probe_inherits_explicit_finish_policy_without_mutating_prior(builder, mode):
    prior = {"prompt": fixture()["prompt"]}
    compose = next(n["inputs"] for n in prior["prompt"].values()
                   if n["class_type"] == "MiniMaxH3VideoOutpaintComposeT8")
    if mode == "legacy_absent":
        compose.pop("source_mode", None)
        compose.pop("geometry_align", None)
    else:
        compose.update(source_mode=mode, geometry_align=True)
    before = deepcopy(prior)
    graph = builder(prior)
    candidate = next(n["inputs"] for n in graph.values()
                     if n["class_type"] == "MiniMaxH3VideoOutpaintCandidateT8")
    assert candidate["source_mode"] == ("preserve_source" if mode == "legacy_absent" else mode)
    assert candidate["geometry_align"] is (mode != "legacy_absent")
    assert prior == before


@pytest.mark.parametrize("builder", [build_candidate_prompt, build_regional_candidate_prompt])
@pytest.mark.parametrize("field,value", [
    ("source_mode", None), ("source_mode", "unknown"), ("source_mode", False),
    ("geometry_align", "false"), ("geometry_align", 1),
])
def test_candidate_probe_does_not_drop_invalid_finish_inputs(builder, field, value):
    prior = {"prompt": fixture()["prompt"]}
    compose = next(n["inputs"] for n in prior["prompt"].values()
                   if n["class_type"] == "MiniMaxH3VideoOutpaintComposeT8")
    compose[field] = value
    with pytest.raises(ValueError):
        builder(prior)
