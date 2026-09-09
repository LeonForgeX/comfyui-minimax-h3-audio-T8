from copy import deepcopy

import pytest

from tools.run_video_outpaint_selected_probe import (
    build_selected_prompt, selected_source_mode, submit_with_owned_gpu_trigger,
    verify_selected_delivery,
)
from tools.run_video_outpaint_candidate_probe import build_candidate_prompt
from test_video_outpaint_workflows import fixture


def prior():
    return {"prompt": build_candidate_prompt({"prompt": fixture()["prompt"]}), "candidate_id": "a"*64}


@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode", "legacy"])
def test_selected_postflight_binds_preview_mode_to_delivery(monkeypatch, mode):
    from tools import run_video_outpaint_selected_probe as selected

    original = prior()
    original.update(source_sha256="b" * 64, plan_sha256="c" * 64)
    inputs = original["prompt"]["generated_candidate"]["inputs"]
    expected = "preserve_source" if mode == "legacy" else mode
    original["preview_report"] = {"source_exact_before_encoding": expected == "preserve_source"}
    if mode == "legacy":
        inputs.pop("source_mode", None)
    else:
        inputs["source_mode"] = mode
        original["preview_report"].update(source_mode=mode, source_reconstructed=mode == "joint_decode")
    before = deepcopy(original)
    calls = []
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"verified": True}
    monkeypatch.setattr(selected.t8.cases, "verify_delivery", verify)
    assert verify_selected_delivery(original, "sidecar", "d" * 64) == {"verified": True}
    assert calls == [(("sidecar", "d" * 64, "b" * 64, "c" * 64),
                      {"expected_source_mode": expected})]
    assert original == before


@pytest.mark.parametrize("damage", ["missing", "invalid", "exact", "reconstructed", "graph", "bool_exact"])
def test_selected_preview_policy_contradictions_rejected(damage):
    data = prior()
    data["prompt"]["generated_candidate"]["inputs"]["source_mode"] = "joint_decode"
    data["preview_report"] = {"source_mode": "joint_decode", "source_exact_before_encoding": False,
                              "source_reconstructed": True}
    if damage == "missing":
        data.pop("preview_report")
    elif damage == "invalid":
        data["preview_report"]["source_mode"] = None
    elif damage == "exact":
        data["preview_report"]["source_exact_before_encoding"] = True
    elif damage == "reconstructed":
        data["preview_report"]["source_reconstructed"] = False
    elif damage == "graph":
        data["prompt"]["generated_candidate"]["inputs"]["source_mode"] = "preserve_source"
    else:
        data["preview_report"]["source_exact_before_encoding"] = 0
    with pytest.raises(ValueError):
        selected_source_mode(data)


def test_selection_is_explicit_and_preserves_model_vae_preparation_without_new_seed():
    original = prior()
    before = deepcopy(original)
    graph = build_selected_prompt(original)
    assert original == before
    kinds = {n["class_type"] for n in graph.values()}
    assert not kinds & {"MiniMaxH3VideoOutpaintCandidateT8", "MiniMaxH3VideoOutpaintPrepareT8", "CLIPLoader", "SaveVideo"}
    assert graph["read_candidate"]["inputs"]["candidate_id"] == original["candidate_id"]
    assert graph["test_select"]["inputs"]["confirm_selection"] is True
    assert graph["continue_selected"]["inputs"]["selected"] == ["test_select", 0]
    assert set(graph["continue_selected"]["inputs"]) == {"model", "selected"}
    assert "color_match" not in graph["save_selected"]["inputs"]
    assert graph["save_selected"]["inputs"]["sampled"] == ["continue_selected", 0]
    assert graph["selection_identifier"]["inputs"]["source"] == ["test_select", 2]


@pytest.mark.parametrize("damage", ["short_id", "path_id", "missing_candidate", "reserved_id"])
def test_ambiguous_or_invalid_selection_is_not_guessed(damage):
    data = prior()
    if damage == "short_id":
        data["candidate_id"] = "a"*63
    elif damage == "path_id":
        data["candidate_id"] = "../candidate.json"
    elif damage == "missing_candidate":
        data["prompt"].pop("generated_candidate")
    else:
        data["prompt"]["test_select"] = {"class_type": "anything", "inputs": {}}
    with pytest.raises(ValueError):
        build_selected_prompt(data)


def test_selected_prompt_is_reusable_for_fresh_process_resume_without_regeneration():
    graph = build_selected_prompt(prior())
    assert graph["continue_selected"]["class_type"] == "MiniMaxH3VideoOutpaintContinueCandidateT8"
    assert "resume" not in graph["continue_selected"]["inputs"]
    assert not any(node["class_type"] == "MiniMaxH3VideoOutpaintCandidateT8" for node in graph.values())


@pytest.mark.asyncio
async def test_owned_gpu_interrupt_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="positive"):
        await submit_with_owned_gpu_trigger(
            object(), server="http://127.0.0.1:1", prompt={}, timeout_seconds=1,
            trigger_free_below_mib=0, trigger_timeout_seconds=1)
