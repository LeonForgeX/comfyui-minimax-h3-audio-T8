from copy import deepcopy

import pytest

from tools.run_video_outpaint_recompose_probe import build_recompose_prompt
from tools.run_video_outpaint_selected_probe import build_selected_prompt
from tools.run_video_outpaint_candidate_probe import build_candidate_prompt
from test_video_outpaint_workflows import fixture


def _prior():
    candidate = {"prompt": build_candidate_prompt({"prompt": fixture()["prompt"]}), "candidate_id": "a" * 64}
    return {"prompt": build_selected_prompt(candidate)}


def test_recompose_graph_loads_only_completed_selection_and_video_vae():
    prior = _prior()
    before = deepcopy(prior)
    graph = build_recompose_prompt(prior, candidate_name="learned_candidate_01", selection_id="b" * 64)
    assert prior == before
    kinds = {node["class_type"] for node in graph.values()}
    assert "MiniMaxH3VideoOutpaintLoadCompletedSelectionT8" in kinds
    assert "MiniMaxH3VideoOutpaintComposeCandidateT8" in kinds
    assert "VAELoader" in kinds and "LoadVideo" in kinds and "MiniMaxH3VideoOutpaintPlanT8" in kinds
    assert not kinds & {"UNETLoader", "CLIPLoader", "MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward",
                        "MiniMaxH3VideoOutpaintContinueCandidateT8", "MiniMaxH3VideoOutpaintSampleT8"}
    assert graph["save_recomposed"]["inputs"]["sampled"] == ["load_completed_selection", 0]


@pytest.mark.parametrize("selection", ["", "a" * 63, "A" * 64, "../selection.json"])
def test_recompose_graph_never_guesses_selection(selection):
    with pytest.raises(ValueError, match="selection_id"):
        build_recompose_prompt(_prior(), candidate_name="learned_candidate_01", selection_id=selection)
