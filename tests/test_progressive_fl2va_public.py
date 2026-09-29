"""CPU contract for the opt-in single-clip FL2VA Progressive route.

This checks real torch conditioning geometry, never a trained H3/GPU render.
"""

from copy import deepcopy

import pytest
import torch

from h3_audio_t8_pkg.progressive_sampling_contract import (plan_progressive_first_sample,
    plan_progressive_initialized_sample)
from h3_audio_t8_pkg.progressive_sampling_runtime import prepare_stage_conditioning


def plan():
    video = torch.zeros(1, 24, 2, 4, 8)
    audio = torch.zeros(1, 32, 2, 8)
    return plan_progressive_first_sample(video, audio, torch.linspace(1, 0, 9),
                                         low_evaluations=6, task="fl2va")


def conditioning():
    return [[torch.zeros(1, 3, 8), {
        "minimax_keyframes": [
            {"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, 4, 8), .2)},
            {"resolved_frame_index": 4, "latent": torch.full((1, 24, 1, 4, 8), .8)},
        ],
        "minimax_frame_count": 5,
        "minimax_token_tags": torch.tensor([0, 0, 1], dtype=torch.long),
    }]]


@pytest.mark.parametrize("policy", ["legacy_bilinear", "preserve_mean"])
def test_two_endpoints_keep_order_and_original_target_latents(policy):
    source = conditioning()
    original = deepcopy(source)
    low, high = prepare_stage_conditioning(source, plan(), positive=True, guide_resize=policy)
    lo = low[0][1]["minimax_keyframes"]
    hi = high[0][1]["minimax_keyframes"]
    assert [item["resolved_frame_index"] for item in lo] == [0, 4]
    assert [item["resolved_frame_index"] for item in hi] == [0, 4]
    for index in (0, 1):
        assert lo[index]["latent"].shape == (1, 24, 1, 2, 4)
        assert hi[index]["latent"] is source[0][1]["minimax_keyframes"][index]["latent"]
        torch.testing.assert_close(lo[index]["latent"].mean(), hi[index]["latent"].mean())
        torch.testing.assert_close(source[0][1]["minimax_keyframes"][index]["latent"],
                                   original[0][1]["minimax_keyframes"][index]["latent"])
    assert low[0][1]["minimax_token_tags"] is high[0][1]["minimax_token_tags"] is source[0][1]["minimax_token_tags"]


def test_missing_or_swapped_tail_rejected_before_low_stage():
    with pytest.raises(ValueError, match="FL2VA requires"):
        prepare_stage_conditioning([[torch.zeros(1, 3, 8), {}]], plan(), positive=True)
    for indices in ((0,), (4, 0), (0, 3), (0, 4, 4)):
        source = conditioning()
        source[0][1]["minimax_keyframes"] = [
            {"resolved_frame_index": index, "latent": torch.zeros(1, 24, 1, 4, 8)} for index in indices]
        with pytest.raises(ValueError):
            prepare_stage_conditioning(source, plan(), positive=True)


def test_wrong_size_nan_or_frame_count_rejected():
    for change in ("wrong_size", "nan", "wrong_time"):
        source = conditioning()
        if change == "wrong_size":
            source[0][1]["minimax_keyframes"][1]["latent"] = torch.zeros(1, 24, 1, 2, 4)
        elif change == "nan":
            source[0][1]["minimax_keyframes"][1]["latent"][0, 0, 0, 0, 0] = float("nan")
        else:
            source[0][1]["minimax_frame_count"] = 22
        with pytest.raises(ValueError):
            prepare_stage_conditioning(source, plan(), positive=True)


def test_extra_reference_or_changed_schema_cannot_be_silently_used():
    for key, value in (("minimax_refs", [{"kind": "image"}]),
                       ("t8_long_video_schema", 2)):
        source = conditioning()
        source[0][1][key] = value
        with pytest.raises(ValueError, match="FL2VA"):
            prepare_stage_conditioning(source, plan(), positive=True)


def test_foreign_hook_is_preserved_in_both_stages(caplog):
    source = conditioning()
    hook = lambda *args: args
    source[0][1]["hook"] = hook
    low, high = prepare_stage_conditioning(source, plan(), positive=True)
    assert low[0][1]["hook"] is high[0][1]["hook"] is hook
    assert source[0][1]["hook"] is hook
    assert "hook" in caplog.text


def test_other_tasks_and_initialized_input_stay_separate():
    assert plan().task == "fl2va"
    video = torch.zeros(1, 24, 2, 4, 8)
    audio = torch.zeros(1, 32, 2, 8)
    with pytest.raises(ValueError, match="empty-AV fl2va"):
        plan_progressive_initialized_sample(video, audio, torch.linspace(1, 0, 9),
                                             low_evaluations=6, task="fl2va")
    with pytest.raises(ValueError, match="T2VA cannot"):
        prepare_stage_conditioning(conditioning(), plan_progressive_first_sample(
            video, audio, torch.linspace(1, 0, 9), low_evaluations=6), positive=True)
