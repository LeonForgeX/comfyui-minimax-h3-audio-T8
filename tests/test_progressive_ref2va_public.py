"""CPU geometry and ownership checks, without trained weights or GPU claims."""
from copy import deepcopy

import pytest
import torch

from h3_audio_t8_pkg.progressive_sampling_contract import (
    plan_progressive_first_sample, plan_progressive_initialized_sample)
from h3_audio_t8_pkg.progressive_sampling_runtime import prepare_stage_conditioning
from h3_audio_t8_pkg.progressive_references import describe_references


def plan():
    return plan_progressive_first_sample(torch.zeros(1, 24, 2, 4, 8),
        torch.zeros(1, 32, 2, 8), torch.linspace(1, 0, 9), low_evaluations=6, task="ref2va")


def block(kind):
    result = {"kind": kind}
    if kind != "audio":
        time = 1 if kind == "image" else 7
        result.update(latent_h=6, latent_w=10, latent=torch.randn(1, 24, time, 6, 10))
        if kind != "image":
            result.update(latent_t=time, ref_audio_t=0, audio_latent=None)
    if kind in {"audio", "video_audio"}:
        result.update(ref_audio_t=37, audio_latent=torch.randn(1, 32, 2, 37))
    return result


@pytest.mark.parametrize("kinds", [("image",), ("image", "image"), ("video",),
    ("video_audio",), ("audio",), ("image", "video_audio", "audio")])
def test_original_reference_geometry_order_audio_and_tags_survive_both_stages(kinds):
    refs = [block(kind) for kind in kinds]
    before = deepcopy(refs)
    embedding = torch.zeros(1, 3, 8)
    tags = torch.tensor([0, 1, 1])
    source = [[embedding, {"minimax_refs": refs, "minimax_token_tags": tags, "t8_long_video_schema": 1}]]
    low, high = prepare_stage_conditioning(source, plan(), positive=True)
    for branch in (low, high):
        assert branch[0][0] is embedding and branch[0][1]["minimax_token_tags"] is tags
        actual = branch[0][1]["minimax_refs"]
        assert actual is not refs and [r["kind"] for r in actual] == list(kinds)
        for old, new, saved in zip(refs, actual, before):
            assert old is not new
            for key, value in old.items():
                if isinstance(value, torch.Tensor):
                    assert new[key] is value and torch.equal(value, saved[key])
    assert describe_references(refs)[0]["kind"] == kinds[0]


@pytest.mark.parametrize("change", [
    {"kind": "unknown"}, {"latent_h": 3}, {"latent_w": True},
    {"latent": torch.zeros(2, 24, 1, 6, 10)}, {"latent": torch.ones(1, 24, 1, 6, 10) * float("nan")},
    {"latent": torch.ones(1, 24, 1, 6, 10, dtype=torch.int64)},
    {"audio_latent": torch.zeros(1, 32, 2, 5)},
])
def test_invalid_reference_rejected_before_stage_work(change):
    ref = {**block("image"), **change}
    with pytest.raises(ValueError, match="Ref2VA"):
        prepare_stage_conditioning([[torch.zeros(1, 3, 8), {"minimax_refs": [ref]}]], plan(), positive=True)


@pytest.mark.parametrize("change", [{"latent_t": 3}, {"ref_audio_t": 0},
    {"ref_audio_t": 36}, {"audio_latent": None}])
def test_video_and_soundtrack_time_are_validated_independently(change):
    with pytest.raises(ValueError, match="Ref2VA"):
        describe_references([{**block("video_audio"), **change}])


def test_missing_reference_negative_branch_limits_keyframe_and_initialized_boundaries():
    source = [[torch.zeros(1, 3, 8), {}]]
    with pytest.raises(ValueError, match="Ref2VA"):
        prepare_stage_conditioning(source, plan(), positive=True)
    prepare_stage_conditioning(source, plan(), positive=False)
    with pytest.raises(ValueError, match="limits"):
        describe_references([block("image") for _ in range(10)])
    # The native limits are per modality, allowing 9 + 3 + 3 blocks together.
    assert len(describe_references([block("image") for _ in range(9)] +
        [block("video_audio") for _ in range(3)] + [block("audio") for _ in range(3)])) == 15
    source[0][1].update(minimax_refs=[block("image")], minimax_keyframes=[{}])
    with pytest.raises(ValueError, match="first/last"):
        prepare_stage_conditioning(source, plan(), positive=True)
    with pytest.raises(ValueError, match="empty-AV"):
        plan_progressive_initialized_sample(torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8),
            torch.linspace(1, 0, 9), low_evaluations=6, task="ref2va")


def test_user_hooks_are_preserved_with_an_advisory(caplog):
    hook = lambda *args: args
    source = [[torch.zeros(1, 3, 8), {"minimax_refs": [block("image")], "hook": hook}]]
    low, high = prepare_stage_conditioning(source, plan(), positive=True)
    assert low[0][1]["hook"] is high[0][1]["hook"] is hook
    assert "hook" in caplog.text
