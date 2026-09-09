from __future__ import annotations

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_guidance import build_outpaint_guidance
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_regional_conditioning import (
    REGIONAL_PAYLOAD_KEY,
    OutpaintRegionalConditioningProvider,
    prepare_outpaint_regional_conditioning,
)


class FakeClip:
    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, prompt):
        length = len(prompt.split()) + 1
        offset = sum(prompt.encode()) % 31
        cross = torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 4) + offset
        return [[cross, {"minimax_token_tags": torch.ones(length, dtype=torch.long),
                         "pooled_output": None}]]


def _plan():
    return build_outpaint_plan(
        source_sha256="b" * 64, width=320, height=240, frame_count=22,
        aspect="custom", left=64, top=64, right=64, bottom=64,
        generation_megapixels=0.5,
    )


def test_regional_conditioning_is_immutable_bound_and_reloadable(tmp_path):
    plan = _plan()
    guidance = build_outpaint_guidance(plan, [
        {"shot": 0, "region": "top", "prompt": "blue cloudy sky"},
        {"shot": 0, "region": "bottom", "prompt": "stone floor"},
    ], [])
    provider = prepare_outpaint_regional_conditioning(
        FakeClip(), "global subject", guidance, plan, tmp_path)
    result = provider(0, 0)
    assert isinstance(provider, OutpaintRegionalConditioningProvider)
    assert result[0][0].shape[1] == provider.binding["shots"][0]["text_len"]
    assert provider.binding["shots"][0]["regions"][0]["text_key_start"] == 3
    runtime = result[0][1]["model_conds"][REGIONAL_PAYLOAD_KEY]
    assert runtime.cond == f"{provider.binding['binding_sha256']}:0"
    reloaded = OutpaintRegionalConditioningProvider(tmp_path, plan)
    assert reloaded.verify() == provider.verify()
    assert torch.equal(reloaded(0, 0)[0][0], result[0][0])


def test_changed_prompts_refuse_to_overwrite_same_run(tmp_path):
    plan = _plan()
    guidance = build_outpaint_guidance(
        plan, [{"shot": 0, "region": "expanded", "prompt": "first"}], [])
    prepare_outpaint_regional_conditioning(FakeClip(), "base", guidance, plan, tmp_path)
    changed = build_outpaint_guidance(
        plan, [{"shot": 0, "region": "expanded", "prompt": "second"}], [])
    with pytest.raises(ValueError, match="differ"):
        prepare_outpaint_regional_conditioning(FakeClip(), "base", changed, plan, tmp_path)


def test_plain_and_regional_manifests_are_mutually_exclusive(tmp_path):
    (tmp_path / "outpaint_conditioning.json").write_text("{}", encoding="utf-8")
    guidance = build_outpaint_guidance(
        _plan(), [{"shot": 0, "region": "top", "prompt": "sky"}], [])
    with pytest.raises(ValueError, match="plain conditioning"):
        prepare_outpaint_regional_conditioning(FakeClip(), "base", guidance, _plan(), tmp_path)
