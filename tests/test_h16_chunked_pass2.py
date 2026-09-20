from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.nodes_h16_chunked_pass2 import (
    DeciiaChunkedPass2Sampler,
    _merge_audio_segments,
    _raw_conditioning_from_guider,
)


def test_h16_3_node_is_append_only_and_safe_by_default():
    schema = DeciiaChunkedPass2Sampler.GET_NODE_INFO_V1()
    assert schema["name"] == "DeciiaChunkedPass2Sampler"
    assert schema["input"]["required"]["audio_output"][1]["default"] == "preserve_first_pass"
    assert schema["input"]["required"]["temporal_strategy"][1]["default"] == "guarded_overlap_exp"
    assert schema["output_name"] == ["output", "denoised_output"]


def test_h16_3_audio_merge_maps_absolute_time_and_crossfades_overlap():
    source = torch.zeros(1, 32, 2, 10)
    first = torch.ones(1, 32, 2, 5)
    second = torch.full((1, 32, 2, 7), 2.0)
    segments = [(0, 0, 5, 5), (3, 3, 10, 10)]
    merged, placed = _merge_audio_segments(source, segments, [first, second], 1.0)
    assert placed == 2
    assert torch.equal(merged[..., :3], torch.ones(1, 32, 2, 3))
    assert torch.allclose(merged[..., 3], torch.full((1, 32, 2), 1.0))
    assert torch.allclose(merged[..., 4], torch.full((1, 32, 2), 2.0))
    assert torch.allclose(merged[..., 5], torch.full((1, 32, 2), 2.0))
    assert torch.equal(merged[..., 6:], torch.full((1, 32, 2, 4), 2.0))


def test_h16_3_audio_merge_rejects_capture_count_or_shape_mismatch():
    source = torch.zeros(1, 32, 2, 10)
    segments = [(0, 0, 5, 5)]
    with pytest.raises(RuntimeError, match="do not match"):
        _merge_audio_segments(source, segments, [], 1.0)
    with pytest.raises(RuntimeError, match="refined audio shape"):
        _merge_audio_segments(source, segments, [torch.zeros(1, 32, 2, 4)], 1.0)


def test_h16_3_guider_conditioning_round_trip():
    guider = SimpleNamespace(
        original_conds={
            "positive": [{"cross_attn": torch.ones(1), "uuid": "p"}],
            "negative": [{"cross_attn": None, "uuid": "n"}],
        },
        cfg=1.25,
        model_patcher=object(),
    )
    positive, negative, cfg, model = _raw_conditioning_from_guider(guider)
    assert positive == [[positive[0][0], {"uuid": "p"}]]
    assert negative == [[None, {"uuid": "n"}]]
    assert cfg == 1.25 and model is guider.model_patcher


def test_h16_3_guider_requires_positive_and_model():
    with pytest.raises(ValueError, match="positive"):
        _raw_conditioning_from_guider(SimpleNamespace(original_conds={}))
    with pytest.raises(ValueError, match="ModelPatcher"):
        _raw_conditioning_from_guider(
            SimpleNamespace(original_conds={"positive": [[None, {}]]})
        )


def test_h16_3_formal_workflow_is_a_drop_in_pass2_replacement():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples/workflows/13-latent-upscale"
        / "2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    pass2 = nodes[19]
    assert pass2["type"] == "DeciiaChunkedPass2Sampler"
    assert [item["link"] for item in pass2["inputs"][:5]] == [31, 32, 33, 34, 35]
    assert pass2["widgets_values"] == [
        "guarded_overlap_exp",
        34,
        17,
        0.999,
        "refined_exp",
    ]
    assert pass2["outputs"][0]["links"] == [36]
    assert nodes[20]["inputs"][0]["link"] == 36
    assert nodes[12]["outputs"][1]["links"] == [16]
    assert nodes[13]["inputs"][0]["link"] == 16
    assert nodes[15]["inputs"][0]["link"] == 23
    assert nodes[17]["inputs"][1]["link"] == 30
    assert not any(
        node["type"] == "SamplerCustomAdvanced" and node["id"] == 19
        for node in workflow["nodes"]
    )
