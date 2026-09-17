from __future__ import annotations

import copy
import json

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as chunked
from h3_audio_t8_pkg.learned_latent_upscale_advanced import learned_upscale_geometry
from h3_audio_t8_pkg.nodes_chunked_two_pass_upscale_advanced import MiniMaxH3ChunkedTwoPassPlanT8Advanced as Node
from test_chunked_two_pass_parity import _plan, harness as _shared_harness


# Reuse the real executor fixture without marking an unused import as a fixture.
harness = _shared_harness


def _latent(width=448, height=224):
    return {"samples": chunked.comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 15, height // 16, width // 16), torch.zeros(1, 32, 2, 85))),
        "custom": {"keep": True}}


@pytest.mark.parametrize("width,height", [(448, 224), (736, 416), (512, 768)])
@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0, 4.0])
def test_scale_modes_use_exactly_the_ordinary_upscaler_math(width, height, scale):
    source = _latent(width, height)
    before = source["samples"].tensors
    plan = _plan(size_mode="scale_by", scale_by=scale, source_latent=source)
    expected = learned_upscale_geometry(width // 16, height // 16, "scale_by", scale, .70,
        64, 64, "preserve_source", 1.05)
    assert plan["geometry"] == expected
    assert (plan["target_width"], plan["target_height"]) == (expected["output_width"], expected["output_height"])
    assert plan["target_width"] % 32 == plan["target_height"] % 32 == 0
    assert source["samples"].tensors[0] is before[0] and source["samples"].tensors[1] is before[1]
    assert all(not t.count_nonzero() for t in before)


@pytest.mark.parametrize("mode,kwargs", [
    ("target_megapixels", {"target_megapixels": .4}),
    ("target_dimensions", {"target_width": 1152, "target_height": 640}),
    ("target_dimensions", {"target_width": 896, "target_height": 512, "aspect_policy": "honor_dimensions_exp", "max_anisotropy": 2.}),
])
def test_area_and_dimension_modes_match_ordinary_geometry(mode, kwargs):
    source = _latent()
    plan = _plan(size_mode=mode, source_latent=source, **kwargs)
    expected = learned_upscale_geometry(28, 14, **plan["size_selection"])
    assert plan["geometry"] == expected


@pytest.mark.parametrize("scale", [0., .5, 4.01, float("nan"), float("inf")])
def test_bad_or_out_of_model_range_multipliers_rejected(scale):
    with pytest.raises(ValueError, match="scale_by"):
        _plan(size_mode="scale_by", scale_by=scale, source_latent=_latent())


@pytest.mark.parametrize("mode", ["scale_by", "target_megapixels"])
def test_source_required_for_automatic_modes(mode):
    with pytest.raises(ValueError, match="source_latent"):
        _plan(size_mode=mode)


def test_manual_legacy_plan_is_unchanged_without_source():
    old = _plan()
    explicit = _plan(size_mode="target_dimensions", scale_by=2., target_megapixels=.70,
        aspect_policy="preserve_source", max_anisotropy=1.05, source_latent=None)
    assert old == explicit
    assert "geometry" not in old and "size_selection" not in old
    assert old["target_width"] == old["target_height"] == 64
    # Outputs0/1 and all old required widget positions remain stable.
    schema = Node.GET_NODE_INFO_V1()
    assert schema["output_name"] == ["plan", "report_json", "width", "height"]
    assert list(schema["input"]["required"])[:3] == ["model_name", "target_width", "target_height"]
    assert schema["input"]["optional"]["size_mode"][1]["default"] == "target_dimensions"


def test_node_outputs_calculated_dimensions_and_json_without_latent_storage():
    from test_chunked_two_pass_parity import _report
    args = dict(model_name="mock.safetensors", target_width=64, target_height=64,
        temporal_chunk_frames=34, temporal_overlap_frames=17, anchor_strength=.999,
        tile_width=64, tile_height=64, spatial_overlap=0, spatial_fade=0, minimum_tile_size=32,
        overlap_blend="smoothstep", precision="fp16", release_policy="offload_after",
        sampling_contract="standard_joint_4plus4_exp", parity_report_json=json.dumps(_report()),
        size_mode="scale_by", scale_by=2., source_latent=_latent())
    plan, report, width, height = Node.execute(**args).result
    assert (width, height) == (896, 448)
    assert json.loads(report)["geometry"] == plan["geometry"]
    assert "source_latent" not in plan


def test_sizing_does_not_change_refine_counts_audio_or_overlap(harness):
    calls, run = harness
    source = _latent(32, 32)
    sized = _plan(size_mode="scale_by", scale_by=2., source_latent=source)
    old = _plan()
    assert {k: v for k, v in sized.items() if k not in {"geometry", "size_selection"}} == old
    output, report = run(plan=sized)
    assert calls["upscale"] == calls["noise"] == 1
    assert report["segment_count"] == 2 and report["total_model_calls_including_coarse_contract"] == 12
    v, a = output["samples"].tensors
    assert torch.equal(v, torch.full_like(v, 2)) and torch.equal(a, torch.full_like(a, 3))


@pytest.mark.parametrize("kind", ["source", "target", "receipt"])
def test_source_or_target_drift_rejected_before_any_inference(harness, kind):
    calls, run = harness
    plan = _plan(size_mode="scale_by", scale_by=2., source_latent=_latent(32, 32))
    if kind == "source":
        plan = _plan(size_mode="scale_by", scale_by=1., source_latent=_latent(64, 64))
    elif kind == "target":
        plan["target_width"] = 96
    else:
        plan = copy.deepcopy(plan)
        plan["geometry"]["scale_x"] = 100.
    with pytest.raises(ValueError, match="source LATENT or computed target"):
        run(plan=plan)
    assert calls["upscale"] == calls["noise"] == 0 and calls["sample"] == []


def test_dimension_stretch_and_shrink_rejected_using_actual_source():
    with pytest.raises(ValueError, match="non-shrinking"):
        _plan(source_latent=_latent(), target_width=64, target_height=64)
    with pytest.raises(ValueError, match="anisotropic"):
        _plan(source_latent=_latent(), target_width=896, target_height=640, aspect_policy="honor_dimensions_exp")
