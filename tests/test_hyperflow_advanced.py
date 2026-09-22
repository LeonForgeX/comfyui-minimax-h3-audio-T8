from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
import comfy.ops
from comfy.ldm.minimax import model as native_h3
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher
from torch import nn

from h3_audio_t8_pkg.hyperflow_runtime_advanced import (
    ATTACHMENT_KEY,
    HyperFlowBinding,
    HyperFlowStep,
    active_step,
    install_hyperflow,
    pop_step,
    push_step,
)
from h3_audio_t8_pkg.hyperflow_sampling_advanced import (
    build_hyperflow_plan,
    sample_hyperflow_continuous,
    shifted_grid,
)
from h3_audio_t8_pkg.hyperflow_weights_advanced import (
    DEFAULT_RAW_GRID,
    load_hyperflow_original,
    parse_metadata,
    HyperFlowWeights,
)


def _metadata():
    return {
        "hyperflow": "true",
        "hyperflow_version": "1.0",
        "hyperflow_gate": "0.25",
        "lora_rank": "2",
        "lora_alpha": "2",
        "base_model": "MiniMaxAI/MiniMax-H3",
        "hyperflow_sigmas": json.dumps(DEFAULT_RAW_GRID),
        "hyperflow_video_shift": "12.0",
        "hyperflow_audio_shift": "3.0",
    }


def _synthetic_source():
    state = {}

    def add(module, out, width, *, fill=1.0):
        name = "transformer." + module
        state[name + ".lora_A.weight"] = torch.full((2, width), fill, dtype=torch.float32)
        state[name + ".lora_B.weight"] = torch.full((out, 2), fill, dtype=torch.float32)

    for family, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
        for index in range(count):
            prefix = f"{family}.{index}"
            for part in ("q", "k", "v"):
                add(prefix + f".attn.to_{part}", 3, 4, fill={"q": 1, "k": 2, "v": 3}[part])
            add(prefix + ".attn.to_out.0", 4, 3)
            add(prefix + ".ff.net.0.proj", 8, 4)
            add(prefix + ".ff.net.2", 4, 4)
    for base in ("time_embedder", "endpoint_time_embedder"):
        add(base + ".linear_1", 4, 4)
        add(base + ".linear_2", 4, 4)
    assert len(state) == 632
    return state


def test_full_original_maps_every_tensor_and_preserves_two_time_endpoint(tmp_path):
    path = tmp_path / "original.safetensors"
    save_file(_synthetic_source(), str(path), metadata=_metadata())
    converted = load_hyperflow_original(path)
    assert converted.source_tensor_count == 632
    assert converted.fused_qkv_groups == 52
    assert len(converted.patches) == 210
    assert set(converted.endpoint) == {"proj_in", "proj_out"}
    assert len(converted.source_sha256) == 64
    a, b, alpha = converted.patches["blocks.0.attn.qkv_proj"]
    assert a.shape == (6, 4)
    assert b.shape == (9, 6)
    assert alpha == 6.0  # rank 3r, so alpha must also be 3x
    assert torch.count_nonzero(b[:3, 2:]) == 0
    assert torch.count_nonzero(b[3:6, :2]) == 0


def test_original_loader_rejects_missing_endpoint_and_nonfinite(tmp_path):
    state = _synthetic_source()
    state.pop("transformer.endpoint_time_embedder.linear_2.lora_B.weight")
    missing = tmp_path / "missing.safetensors"
    save_file(state, str(missing), metadata=_metadata())
    with pytest.raises(ValueError, match="Incomplete HyperFlow"):
        load_hyperflow_original(missing)
    state = _synthetic_source()
    state["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"][0, 0] = float("nan")
    bad = tmp_path / "nan.safetensors"
    save_file(state, str(bad), metadata=_metadata())
    with pytest.raises(ValueError, match="Non-finite"):
        load_hyperflow_original(bad)


def test_header_rejects_wrong_shift_and_grid():
    data = _metadata()
    data["hyperflow_audio_shift"] = "NaN"
    with pytest.raises(ValueError, match="audio shift"):
        parse_metadata(data)
    data = _metadata()
    data["hyperflow_sigmas"] = "[1, 0.5, 0]"
    with pytest.raises(ValueError, match="nine"):
        parse_metadata(data)


class _Model:
    def __init__(self):
        self.model = object()
        self.binding = HyperFlowBinding(
            owner="unit-owner", sha256="a" * 64, version="1.0", gate=0.25,
            video_shift=12.0, audio_shift=3.0,
            raw_sigmas=DEFAULT_RAW_GRID, model_identity=id(self.model),
        )

    def get_attachment(self, key):
        assert key == ATTACHMENT_KEY
        return self.binding


@pytest.mark.parametrize("split", [1, 4, 7])
def test_continuous_av_split_is_exact_without_renoise(split):
    model = _Model()
    full = build_hyperflow_plan(model)
    first = build_hyperflow_plan(model, 0, split)
    second = build_hyperflow_plan(model, split, 8)
    starts = []

    def network(x, sigma, **_kwargs):
        step = active_step()
        assert step is not None
        derivative = torch.cat((
            torch.full_like(x[..., :2], 0.5 + step.absolute_index / 10),
            torch.full_like(x[..., 2:], 0.25 + step.absolute_index / 20),
        ), dim=-1)
        return x - sigma.reshape(-1, 1, 1) * derivative

    x0 = torch.zeros((1, 1, 4), dtype=torch.float32)
    def run(x, plan):
        starts.append(x.clone())
        return sample_hyperflow_continuous(
            network, x, plan, video_values=2, packed_values=4,
            audio_velocity_is_raw=True, disable=True,
        )

    entire = run(x0.clone(), full)
    middle = run(x0.clone(), first)
    split_output = run(middle, second)
    assert torch.equal(split_output, entire)
    assert torch.equal(starts[-1], middle)
    assert active_step() is None
    assert full.nfe == 8
    assert first.nfe + second.nfe == 8
    assert full.video_sigmas[split] == first.video_segment[-1] == second.video_segment[0]
    assert full.audio_sigmas[split] == first.audio_segment[-1] == second.audio_segment[0]


def test_video_and_audio_midpoint_are_not_raw_half():
    video = shifted_grid(DEFAULT_RAW_GRID, 12.0)
    audio = shifted_grid(DEFAULT_RAW_GRID, 3.0)
    assert video[4].item() == pytest.approx(12 / 13)
    assert audio[4].item() == pytest.approx(0.75)


def test_sampler_rejects_plan_from_other_model():
    from h3_audio_t8_pkg.hyperflow_sampling_advanced import _match_plan

    model = _Model()
    plan = build_hyperflow_plan(model)
    model.binding = HyperFlowBinding(
        owner="another-owner", sha256="b" * 64, version="1.0", gate=0.25,
        video_shift=12.0, audio_shift=3.0,
        raw_sigmas=DEFAULT_RAW_GRID, model_identity=id(model.model),
    )
    with pytest.raises(ValueError, match="different adapter"):
        _match_plan(model, plan)


def test_continuation_state_and_new_noise_paths_are_mutually_exclusive():
    from h3_audio_t8_pkg.hyperflow_sampling_advanced import setup_hyperflow_sampler

    model = _Model()
    head = build_hyperflow_plan(model, 0, 4)
    tail = build_hyperflow_plan(model, 4, 8)
    with pytest.raises(ValueError, match="nonzero absolute restart"):
        setup_hyperflow_sampler(
            model, {}, head, internal_continuation=True,
            x_sigma_override=lambda: torch.zeros(1),
        )
    with pytest.raises(ValueError, match="cannot inject"):
        setup_hyperflow_sampler(
            model, {}, tail, internal_continuation=True,
            new_noise_restart=True, x_sigma_override=lambda: torch.zeros(1),
        )


def test_partial_upscale_nodes_require_their_absolute_head_and_tail_plans():
    from h3_audio_t8_pkg.nodes_hyperflow_advanced import (
        MiniMaxH3HyperFlowCoarseSamplerT8Advanced,
        MiniMaxH3HyperFlowHeadPlanT8Advanced,
        MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced,
    )

    model = _Model()
    head = build_hyperflow_plan(model, 0, 4)
    tail = build_hyperflow_plan(model, 4, 8)
    assert MiniMaxH3HyperFlowHeadPlanT8Advanced.define_schema().node_id.endswith("HeadPlanT8Advanced")
    assert MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced.define_schema().node_id.endswith("PartialRefineSamplerT8Advanced")
    with pytest.raises(ValueError, match="0:4 Head Plan"):
        MiniMaxH3HyperFlowCoarseSamplerT8Advanced.execute(model, {}, tail)
    with pytest.raises(ValueError, match="4:8 Tail Plan"):
        MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced.execute(model, {}, head)


def test_legacy_long_video_runner_rejects_hyperflow_owner_before_cache_or_sampling():
    from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8

    model = _Model()
    with pytest.raises(ValueError, match=r"cannot enter the legacy native4\+4 long-video runner"):
        MiniMaxH3DualModelLongVideoEXPT8.execute(
            model, model, 256, 256, "unused.safetensors", 4, 4, 12., 3., 12., 3.,
            "auto", 0.,
        )


def test_original_v1_metadata_rejects_changed_nine_point_grid():
    data = _metadata()
    changed = list(DEFAULT_RAW_GRID)
    changed[3] = changed[3] - 0.01
    data["hyperflow_sigmas"] = json.dumps(changed)
    with pytest.raises(ValueError, match="trained original"):
        parse_metadata(data)


@pytest.mark.skipif(not os.environ.get("T8_HYPERFLOW_REAL_WEIGHT"), reason="optional 2.8 GB original weight")
def test_real_original_weight_complete_numeric_conversion():
    path = Path(os.environ["T8_HYPERFLOW_REAL_WEIGHT"])
    converted = load_hyperflow_original(path)
    assert converted.source_sha256 == "9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447"
    assert converted.metadata.rank == 256
    assert converted.metadata.alpha == 256
    assert converted.metadata.gate == 0.25
    assert len(converted.patches) == 210
    assert len(converted.endpoint) == 2
    assert converted.fused_qkv_groups == 52


def _tiny_native_model(monkeypatch):
    monkeypatch.setattr(native_h3, "optimized_attention", attention.attention_pytorch)
    torch.manual_seed(27091)
    diffusion = native_h3.MiniMaxH3Model(
        hidden_size=8, num_layers=50, token_refiner_num_layers=2,
        num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
        text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, rope_inv_freq_len=1,
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init,
    )
    for parameter in diffusion.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    diffusion.rope.inv_freq.fill_(1.0)
    diffusion.requires_grad_(False)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = diffusion
            self.device = torch.device("cpu")

    base = Base()
    return ModelPatcher(base, torch.device("cpu"), torch.device("cpu")), diffusion


def _tiny_weights(diffusion):
    paths = []
    for group in (diffusion.blocks, diffusion.token_refiner.blocks):
        for index, block in enumerate(group):
            scope = "blocks" if group is diffusion.blocks else "token_refiner.blocks"
            for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                paths.append(f"{scope}.{index}.{suffix}")
    paths += ["time_embedder.proj_in", "time_embedder.proj_out"]
    modules = dict(diffusion.named_modules())
    patches = {}
    for path in paths:
        out, width = modules[path].weight.shape
        rank = 6 if path.endswith(".attn.qkv_proj") else 2
        patches[path] = (
            torch.full((rank, width), 0.01),
            torch.full((out, rank), 0.01),
            float(rank),
        )
    endpoint = {}
    for part in ("proj_in", "proj_out"):
        out, width = getattr(diffusion.time_embedder, part).weight.shape
        endpoint[part] = (torch.full((2, width), 0.01), torch.full((out, 2), 0.01), 2.0)
    return HyperFlowWeights(
        path=Path("tiny.safetensors"), source_sha256="c" * 64,
        metadata=parse_metadata(_metadata()), patches=patches,
        endpoint=endpoint, source_tensor_count=632, fused_qkv_groups=52,
    )


def test_dedicated_loader_rejects_preexisting_generic_hyperflow_attachment(monkeypatch):
    base, diffusion = _tiny_native_model(monkeypatch)
    base.set_attachments("t8_h3_lora_metadata", {"hyperflow": "true"})
    with pytest.raises(ValueError, match="content LoRA"):
        install_hyperflow(base, _tiny_weights(diffusion))
    assert base.get_attachment(ATTACHMENT_KEY) is None


def test_dedicated_loader_reports_preexisting_content_time_patches(monkeypatch):
    base, diffusion = _tiny_native_model(monkeypatch)
    target = "diffusion_model.time_embedder.proj_in.weight"
    weight = diffusion.time_embedder.proj_in.weight
    assert base.add_patches({target: ("diff", (torch.zeros_like(weight),))}, .1) == [target]
    _, _, report = install_hyperflow(base, _tiny_weights(diffusion))
    assert report["preexisting_content_time_patch_count"] == 1
    assert "base time branch only" in report["content_time_scope_warning"]


@pytest.mark.parametrize("masked", [False, True])
def test_real_core_tiny_owner_patch_and_forward_cleanup(monkeypatch, masked):
    base, diffusion = _tiny_native_model(monkeypatch)
    original_forward = diffusion.time_embedder.forward
    weights = _tiny_weights(diffusion)
    patched, binding, report = install_hyperflow(base, weights)
    assert report["applied_backbone_and_base_time_targets"] == 210
    assert patched.get_attachment(ATTACHMENT_KEY) == binding
    assert base.get_attachment(ATTACHMENT_KEY) is None
    patched.patch_model()
    try:
        assert getattr(diffusion.time_embedder.forward, "_t8_hyperflow_owner", None) == binding.owner
        sigma = shifted_grid(DEFAULT_RAW_GRID, 12.0)
        index = 0
        step = HyperFlowStep(binding.owner, index, float(sigma[index]), float(sigma[index + 1]))
        x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
        context = torch.randn(1, 2, 8)
        options = {
            "sample_sigmas": sigma,
            "wrappers": patched.wrappers,
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
        }
        payload = {"layout": native_h3.PackedLayout(2, 2, 4, 4, 3)}
        mask_kwargs = {}
        if masked:
            video_mask = torch.ones(1, 1, 2, 4, 4)
            video_mask[:, :, 0, :2] = 0.5
            video_mask[:, :, 1, :2] = 0.0
            mask_kwargs["denoise_mask"] = video_mask
            mask_kwargs["audio_denoise_mask"] = torch.tensor([[[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]]])
        token = push_step(step)
        try:
            with torch.no_grad():
                video, audio = diffusion(
                    x, torch.tensor([float(sigma[index]) * 1000]), context,
                    transformer_options=options, minimax_payload=payload,
                    **mask_kwargs,
                )
        finally:
            pop_step(token)
        assert video.shape == x[0].shape and audio.shape == x[1].shape
        assert bool(torch.isfinite(video).all()) and bool(torch.isfinite(audio).all())
        assert active_step() is None
        with pytest.raises(RuntimeError, match="typed sampler"):
            diffusion(x, torch.tensor([float(sigma[index]) * 1000]), context,
                      transformer_options=options, minimax_payload=payload, **mask_kwargs)
    finally:
        patched.unpatch_model()
    assert diffusion.time_embedder.forward == original_forward


def test_tiny_core_stage_a_b_a_and_owner_failure_cleanup(monkeypatch):
    base, diffusion = _tiny_native_model(monkeypatch)
    weights = _tiny_weights(diffusion)
    stage_a, binding_a, _ = install_hyperflow(base, weights)
    stage_b, binding_b, _ = install_hyperflow(base, weights)
    sigma = shifted_grid(DEFAULT_RAW_GRID, 12.0)
    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
    context = torch.randn(1, 2, 8)
    payload = {"layout": native_h3.PackedLayout(2, 2, 4, 4, 3)}
    original = diffusion.time_embedder.forward

    def one_stage(patched, binding, *, wrong_owner=False):
        patched.patch_model()
        try:
            owner = "wrong-owner" if wrong_owner else binding.owner
            token = push_step(HyperFlowStep(owner, 0, float(sigma[0]), float(sigma[1])))
            try:
                with torch.no_grad():
                    return diffusion(
                        x, torch.tensor([float(sigma[0]) * 1000]), context,
                        transformer_options={
                            "sample_sigmas": sigma, "wrappers": patched.wrappers,
                            "minimax_h3_sigma_shift_video": 12.0,
                            "minimax_h3_sigma_shift_audio": 3.0,
                        },
                        minimax_payload=payload,
                    )
            finally:
                pop_step(token)
        finally:
            patched.unpatch_model()
            assert diffusion.time_embedder.forward == original
            assert active_step() is None

    first = one_stage(stage_a, binding_a)
    second = one_stage(stage_b, binding_b)
    again = one_stage(stage_a, binding_a)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    with pytest.raises(RuntimeError, match="typed sampler"):
        one_stage(stage_b, binding_b, wrong_owner=True)
    final = one_stage(stage_a, binding_a)
    assert all(torch.equal(a, b) for a, b in zip(first, final))
