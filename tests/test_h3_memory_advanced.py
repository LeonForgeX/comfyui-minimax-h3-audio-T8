from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType

import pytest
import torch

from comfy import ops
from comfy.ldm.minimax import model as core_h3
from comfy.ldm.minimax.model import DiTBlock
from comfy.ldm.modules import attention as core_attention
from comfy.patcher_extension import WrapperExecutor

import h3_audio_t8_pkg
from h3_audio_t8_pkg.h3_memory_advanced import (
    ATTACHMENT_KEY,
    ATTENTION_WRAPPER_KEY,
    FFN_WRAPPER_KEY,
    RUNTIME_TOKEN_KEY,
    configure_chunk_feed_forward,
    configure_low_vram_attention,
)
from h3_audio_t8_pkg.nodes_h3_memory_advanced import (
    MiniMaxH3ChunkFeedForwardT8Advanced,
    MiniMaxH3LowVRAMAttentionT8Advanced,
)
from test_prompt_relay_core_compat import model_fixture


def _small_model(block_count=2):
    model = model_fixture()
    model.model.diffusion_model.blocks = torch.nn.ModuleList(
        [
            DiTBlock(
                24,
                3,
                8,
                32,
                24,
                1e-6,
                1e-6,
                dtype=torch.float32,
                device="cpu",
                operations=ops.disable_weight_init,
            )
            for _ in range(block_count)
        ]
    )
    generator = torch.Generator().manual_seed(915)
    with torch.no_grad():
        for value in model.model.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
        for block in model.model.diffusion_model.blocks:
            block.attn.q_norm.weight.fill_(1.0)
            block.attn.k_norm.weight.fill_(1.0)
    return model


def _block_args(rows=17):
    return (
        torch.randn(rows, 24, generator=torch.Generator().manual_seed(916)),
        torch.randn(1, 24, generator=torch.Generator().manual_seed(917)),
        [(0, rows, 0)],
        None,
    )


def test_node_schemas_have_deliberately_small_public_surface():
    attention = MiniMaxH3LowVRAMAttentionT8Advanced.define_schema()
    ffn = MiniMaxH3ChunkFeedForwardT8Advanced.define_schema()
    assert attention.node_id == "MiniMaxH3LowVRAMAttentionT8Advanced"
    assert [item.id for item in attention.inputs] == ["model", "head_chunks"]
    attention_inputs = {item.id: item for item in attention.inputs}
    assert attention_inputs["head_chunks"].default == 4
    assert [item.id for item in ffn.inputs] == ["model", "chunks", "seq_threshold"]
    ffn_inputs = {item.id: item for item in ffn.inputs}
    assert ffn_inputs["chunks"].default == 2
    assert ffn_inputs["seq_threshold"].default == 4096
    assert [item.id for item in attention.outputs] == ["model", "report_json"]
    assert [item.id for item in ffn.outputs] == ["model", "report_json"]


def test_ffn_chunks_one_is_an_exact_object_identity_bypass():
    sentinel = object()
    returned, report = configure_chunk_feed_forward(sentinel, 1, 4096)
    assert returned is sentinel
    assert report["status"] == "identity"
    assert report["bit_exact_claim"] is True


def test_attention_head_chunks_one_is_still_an_active_early_release_patch():
    source = _small_model()
    patched, report = configure_low_vram_attention(source, 1)
    assert patched is not source
    assert report["status"] == "active"
    assert report["head_chunks"] == 1
    assert len(patched.object_patches) == 4
    assert source.object_patches == {}


def test_two_nodes_compose_in_either_order_without_mutating_source():
    for order in ("attention_first", "ffn_first"):
        source = _small_model()
        if order == "attention_first":
            middle, _ = configure_low_vram_attention(source, 2)
            patched, _ = configure_chunk_feed_forward(middle, 3, 256)
        else:
            middle, _ = configure_chunk_feed_forward(source, 3, 256)
            patched, _ = configure_low_vram_attention(middle, 2)
        assert source.object_patches == {}
        assert len(middle.object_patches) in (2, 4)
        assert len(patched.object_patches) == 6
        receipt = patched.get_attachment(ATTACHMENT_KEY)
        assert receipt.attention is not None and receipt.ffn is not None
        tokens = patched.model_options["transformer_options"][RUNTIME_TOKEN_KEY]
        assert tokens["attention"] is receipt.attention.token
        assert tokens["ffn"] is receipt.ffn.token


@pytest.mark.parametrize("head_chunks", [1, 2, 3, 56])
@pytest.mark.parametrize("use_ffn", [False, True])
def test_patched_block_matches_native_h3_equation_on_cpu(
    monkeypatch, head_chunks, use_ffn
):
    monkeypatch.setattr(core_h3, "optimized_attention", core_attention.attention_pytorch)
    source = _small_model(1)
    block = source.model.diffusion_model.blocks[0]
    x, t_emb, segments, rope = _block_args(19)
    expected = block(
        x.clone(),
        t_emb,
        segments,
        rope,
        transformer_options={},
    )
    patched, _ = configure_low_vram_attention(source, head_chunks)
    if use_ffn:
        patched, _ = configure_chunk_feed_forward(patched, 3, 256)
    patched.patch_model(load_weights=False)
    try:
        options = patched.model_options["transformer_options"]
        actual = block(
            x.clone(),
            t_emb,
            segments,
            rope,
            transformer_options=options,
        )
    finally:
        patched.unpatch_model(unpatch_weights=False)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_attention_calls_one_backend_delegate_per_effective_group(monkeypatch):
    source = _small_model(1)
    calls = []

    def delegate(func, q, k, v, heads, **kwargs):
        calls.append((heads, tuple(q.shape)))
        return core_attention.attention_pytorch(
            q, k, v, heads, **{**kwargs, "_inside_attn_wrapper": True}
        )

    source.model_options["transformer_options"]["optimized_attention_override"] = delegate
    patched, report = configure_low_vram_attention(source, 2)
    block = patched.model.diffusion_model.blocks[0]
    patched.patch_model(load_weights=False)
    try:
        x = torch.randn(11, 24)
        result = block.attn(
            x,
            transformer_options=patched.model_options["transformer_options"],
        )
    finally:
        patched.unpatch_model(unpatch_weights=False)
    assert result.shape == x.shape
    assert [item[0] for item in calls] == [2, 1]
    assert "callable_global_override" in report["attention_backend"]


def test_ffn_threshold_is_inclusive_and_chunks_only_above_it(monkeypatch):
    source = _small_model(1)
    patched, _ = configure_chunk_feed_forward(source, 3, 256)
    mlp = patched.model.diffusion_model.blocks[0].mlp
    original = comfy_linear = ops.linear_input_act
    rows = []

    def counted(*args, **kwargs):
        rows.append(int(args[1].shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "linear_input_act", counted)
    patched.patch_model(load_weights=False)
    try:
        mlp(torch.randn(256, 24))
        assert rows == [256]
        rows.clear()
        mlp(torch.randn(257, 24))
        assert rows == [86, 86, 85]
    finally:
        patched.unpatch_model(unpatch_weights=False)
        monkeypatch.setattr(ops, "linear_input_act", comfy_linear)


@pytest.mark.parametrize("kind", ["attention", "ffn"])
def test_runtime_guard_rejects_replaced_forward_before_diffusion(kind):
    source = _small_model(1)
    if kind == "attention":
        patched, _ = configure_low_vram_attention(source, 2)
        wrapper_key = ATTENTION_WRAPPER_KEY
        owner = patched.model.diffusion_model.blocks[0].attn
    else:
        patched, _ = configure_chunk_feed_forward(source, 2, 256)
        wrapper_key = FFN_WRAPPER_KEY
        owner = patched.model.diffusion_model.blocks[0].mlp
    patched.patch_model(load_weights=False)
    try:
        owner.forward = MethodType(lambda self, x, **kwargs: x, owner)
        wrappers = patched.get_wrappers("diffusion_model", wrapper_key)
        executor = WrapperExecutor.new_executor(
            lambda *args, **kwargs: pytest.fail("must reject before diffusion"),
            wrappers,
        )
        with pytest.raises(RuntimeError, match="replaced after binding"):
            executor.execute(
                None,
                None,
                None,
                patched.model_options["transformer_options"],
            )
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_duplicate_and_foreign_forward_owners_are_rejected():
    attention, _ = configure_low_vram_attention(_small_model(1), 2)
    with pytest.raises(RuntimeError, match="already installed"):
        configure_low_vram_attention(attention, 2)
    ffn, _ = configure_chunk_feed_forward(_small_model(1), 2, 256)
    with pytest.raises(RuntimeError, match="already installed"):
        configure_chunk_feed_forward(ffn, 2, 256)

    foreign = _small_model(1)
    foreign.object_patches["diffusion_model.blocks.0.mlp.forward"] = lambda x: x
    with pytest.raises(RuntimeError, match="refuse existing"):
        configure_chunk_feed_forward(foreign, 2, 256)


def test_dit_replacement_is_rejected_but_weight_lora_metadata_is_preserved():
    source = _small_model(1)
    source.patches["diffusion_model.blocks.0.mlp.fc1.weight"] = [(1.0, (object(),))]
    patched, _ = configure_chunk_feed_forward(source, 2, 256)
    assert patched.patches == source.patches
    assert patched.patches is not source.patches
    source.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): object()}
    }
    with pytest.raises(RuntimeError, match="DiT block replacement"):
        configure_chunk_feed_forward(source, 2, 256)


def test_registration_is_append_only_and_features_match():
    ids = [
        node.define_schema().node_id
        for node in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ]
    feature_ids = json.loads(
        (Path(__file__).resolve().parents[1] / "features.json").read_text(
            encoding="utf-8"
        )
    )["nodes"]
    assert len(ids) == 336
    assert ids == feature_ids
    assert ids[-2:] == [
        "MiniMaxH3LowVRAMAttentionT8Advanced",
        "MiniMaxH3ChunkFeedForwardT8Advanced",
    ]


def test_report_json_is_deterministic_and_does_not_serialize_runtime_tokens():
    source = _small_model(1)
    output = MiniMaxH3LowVRAMAttentionT8Advanced.execute(source, 4)
    report = json.loads(output.result[1])
    assert report["head_chunks"] == 4
    assert "token" not in output.result[1].lower()
    output = MiniMaxH3ChunkFeedForwardT8Advanced.execute(source, 1, 4096)
    assert output.result[0] is source
    assert json.loads(output.result[1])["status"] == "identity"
