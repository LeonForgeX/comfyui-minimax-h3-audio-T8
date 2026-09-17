"""CPU owner/protocol checks; no kernel dispatch or GPU qualification claim."""
from __future__ import annotations

import builtins
from itertools import permutations
from types import MethodType

import pytest
import torch

from comfy import ops
from comfy.ldm.minimax.model import DiTBlock
from comfy.patcher_extension import CallbacksMP

from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg import h3_memory_advanced as memory
from test_h3_memory_advanced import _block_args, _small_model


def _gated_model():
    model = _small_model(1)
    block = DiTBlock(
        384, 3, 128, 64, 24, 1e-6, 1e-6, gate_compress=True,
        dtype=torch.bfloat16, device="cpu", operations=ops.disable_weight_init,
    )
    model.model.diffusion_model.blocks = torch.nn.ModuleList([block])
    generator = torch.Generator().manual_seed(916)
    with torch.no_grad():
        for value in block.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
        block.attn.q_norm.weight.fill_(1.0)
        block.attn.k_norm.weight.fill_(1.0)
    return model


def _v2(model, profile="trained_vsa_exp"):
    # Real Core SparseAttnPatch / native gate / ModelPatcher ownership, while
    # keeping sampler initialization and actual kernels out of this unit scope.
    return v2._install_runtime(model, profile, 12288)


@pytest.mark.parametrize("order", list(permutations(("v2", "ffn", "attention"))))
def test_real_native_v2_memory_nodes_compose_in_all_orders(order):
    source = _gated_model()
    gate = source.model.diffusion_model.blocks[0].attn.to_gate_compress
    model = source
    v2_input = None
    for step in order:
        if step == "v2":
            model = _v2(model)
            v2_input = model
        elif step == "ffn":
            model, _ = memory.configure_chunk_feed_forward(model, 2, 256)
        else:
            model, _ = memory.configure_low_vram_attention(model, 2)
    receipt = v2.capture_fast_h3_v2_owner(model)
    assert receipt is not None
    assert receipt.profile == "trained_vsa_exp"
    assert receipt.runtime.head_chunks == 2
    composed = memory.inspect_t8_memory_composition(model, allowed_wrapper_keys=(v2.KEY,))
    assert composed["head_chunks"] == 2
    assert composed["ffn_settings"] == [2, 256]
    assert model.model.diffusion_model.blocks[0].attn.to_gate_compress is gate
    assert source.object_patches == {}
    assert source.get_attachment(v2.KEY) is None
    assert source.model_options["transformer_options"].get("patches_replace", {}) == {}
    assert v2_input is not None
    # Refresh binds a new branch-local runtime without invalidating its input.
    previous = v2.capture_fast_h3_v2_owner(v2_input)
    if order[-1] != "v2":
        assert receipt.runtime is not previous.runtime
    callbacks = model.callbacks
    assert callbacks[CallbacksMP.ON_PREPARE_STATE][v2.KEY] == [receipt.runtime.prepare]
    assert callbacks[CallbacksMP.ON_CLEANUP][v2.KEY] == [receipt.runtime.cleanup]
    assert model.get_wrappers("diffusion_model", v2.KEY) == [receipt.runtime.guard]
    assert model.object_patches["diffusion_model.blocks.0.mlp.forward"] is (
        model.get_attachment(memory.ATTACHMENT_KEY).ffn.methods[0]
    )


@pytest.mark.parametrize("configure", [
    lambda model: memory.configure_chunk_feed_forward(model, 2, 256),
    lambda model: memory.configure_low_vram_attention(model, 2),
])
def test_spoofed_v2_attachment_does_not_allow_dit_or_memory(configure):
    model = _gated_model()
    model.set_attachments(v2.KEY, {"schema": v2.SCHEMA, "profile": "trained_vsa_exp"})
    model.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): lambda *a: None}
    }
    with pytest.raises(RuntimeError, match="foreign or malformed"):
        configure(model)
    assert model.object_patches == {}


@pytest.mark.parametrize("mutation", ["dit", "extra_dit", "override", "prepare", "cleanup", "wrapper"])
@pytest.mark.parametrize("kind", ["ffn", "attention"])
def test_v2_bridge_preserves_user_owners_but_rejects_damaged_private_receipts(mutation, kind, caplog):
    model = _v2(_gated_model())
    options = model.model_options["transformer_options"]
    def foreign(*a, **k):
        return None
    if mutation == "dit":
        options["patches_replace"]["dit"][("double_block", 0)] = foreign
    elif mutation == "extra_dit":
        options["patches_replace"]["dit"][("double_block", 999)] = foreign
    elif mutation == "override":
        options["optimized_attention_override"] = foreign
    elif mutation in ("prepare", "cleanup"):
        role = (CallbacksMP.ON_PREPARE_STATE if mutation == "prepare" else CallbacksMP.ON_CLEANUP)
        model.callbacks[role][v2.KEY] = [foreign]
    else:
        model.wrappers["diffusion_model"][v2.KEY] = [foreign]
    configure = (lambda value: memory.configure_chunk_feed_forward(value, 2, 256)) if kind == "ffn" else (
        lambda value: memory.configure_low_vram_attention(value, 2))
    if mutation in ("dit", "extra_dit", "override"):
        before_dit = dict(options["patches_replace"]["dit"])
        before_override = options.get("optimized_attention_override")
        patched, _ = configure(model)
        live = patched.model_options["transformer_options"]
        assert set(live["patches_replace"]["dit"]) == set(before_dit)
        assert all(live["patches_replace"]["dit"][key] is value for key, value in before_dit.items())
        assert live.get("optimized_attention_override") is before_override
        assert v2.capture_fast_h3_v2_owner(patched).runtime is v2.capture_fast_h3_v2_owner(model).runtime
        assert "refresh bypassed" in caplog.text
    else:
        with pytest.raises(RuntimeError, match="owner was replaced"):
            configure(model)
    assert model.object_patches == {}


def test_dense_v2_memory_preserves_unknown_dit(caplog):
    model = _v2(_gated_model(), "dense_compat_exp")
    model.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): lambda *a: None}
    }
    previous = model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    patched, _ = memory.configure_chunk_feed_forward(model, 2, 256)
    assert patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is previous
    assert "advisory" in caplog.text


def test_unmarked_native_sparse_dit_is_preserved_without_receipt_authentication(caplog):
    model = _v2(_gated_model())
    model.remove_attachments(v2.KEY)
    previous = model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    patched, _ = memory.configure_chunk_feed_forward(model, 2, 256)
    assert patched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is previous
    assert patched.get_attachment(v2.KEY) is None
    assert "preserved/delegated" in caplog.text


@pytest.mark.parametrize("order", [("ffn", "attention"), ("attention", "ffn")])
def test_preexisting_dit_composer_survives_memory_refresh_and_executes(order, caplog):
    source = _gated_model()
    calls = []
    def previous(args, extra):
        calls.append("previous")
        return extra["original_block"](args)
    source.set_model_patch_replace(previous, "dit", "double_block", 0)
    model = _v2(source)
    hook = model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    for kind in order:
        model, _ = (memory.configure_chunk_feed_forward(model, 2, 256) if kind == "ffn" else
                    memory.configure_low_vram_attention(model, 2))
        assert model.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is hook
    sentinel = object()
    result = hook({"transformer_options": model.model_options["transformer_options"],
                   "img": torch.zeros(19, 24), "rope_freqs": None},
                  {"original_block": lambda args: sentinel})
    assert result is sentinel and calls == ["previous"]
    assert "refresh bypassed" in caplog.text


def test_memory_identity_ffn_one_does_not_authenticate_or_modify_v2():
    model = _v2(_gated_model())
    model.model_options["transformer_options"]["patches_replace"]["dit"] = {}
    result, report = memory.configure_chunk_feed_forward(model, 1, 256)
    assert result is model
    assert report["status"] == "identity"


def test_ordinary_memory_path_does_not_import_v2(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if "fast_h3_v2_advanced" in name:
            pytest.fail("unmarked ordinary MODEL must not import the V2 module")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    model, _ = memory.configure_chunk_feed_forward(_small_model(1), 2, 256)
    model, _ = memory.configure_low_vram_attention(model, 2)
    assert model.get_attachment(v2.KEY) is None


def test_injected_block_attention_receives_tensor_and_skips_low_vram_qkv(monkeypatch):
    source = _small_model(1)
    block = source.model.diffusion_model.blocks[0]
    x, t_emb, segments, rope = _block_args(19)
    calls = []

    def attention(hidden, rope_freqs=None, transformer_options=None):
        assert torch.is_tensor(hidden)
        calls.append((hidden.shape, rope_freqs, transformer_options))
        return hidden * 0.25

    options = {"proof": object()}
    expected = block(x.clone(), t_emb, segments, rope, options, attention=attention)
    calls.clear()
    patched, _ = memory.configure_low_vram_attention(source, 2)
    monkeypatch.setattr(memory, "_attention_impl", lambda *a, **k: pytest.fail("must use injected producer"))
    patched.patch_model(load_weights=False)
    try:
        actual = block(x.clone(), t_emb, segments, rope, options, attention=attention)
    finally:
        patched.unpatch_model(unpatch_weights=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [(torch.Size([19, 24]), rope, options)]


def test_none_injected_attention_keeps_old_private_consumer_guard():
    source = _small_model(1)
    block = source.model.diffusion_model.blocks[0]
    patched, _ = memory.configure_low_vram_attention(source, 2)
    patched.patch_model(load_weights=False)
    try:
        block.attn.forward = MethodType(
            lambda self, value, **kwargs: torch.zeros_like(value[0]), block.attn
        )
        with pytest.raises(RuntimeError, match="did not consume"):
            block(*_block_args(19), transformer_options={}, attention=None)
    finally:
        patched.unpatch_model(unpatch_weights=False)
