"""Actual installed KJ definitions + tiny native blocks; CPU-only mechanics."""
import ast
import importlib.util
import inspect
import logging
from pathlib import Path
import sys
from types import FunctionType, MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import comfy.model_management as mm
import comfy.quant_ops
from comfy import ops
from comfy.ldm.minimax.model import DiTBlock, MiniMaxH3Model
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrapperExecutor
from comfy_api.latest import io

from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import relay_kj_memory as memory
from h3_audio_t8_pkg.relay_kj_backend import _codes
from test_prompt_relay_core_compat import bound_layout, model_fixture, compose_eav
from test_relay_kj_backend import kj  # noqa: F401 -- actual selector/CPU Sage fixture


@pytest.fixture
def memory_nodes(monkeypatch, request):
    request.getfixturevalue("kj")
    root = Path(inspect.getsourcefile(attention)).parents[3] / "custom_nodes/ComfyUI-KJNodes/nodes"
    lowmem_path = root / "minimax_nodes.py"
    sage_path = root / "ltxv_nodes.py"
    if not lowmem_path.is_file() or not sage_path.is_file():
        pytest.skip("actual KJ memory source not installed")
    spec = importlib.util.spec_from_file_location("test_relay_kj_lowmem", lowmem_path)
    lowmem = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, lowmem)
    spec.loader.exec_module(lowmem)
    source = sage_path.read_bytes()
    compiled = tuple(_codes(compile(source, str(sage_path), "exec", dont_inherit=True)))
    sage = ModuleType("test_relay_kj_memory_sage")
    sage.__file__ = str(sage_path)
    sage.__dict__.update(torch=torch, mm=mm, _ck=comfy.quant_ops.ck, io=io,
                         logging=logging, _cuda_archs=("cpu_fixture",), _MiniMaxH3Model=MiniMaxH3Model)
    monkeypatch.setitem(sys.modules, sage.__name__, sage)
    code = next(item for item in compiled if item.co_name == "minimax_sageattn_forward")
    sage.minimax_sageattn_forward = FunctionType(code, vars(sage), argdefs=(None, {}))
    declaration = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                       and node.name == "MiniMaxH3MemoryEfficientSageAttentionPatch")
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(sage_path), "exec", dont_inherit=True), vars(sage))
    calls = []

    def cpu_sage(qkv, dtype):
        calls.append("original_direct_sage")
        q, k, v = qkv
        qkv.clear()
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).to(dtype)

    sage._sageattn_int8_fp8_nhd = cpu_sage
    return lowmem, sage, calls


def small_model():
    model = model_fixture()
    model.model.diffusion_model.blocks = torch.nn.ModuleList([
        DiTBlock(24, 3, 8, 32, 24, 1e-6, 1e-6, dtype=torch.float32,
                 device="cpu", operations=ops.disable_weight_init) for _ in range(2)])
    generator = torch.Generator().manual_seed(115)
    with torch.no_grad():
        for value in model.model.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * .1)
        for block in model.model.diffusion_model.blocks:
            block.attn.q_norm.weight.fill_(1.)
            block.attn.k_norm.weight.fill_(1.)
    return model


def patched_source(memory_nodes, kind="sage_lowmem_ffn", groups=2):
    lowmem, sage, _ = memory_nodes
    model = small_model()
    if "sage" in kind:
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
    if "lowmem" in kind:
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, groups).result[0]
    if "ffn" in kind:
        model = lowmem.MiniMaxChunkFeedForward.execute(model, 3, 256).result[0]
    return model


def apply_methods(model):
    saved = {}
    for path, method in model.object_patches.items():
        if path.startswith("diffusion_model.blocks."):
            owner = method.__self__
            saved[path] = (owner, owner.forward)
            owner.forward = method
    return saved


def restore_methods(saved):
    for owner, method in saved.values():
        owner.forward = method


def test_actual_memory_sage_bypasses_global_owner_before_adapter(memory_nodes, monkeypatch):
    model = patched_source(memory_nodes, "sage")
    method = model.object_patches["diffusion_model.blocks.0.attn.forward"]
    def forbidden(*args, **kwargs):
        pytest.fail("original direct Sage does not visit the override")
    x = torch.randn(11, 24)
    monkeypatch.setattr(attention, "optimized_attention", forbidden)
    method(x, transformer_options={"optimized_attention_override": forbidden})
    assert memory_nodes[2] == ["original_direct_sage"]


@pytest.mark.parametrize("kind", ["sage", "sage_lowmem", "sage_lowmem_ffn", "lowmem", "lowmem_ffn", "ffn"])
def test_complete_actual_kj_sets_are_recognized_without_weights(memory_nodes, kind):
    source = patched_source(memory_nodes, kind)
    info = memory.inspect_memory_composition(source)
    assert info is not None
    assert info["backend"] is None
    assert relay._assert_core_contract(source)
    binding, _ = bound_layout("video_only_paper")
    patched, _ = relay.patch_prompt_relay_model(source, binding, 32)
    observed = memory.inspect_memory_composition(patched)
    assert set(observed["methods"]) == set(info["methods"])
    assert source.object_patches == info["methods"]
    if kind != "ffn":
        assert isinstance(observed["backend"], memory.HeadGroupedBackend)
    compose_eav(patched, "apply_exp")


@pytest.mark.parametrize("kind", ["sage", "sage_lowmem_ffn", "lowmem_ffn"])
@pytest.mark.parametrize("groups", [1, 2, 3, 56])
def test_real_projection_and_relay_match_dense_equation(memory_nodes, monkeypatch, kind, groups):
    monkeypatch.setattr(attention, "optimized_attention", attention.attention_pytorch)
    # The installed lowmem module's binding must track this explicit CPU delegate.
    monkeypatch.setattr(memory_nodes[0], "optimized_attention", attention.attention_pytorch)
    source = patched_source(memory_nodes, kind, groups)
    binding, layout = bound_layout("joint_av_exp")
    patched, _ = relay.patch_prompt_relay_model(source, binding, 32)
    saved = apply_methods(patched)
    attn = patched.model.diffusion_model.blocks[0].attn
    x = torch.randn(layout.seq_len, 24, generator=torch.Generator().manual_seed(514))
    q, k, v = attn.qkv_proj(x).split(24, -1)
    q = attn.q_norm(q.reshape(1, layout.seq_len, 3, 8)).transpose(1, 2)
    k = attn.k_norm(k.reshape(1, layout.seq_len, 3, 8)).transpose(1, 2)
    v = v.reshape(1, layout.seq_len, 3, 8).transpose(1, 2)
    runtime_route = relay._runtime_route(layout, binding, x.device)
    bias = torch.zeros(layout.seq_len, layout.seq_len)
    for segment in runtime_route["query_segments"]:
        for event in binding["events"]:
            distance = ((segment["query_times"] - event["midpoint"]).abs() - event["window"]).clamp_min(0)
            bias[segment["start"]:segment["end"], event["text_key_start"]:event["text_key_end"]] = (
                -.5 * (distance / event["sigma"]).square())[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = attn.out_proj(expected.transpose(1, 2).reshape(layout.seq_len, 24))

    def body(packed, timestep, context, options, **kwargs):
        # A single-item list is the actual KJ block's early-release protocol.
        handed_off = [packed[0]]
        result = attn(handed_off, transformer_options=options)
        assert handed_off == []
        return result

    try:
        wrappers = patched.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
        options = patched.model_options["transformer_options"]
        actual = WrapperExecutor.new_executor(body, wrappers).execute(
            [x], None, None, options, minimax_payload={"layout": layout},
            **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding["binding_hash"]})
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
        assert memory_nodes[2] == []  # No bypass of Relay through KJ's raw helper.
    finally:
        restore_methods(saved)


@pytest.mark.parametrize("mutation", ["partial", "foreign", "wrong_owner", "changed_groups"])
def test_partial_or_changed_memory_compositions_are_rejected(memory_nodes, mutation):
    source = patched_source(memory_nodes)
    binding, _ = bound_layout("video_only_paper")
    if mutation == "changed_groups":
        source, _ = relay.patch_prompt_relay_model(source, binding, 32)
        source.model_options["transformer_options"]["minimax_head_chunks"] = 3
    elif mutation == "partial":
        source.object_patches.pop("diffusion_model.blocks.1.attn.forward")
    elif mutation == "foreign":
        owner = source.model.diffusion_model.blocks[0].attn
        source.object_patches["diffusion_model.blocks.0.attn.forward"] = MethodType(lambda self, x: x, owner)
    else:
        source.object_patches["diffusion_model.blocks.1.attn.forward"] = source.object_patches["diffusion_model.blocks.0.attn.forward"]
    with pytest.raises((ValueError, RuntimeError)):
        memory.inspect_memory_composition(source)


def test_downstream_forward_replacement_is_detected_before_attention(memory_nodes):
    source = patched_source(memory_nodes)
    binding, _ = bound_layout("video_only_paper")
    patched, _ = relay.patch_prompt_relay_model(source, binding, 32)
    backend = relay.prompt_relay_model_contract(patched)["attention_backend"]
    saved = apply_methods(patched)
    try:
        memory.bind_memory_runtime(backend, {})
        owner = patched.model.diffusion_model.blocks[0].attn
        owner.forward = source.object_patches["diffusion_model.blocks.0.attn.forward"]
        with pytest.raises(RuntimeError, match="replaced after Relay binding"):
            memory.bind_memory_runtime(backend, {})
    finally:
        restore_methods(saved)


def test_grouping_leaves_full_head_eav_gain_and_audio_rows_unchanged(monkeypatch):
    calls = []
    contract = {"kind": "cpu_fixture", "head_chunks": 2, "ffn_settings": None, "source_sha256s": []}
    grouped = memory.HeadGroupedBackend(memory._NativeDelegate(), 2, contract)
    monkeypatch.setattr(attention, "optimized_attention", attention.attention_pytorch)
    binding, layout = bound_layout("joint_av_exp")
    q, k, v = [torch.randn(1, 3, layout.seq_len, 8) for _ in range(3)]
    saved = [value.clone() for value in (q, k, v)]
    route = relay._runtime_route(layout, binding, q.device)
    options = {relay.PROMPT_RELAY_RUNTIME_KEY: route,
               eav.EAV_RUNTIME_KEY: {"seq_len": layout.seq_len, "active": True,
                 "video_start": route["video_start"], "video_end": route["video_end"],
                 "frames": 3, "spatial_tokens": 1, "max_workspace_mib": 32, "tau": 4,
                 "g_hard_limit": 10, "runtime": SimpleNamespace(record=lambda *a, **kw: calls.append(kw)),
                 "forward_index": 0, "mode": "apply_exp"}}
    baseline = relay.route_prompt_relay_attention(q, k, v, 3, skip_reshape=True,
                  transformer_options=options, query_chunk_rows=32, relay_backend=grouped)
    actual = eav.route_eav_prompt_relay_attention(q, k, v, 3, skip_reshape=True,
                  transformer_options=options, query_chunk_rows=32, relay_backend=grouped)
    assert len(calls) == 1
    torch.testing.assert_close(actual[:, :route["video_start"]], baseline[:, :route["video_start"]])
    torch.testing.assert_close(actual[:, route["video_start"]:], baseline[:, route["video_start"]:] * calls[0]["g"])
    for before, after in zip(saved, (q, k, v)):
        torch.testing.assert_close(before, after)


def test_eav_only_memory_runtime_cleans_on_failure_then_separate_bind_works(memory_nodes, monkeypatch):
    monkeypatch.setattr(attention, "optimized_attention", attention.attention_pytorch)
    monkeypatch.setattr(memory_nodes[0], "optimized_attention", attention.attention_pytorch)
    source = patched_source(memory_nodes)
    _, layout = bound_layout("joint_av_exp")
    sigmas = torch.cat([torch.linspace(1., .05, 20), torch.zeros(1)])
    config = dict(mode="apply_exp", tau=4., start_video_progress=0., end_video_progress=1.,
                  max_workspace_mib=32, g_hard_limit=3.)
    latent = [torch.zeros(1, 24, 3, 2, 2), torch.zeros(1, 32, 2)]
    for fail in (True, False):
        patched, runtime, _ = eav.build_eav_model(source, sigmas, **config)
        saved = apply_methods(patched)
        options = patched.model_options["transformer_options"]
        def body(x, timestep, context, options, **kwargs):
            assert memory.MEMORY_TOKEN_KEY in options[eav.EAV_RUNTIME_KEY]
            if fail:
                raise RuntimeError("simulated cancelled generation")
            return patched.model.diffusion_model.blocks[0].attn([torch.randn(layout.seq_len, 24)],
                                                               transformer_options=options)
        try:
            executor = WrapperExecutor.new_executor(body, patched.get_wrappers("diffusion_model", eav.EAV_WRAPPER_KEY))
            args = (latent, torch.tensor([500.]), torch.zeros(1, 4, 24), options)
            if fail:
                with pytest.raises(RuntimeError, match="simulated cancelled"):
                    executor.execute(*args, minimax_payload={"layout": layout})
            else:
                result = executor.execute(*args, minimax_payload={"layout": layout})
                assert torch.isfinite(result).all()
                assert len(runtime._forwards[0]["g_values"]) == 1
                assert runtime.config["composed_attention_backend"]["completed_calls"]
            assert eav.EAV_RUNTIME_KEY not in options
            assert memory.MEMORY_TOKEN_KEY not in options
            assert memory.inspect_memory_composition(source)["backend"] is None
        finally:
            restore_methods(saved)
