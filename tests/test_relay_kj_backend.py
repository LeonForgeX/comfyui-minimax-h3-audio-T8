"""No GPU/model loading. Optional live KJ definitions use CPU kernel doubles.

These tests prove composition/routing, not CUDA execution or speed. Upstream
definitions are read locally and compiled without starting the KJ plugin.
"""
import ast
import inspect
import logging
from pathlib import Path
import sys
from types import ModuleType, FunctionType

import pytest
import torch
from comfy.ldm.modules import attention
from comfy.patcher_extension import WrapperExecutor

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import relay_kj_backend as adapter
from test_prompt_relay_core_compat import bound_layout, model_fixture, compose_eav


@pytest.fixture
def kj(monkeypatch):
    core_root = Path(inspect.getsourcefile(attention)).parents[3]
    source = core_root / "custom_nodes/ComfyUI-KJNodes/nodes/model_optimization_nodes.py"
    if not source.is_file():
        pytest.skip("optional actual KJ source not installed")
    tree = ast.parse(source.read_bytes(), filename=str(source))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in {"get_sage_func", "PathchSageAttentionKJ"}]
    module = ModuleType("test_relay_actual_kj_definitions")
    module.__file__ = str(source)
    module.__dict__.update(torch=torch, wrap_attn=attention.wrap_attn,
                           attention_pytorch=attention.attention_pytorch, logging=logging)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(tree, str(source), "exec", dont_inherit=True), vars(module))
    # Python 3.13 uses module-wide imported-name knowledge in LOAD_GLOBAL /
    # LOAD_ATTR optimizations. Use the whole-module compiler's exact function
    # code without executing the module (which would start unrelated nodes).
    full_codes = tuple(adapter._codes(compile(source.read_bytes(), str(source), "exec", dont_inherit=True)))
    factory_code = next(code for code in full_codes if code.co_name == "get_sage_func")
    module.get_sage_func = FunctionType(factory_code, vars(module), argdefs=(False,))

    def fake_kernel(q, k, v, *, tensor_layout="HND", attn_mask=None, **kwargs):
        if tensor_layout == "NHD":
            q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return result.transpose(1, 2) if tensor_layout == "NHD" else result

    package = ModuleType("sageattention")
    for name in ("sageattn", "sageattn_qk_int8_pv_fp16_cuda",
                 "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"):
        setattr(package, name, fake_kernel)
    monkeypatch.setitem(sys.modules, "sageattention", package)
    return module, package


def selector(kj, mode="auto", compile=False):
    model, = kj[0].PathchSageAttentionKJ().patch(model_fixture(), mode, compile)
    return model, model.model_options["transformer_options"]["optimized_attention_override"]


@pytest.mark.parametrize("mode", ["auto", "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++"])
@pytest.mark.parametrize("compile", [False, True])
def test_actual_kj_selector_authentication_without_plugin_start_or_gpu(kj, mode, compile):
    model, override = selector(kj, mode, compile)
    # The older native-only recognizer rejects the actual KJ closure.
    from h3_audio_t8_pkg.h3_core_compat import plain_attention_backend
    assert plain_attention_backend(override) is None
    backend = adapter.capture_kj_relay_backend(override)
    assert backend is not None
    assert backend.mode == mode
    assert not backend.mask_supported  # A double is not authenticated CUDA evidence.
    assert backend.report()["completed_calls"] == {}
    assert relay._assert_core_contract(model)


@pytest.mark.parametrize("mutation", ["kernel", "factory", "globals", "raw_attention"])
def test_changed_loaded_kj_contract_is_not_accepted(kj, monkeypatch, mutation):
    _, override = selector(kj)
    assert adapter.capture_kj_relay_backend(override) is not None
    if mutation == "kernel":
        monkeypatch.setattr(kj[1], "sageattn", lambda *a, **k: None)
    elif mutation == "factory":
        monkeypatch.setattr(kj[0], "get_sage_func", lambda *a, **k: None)
    elif mutation == "globals":
        monkeypatch.setattr(kj[0], "attention_pytorch", lambda *a, **k: None)
    else:
        wrapped = adapter._cells(override)["new_attention"]
        monkeypatch.setattr(wrapped, "__wrapped__", lambda *a, **k: None)
    assert adapter.capture_kj_relay_backend(override) is None


def test_unknown_override_is_not_invoked():
    def foreign(*args, **kwargs):
        pytest.fail("inspection must never execute a foreign override")
    assert adapter.capture_kj_relay_backend(foreign) is None


@pytest.mark.parametrize("query_route", ["video_only_paper", "joint_av_exp"])
def test_relay_retains_kj_delegate_and_bias_through_real_owner(kj, monkeypatch, query_route):
    source, original = selector(kj)
    binding, layout = bound_layout(query_route)
    model, _ = relay.patch_prompt_relay_model(source, binding, 32)
    backend = relay.prompt_relay_model_contract(model)["attention_backend"]
    assert backend.mode == "auto"
    assert source.model_options["transformer_options"]["optimized_attention_override"] is original
    generator = torch.Generator().manual_seed(718)
    q, k, v = [torch.randn(1, 2, layout.seq_len, 8, generator=generator) for _ in range(3)]
    route = relay._runtime_route(layout, binding, q.device)
    bias = torch.zeros(layout.seq_len, layout.seq_len)
    for segment in route["query_segments"]:
        # Independent reference equation, not the bias builder under test.
        for event in binding["events"]:
            outside = ((segment["query_times"] - event["midpoint"]).abs() - event["window"]).clamp_min(0)
            bias[segment["start"]:segment["end"], event["text_key_start"]:event["text_key_end"]] = (
                -.5 * (outside / event["sigma"]).square())[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = expected.transpose(1, 2).reshape(1, layout.seq_len, 16)

    def forbidden_global(*args, **kwargs):
        pytest.fail("selected KJ delegate must not be silently replaced by global attention")
    monkeypatch.setattr(attention, "optimized_attention", forbidden_global)

    def body(x, timestep, context, options, **kwargs):
        return attention.attention_pytorch(q, k, v, 2, skip_reshape=True, transformer_options=options)

    for _ in range(2):
        options = model.model_options["transformer_options"]
        executor = WrapperExecutor.new_executor(body, model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY))
        actual = executor.execute([q], None, None, options, minimax_payload={"layout": layout},
                                  **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding["binding_hash"]})
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
    assert backend.report()["completed_calls"]["pytorch:non_cuda"] > 0
    assert not any(name.startswith("sage:") for name in backend.counters)


def test_two_model_clones_have_independent_backend_audits(kj):
    source, _ = selector(kj)
    binding, _ = bound_layout("video_only_paper")
    first, _ = relay.patch_prompt_relay_model(source, binding, 32)
    second, _ = relay.patch_prompt_relay_model(source, binding, 32)
    a = relay.prompt_relay_model_contract(first)["attention_backend"]
    b = relay.prompt_relay_model_contract(second)["attention_backend"]
    assert a is not b
    a.counters["fixture"] += 1
    assert b.counters == {}


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_eav_composer_captures_same_authenticated_kj_backend(kj, mode):
    source, _ = selector(kj)
    binding, _ = bound_layout("video_only_paper")
    model, _ = relay.patch_prompt_relay_model(source, binding, 32)
    backend = relay.prompt_relay_model_contract(model)["attention_backend"]
    combined, _, _ = compose_eav(model, mode)
    override = combined.model_options["transformer_options"]["optimized_attention_override"]
    # Follow the actual Core setter and composer closures, not an attachment tag.
    router = adapter._cells(override)["optimized_attention"]
    assert adapter._cells(router)["relay"]["attention_backend"] is backend


def test_cuda_mode_never_forwards_bias_to_an_ignoring_kernel():
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported mask must not reach the kernel")
    backend = adapter.KJRelayBackend("auto", forbidden, "fixture")
    q = torch.randn(1, 2, 3, 8)
    with pytest.raises(RuntimeError, match="cannot consume Relay bias"):
        backend._selected_attention(q, q, q, 2, mask=torch.zeros(3, 3), skip_reshape=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("skip_reshape", [True, False])
def test_mask_aware_cpu_double_preserves_layout_dtype_and_float_bias(kj, dtype, skip_reshape):
    # Direct adapter mechanics only; this does NOT grant the double kernel audit.
    backend = adapter.KJRelayBackend("sageattn_qk_int8_pv_fp16_triton", kj[1].sageattn, "fixture", True)
    q = torch.randn(1, 2, 3, 8).to(dtype)
    k, v = [torch.randn(1, 2, 7, 8).to(dtype) for _ in range(2)]
    bias = torch.randn(3, 7).to(dtype)
    inputs = [q, k, v] if skip_reshape else [x.transpose(1, 2).reshape(1, -1, 16) for x in (q, k, v)]
    actual = backend._selected_attention(*inputs, 2, mask=bias, skip_reshape=skip_reshape)
    compute_dtype = torch.float16 if dtype == torch.float32 else dtype
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype), attn_mask=bias.to(compute_dtype))
    expected = expected.to(dtype).transpose(1, 2).reshape(1, 3, 16)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype


def test_kernel_failure_does_not_record_success(kj):
    backend = adapter.KJRelayBackend("auto", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kernel failed")), "fixture")
    q = torch.zeros(1, 2, 3, 8)
    with pytest.raises(RuntimeError, match="kernel failed"):
        backend._selected_attention(q, q, q, 2, skip_reshape=True)
    assert backend.counters == {}
