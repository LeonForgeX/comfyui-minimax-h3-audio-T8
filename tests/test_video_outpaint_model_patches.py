"""Optional installed-KJ checks use tiny native blocks, never diffusion weights."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import subprocess
from functools import lru_cache
from types import MethodType

import pytest
import torch
import folder_paths
from comfy import ops
from comfy.ldm.minimax.model import DiTBlock

from h3_audio_t8_pkg.video_outpaint_identity import native_stock_model_identity
from h3_audio_t8_pkg.video_outpaint_model_patches import KJ_SOURCE_SHA256, inspect_outpaint_model_patches
from h3_audio_t8_pkg.video_outpaint_guidance import build_outpaint_guidance
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_regional import patch_outpaint_regional_model
from h3_audio_t8_pkg.video_outpaint_regional_conditioning import prepare_outpaint_regional_conditioning
from test_video_outpaint_execution import _model


KJ_TEST_SOURCES = {
    "legacy_crlf": ("3f20054214fec9f9234fd3841ae6f1e4287948f6", True,
                    "c371576b1bb31a2f518bdb4ceda43cb10b20338f0c9d68f99ed1be76ce06478f"),
    "legacy_lf": ("3f20054214fec9f9234fd3841ae6f1e4287948f6", False,
                  "6f3df10c042677270053d75223db3e05ede1106be20fc178220531b73dc868fa"),
    "current_lf": ("da90cca", False,
                   "c67d638cd1f060ff4dd53dc3843aab77b980ea783b24ca4271300cbe040cb91b"),
}
KJ_INSTALLED_SHA = "acbfdd2c25ebec34b1ade23d4856931209a9e1d5b690b810f2cef0af47832642"


@lru_cache(maxsize=3)
def _local_kj_revision(root, revision):
    try:
        return subprocess.check_output(["git", "-C", root, "show",
                                        f"{revision}:nodes/minimax_nodes.py"],
                                       stderr=subprocess.PIPE, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


@pytest.fixture(params=["installed", *KJ_TEST_SOURCES])
def kj(monkeypatch, request, tmp_path):
    path = Path(folder_paths.__file__).resolve().parent / "custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py"
    if request.param == "installed":
        if not path.exists():
            pytest.skip("optional pinned KJ implementation is not installed")
        assert hashlib.sha256(path.read_bytes()).hexdigest() in {KJ_SOURCE_SHA256, KJ_INSTALLED_SHA}
    else:
        revision, crlf, expected_sha = KJ_TEST_SOURCES[request.param]
        blob = _local_kj_revision(str(path.parents[1]), revision)
        if blob is None:
            pytest.skip("optional local KJ Git revision unavailable; no download")
        if crlf:
            blob = blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        assert hashlib.sha256(blob).hexdigest() == expected_sha
        path = tmp_path / "minimax_nodes.py"
        path.write_bytes(blob)
    spec = importlib.util.spec_from_file_location("outpaint_test_kj_memory", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _native_blocks():
    model = _model()
    model.model.diffusion_model = torch.nn.Module()
    model.model.diffusion_model.blocks = torch.nn.ModuleList([
        DiTBlock(8, 2, 4, 16, 8, 1e-6, 1e-6, dtype=torch.float32,
                 device="cpu", operations=ops.disable_weight_init) for _ in range(2)])
    with torch.no_grad():
        for tensor in model.model.parameters():
            tensor.fill_(0.1)
    return model


def _pair(kj, *, heads=2, chunks=4):
    lowmem = kj.MiniMaxLowVRAMAttention.execute(_native_blocks(), heads).result[0]
    return kj.MiniMaxChunkFeedForward.execute(lowmem, chunks, 4096).result[0]


class _Clip:
    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, prompt):
        count = len(prompt.split()) + 1
        return [[torch.zeros(1, count, 8), {
            "minimax_token_tags": torch.ones(count, dtype=torch.long), "pooled_output": None}]]


def _regional_provider(tmp_path):
    plan = build_outpaint_plan(
        source_sha256="d" * 64, width=320, height=240, frame_count=22,
        aspect="custom", left=64, top=64, right=64, bottom=64,
        generation_megapixels=0.5)
    guidance = build_outpaint_guidance(
        plan, [{"shot": 0, "region": "top", "prompt": "sky"}], [])
    return prepare_outpaint_regional_conditioning(_Clip(), "subject", guidance, plan, tmp_path)


def test_pinned_kj_pair_identity_tracks_settings_and_applied_restored_state(kj):
    model = _pair(kj)
    identity = native_stock_model_identity(model)
    assert identity == native_stock_model_identity(_pair(kj))
    assert identity != native_stock_model_identity(_pair(kj, heads=1))
    assert identity != native_stock_model_identity(_pair(kj, chunks=2))
    # Exercise the same binding states Comfy's patch/unpatch lifecycle creates.
    for path, method in model.object_patches.items():
        owner = model.model.get_submodule(path[:-len(".forward")])
        owner.forward = method
    assert identity == native_stock_model_identity(model)
    for path in model.object_patches:
        owner = model.model.get_submodule(path[:-len(".forward")])
        owner.forward = MethodType(type(owner).forward, owner)
    assert identity == native_stock_model_identity(model)


@pytest.mark.parametrize("kind", ["ffn", "attention"])
def test_each_complete_memory_patch_can_be_used_independently(kj, kind):
    model = _native_blocks()
    if kind == "ffn":
        model = kj.MiniMaxChunkFeedForward.execute(model, 4, 4096).result[0]
    else:
        model = kj.MiniMaxLowVRAMAttention.execute(model, 4).result[0]
    assert native_stock_model_identity(model)["tensor_count"] > 0


def test_regional_router_composes_with_pinned_kj_pair_and_identity(kj, tmp_path):
    model, contract = patch_outpaint_regional_model(_pair(kj), _regional_provider(tmp_path), 128)
    identity = native_stock_model_identity(model)
    assert identity["composition"]["kind"] == "regional_outpaint_composition"
    assert identity["composition"]["regional"] == contract
    assert identity["composition"]["memory"]["kind"] == "pinned_kj_memory"
    model.model_options["transformer_options"]["unknown"] = True
    with pytest.raises(ValueError, match="memory"):
        inspect_outpaint_model_patches(model, regional_contract=contract)
    assert native_stock_model_identity(model)["portable_cache_reuse"] is False


@pytest.mark.parametrize("corruption", ["missing", "wrong_owner", "closure", "options", "live_code", "extra"])
def test_unknown_or_mutated_composition_is_not_a_verified_kj_pair(kj, corruption, monkeypatch):
    model = _pair(kj)
    key = "diffusion_model.blocks.0.mlp.forward"
    if corruption == "missing":
        del model.object_patches[key]
    elif corruption == "wrong_owner":
        model.object_patches[key] = model.object_patches["diffusion_model.blocks.1.mlp.forward"]
    elif corruption == "closure":
        model.object_patches[key].__func__.__closure__[0].cell_contents.num_chunks = 8
    elif corruption == "options":
        model.model_options["transformer_options"]["unverified_attention"] = True
    elif corruption == "live_code":
        monkeypatch.setattr(kj, "minimax_mlp_chunked_forward", lambda *args: None)
    else:
        model.object_patches["diffusion_model.unknown"] = True
    with pytest.raises(ValueError, match="KJ|memory"):
        inspect_outpaint_model_patches(model)
    assert native_stock_model_identity(model)["portable_cache_reuse"] is False


def test_memory_forwards_match_native_tiny_cpu_math(kj, monkeypatch):
    from comfy.ldm.modules.attention import attention_pytorch
    # Force both tiny oracle calls onto CPU SDPA; no claim about INT8/CUDA kernels.
    monkeypatch.setattr(kj, "optimized_attention", attention_pytorch)
    block = _native_blocks().model.diffusion_model.blocks[0]
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(300, 8, generator=generator)
    time = torch.randn(1, 8, generator=generator)
    segments = [(0, 100, 0), (100, 200, 1), (200, 300, 2)]
    from comfy.ldm.minimax import model as native
    monkeypatch.setattr(native, "optimized_attention", attention_pytorch)
    with torch.no_grad():
        expected = block(inputs.clone(), time, segments, None)
        block.attn.forward = MethodType(kj.minimax_attn_lowmem_forward, block.attn)
        block.mlp.forward = kj.MiniMaxFFNChunkPatch(4, 256).__get__(block.mlp)
        actual = kj.minimax_block_lowmem_forward(block, inputs.clone(), time, segments, None,
                                                {"minimax_head_chunks": 2})
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_callback_api_is_explicit_and_keeps_tensor_ownership(kj, monkeypatch):
    import inspect
    from comfy.ldm.modules.attention import attention_pytorch
    from comfy.ldm.minimax import model as native
    monkeypatch.setattr(kj, "optimized_attention", attention_pytorch)
    monkeypatch.setattr(native, "optimized_attention", attention_pytorch)
    block = _native_blocks().model.diffusion_model.blocks[0]
    generator = torch.Generator().manual_seed(78)
    x = torch.randn(30, 8, generator=generator)
    time = torch.randn(1, 8, generator=generator)
    segments = [(0, 30, 0)]
    if "attention" not in inspect.signature(kj.minimax_block_lowmem_forward).parameters:
        with pytest.raises(TypeError, match="attention"):
            kj.minimax_block_lowmem_forward(block, x, time, segments, None, attention=lambda *a: None)
        return
    options = {"minimax_head_chunks": 2}
    native_input = x.clone()
    seen = []
    def callback(h, *, rope_freqs, transformer_options):
        assert isinstance(h, torch.Tensor) and transformer_options is options
        seen.append(h.shape)
        return block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
    with torch.no_grad():
        expected = block(native_input, time, segments, None, transformer_options=options, attention=callback)
        seen.clear()
        actual = kj.minimax_block_lowmem_forward(block, x, time, segments, None, options, attention=callback)
    assert seen == [torch.Size([30, 8])]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x, native_input, atol=0, rtol=0)


def test_callback_error_does_not_mutate_options_or_next_call(kj, monkeypatch):
    import inspect
    from comfy.ldm.modules.attention import attention_pytorch
    monkeypatch.setattr(kj, "optimized_attention", attention_pytorch)
    if "attention" not in inspect.signature(kj.minimax_block_lowmem_forward).parameters:
        return  # Historical API explicitly covered by the previous test.
    block = _native_blocks().model.diffusion_model.blocks[0]
    block.attn.forward = MethodType(kj.minimax_attn_lowmem_forward, block.attn)
    x, time, segments, options = torch.ones(30, 8), torch.ones(1, 8), [(0, 30, 0)], {"minimax_head_chunks": 2}
    with torch.no_grad():
        expected = kj.minimax_block_lowmem_forward(block, x.clone(), time, segments, None, options)
        def fail(*args, **kwargs):
            raise RuntimeError("callback failure")
        with pytest.raises(RuntimeError, match="callback failure"):
            kj.minimax_block_lowmem_forward(block, x.clone(), time, segments, None, options, attention=fail)
        actual = kj.minimax_block_lowmem_forward(block, x.clone(), time, segments, None, options)
    assert options == {"minimax_head_chunks": 2}
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_exact_source_identity_and_probe_policy_agree(kj, tmp_path):
    import inspect
    from h3_audio_t8_pkg.video_outpaint_kj_contract import kj_source_contract
    from tools.outpaint_probe_cases import is_audited_kj_source
    payload = Path(kj.__file__).read_bytes()
    contract = kj_source_contract(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert contract["source_sha256"] == digest
    assert contract["attention_callback"] == ("attention" in inspect.signature(kj.minimax_block_lowmem_forward).parameters)
    assert native_stock_model_identity(_pair(kj))["composition"]["source_sha256"] == digest
    assert is_audited_kj_source(kj.__file__)
    modified = tmp_path / "unverified_source.py"
    modified.write_bytes(payload + b"\n# local change\n")
    assert not is_audited_kj_source(modified)
    assert not is_audited_kj_source(tmp_path / "absent.py")
    with pytest.raises(ValueError, match="compatibility audit"):
        kj_source_contract(modified.read_bytes())
    with pytest.raises(ValueError, match="bytes"):
        kj_source_contract(payload.decode())


@pytest.mark.parametrize("name", ["minimax_attn_lowmem_forward", "minimax_block_lowmem_forward",
                                 "minimax_mlp_chunked_forward", "wrapped_forward"])
@pytest.mark.parametrize("mutation", ["positional", "keyword"])
def test_live_default_mutation_rejected_and_restoration_recovers(kj, monkeypatch, name, mutation):
    model = _pair(kj)
    original = native_stock_model_identity(model)
    fn = (model.object_patches["diffusion_model.blocks.0.mlp.forward"].__func__
          if name == "wrapped_forward" else getattr(kj, name))
    with monkeypatch.context() as patch:
        if mutation == "keyword":
            patch.setattr(fn, "__kwdefaults__", {"attention": lambda *args: None})
        else:
            values = list(fn.__defaults__ or (None,))
            values[-1] = lambda *args: None
            patch.setattr(fn, "__defaults__", tuple(values))
        with pytest.raises(ValueError, match="live defaults"):
            inspect_outpaint_model_patches(model)
        assert native_stock_model_identity(model)["portable_cache_reuse"] is False
    assert native_stock_model_identity(model) == original


@pytest.mark.parametrize("name", ["minimax_attn_lowmem_forward", "minimax_block_lowmem_forward"])
@pytest.mark.parametrize("replacement", [{"optimized_attention_override": "injected"}, None])
def test_live_options_default_requires_empty_plain_dict(kj, monkeypatch, name, replacement):
    model = _pair(kj)
    fn = getattr(kj, name)
    values = list(fn.__defaults__)
    index = next(index for index, value in enumerate(values) if type(value) is dict)
    values[index] = replacement
    monkeypatch.setattr(fn, "__defaults__", tuple(values))
    with pytest.raises(ValueError, match="live defaults"):
        inspect_outpaint_model_patches(model)
    assert native_stock_model_identity(model)["portable_cache_reuse"] is False


def test_default_validation_rejects_dict_subclass_without_comparing_it(kj, monkeypatch):
    class EmptyLookingDict(dict):
        def __bool__(self):
            raise AssertionError("untrusted default must not be called")
    model = _pair(kj)
    fn = kj.minimax_attn_lowmem_forward
    monkeypatch.setattr(fn, "__defaults__", (None, EmptyLookingDict()))
    with pytest.raises(ValueError, match="live defaults"):
        inspect_outpaint_model_patches(model)
    assert native_stock_model_identity(model)["portable_cache_reuse"] is False
