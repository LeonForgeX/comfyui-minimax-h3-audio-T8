"""Real old/new Core hook controls; no substituted attention dispatcher or tensors."""

import json
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import comfy.ops
from comfy.ldm.minimax import model as minimax
from comfy.ldm.modules import attention as core_attention
from comfy.model_management import InterruptProcessingException

from h3_audio_t8_pkg import attention_hooks_advanced as hooks
from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend


def sparse_module():
    return pytest.importorskip('comfy_extras.nodes_sparse_attention')


@pytest.mark.parametrize('coarse', [False, True])
def test_sparse_hook_vsa_preserves_padding_gate_and_live_lengths(monkeypatch, coarse):
    # CPU contract spy; actual CUDA numerical checks live in the GPU evidence.
    sparse = sparse_module()
    patch = sparse.SparseAttnPatch(tau=1.3, topk_ratio=.5, vsa=True,
                                  sigma_start=1., sigma_end=0., min_tokens=0,
                                  dense_blocks=set(), sink_conditioning='exact_kv', extra_tokens=0,
                                  verbose=False)
    layout = minimax.PackedLayout(7, 5, 14, 10, 3)
    n, heads, dim = layout.seq_len, 2, 128
    q = torch.randn(1, heads, n, dim)
    x = torch.randn(n, heads * dim)
    layer = SimpleNamespace(heads=heads, head_dim=dim,
                            to_gate_compress=(lambda value: value * .25) if coarse else None)
    seen = {}

    def kernel(q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v, **kwargs)
        return v

    monkeypatch.setattr(sparse.ck, 'sol_attn', kernel)
    actual = hooks._sparse_hook_attention(layer, x, q, q, q,
                                          {'minimax_h3_layout': layout}, (patch, sparse))
    plan = patch.vsa_plan(layout, x.device)
    assert plan['n'] > n
    torch.testing.assert_close(actual, q.transpose(1, 2).reshape(1, n, -1), rtol=0, atol=0)
    torch.testing.assert_close(seen['q'][:, plan['inv']], q.transpose(1, 2), rtol=0, atol=0)
    assert not torch.count_nonzero(seen['q'][:, plan['src'] < 0])
    assert seen['tail'] is False and seen['block_len'] is plan['block_len']
    assert seen['sink_blocks'] == seen['sink_q'] == [0, plan['n_prefix']]
    assert seen['topk_ratio'] == .5 and seen['tau'] == 1.3 and seen['token_aug'] == 0
    if coarse:
        torch.testing.assert_close(seen['coarse_gate'][:, plan['inv']],
                                   (x * .25).view(1, n, heads, dim), rtol=0, atol=0)
    else:
        assert 'coarse_gate' not in seen


@pytest.mark.parametrize('changed', ['tokens', 'heads', 'dtype'])
def test_sparse_hook_rejects_layout_changing_qkv_without_dense_fallback(changed):
    layer = SimpleNamespace(heads=2, head_dim=128)
    x = torch.randn(4, 256)
    q = torch.randn(1, 2, 4, 128)
    altered = {'tokens': q[:, :, :3], 'heads': q[:, :1], 'dtype': q.double()}[changed]
    with pytest.raises(RuntimeError, match='must retain QKV'):
        hooks._sparse_hook_attention(layer, x, altered, q, q, {}, (object(), object()))


def test_official_producer_attention_closure_authentication_rejects_forged_marker():
    sparse = sparse_module()
    patch = sparse.SparseAttnPatch(1.3, 0., False, 1., 0., 0, set(), 'off', 0, False)
    block = SimpleNamespace(attn=object())
    replacement = sparse.make_h3_block_patch(block, 3, patch)
    attention = inspect.getclosurevars(replacement).nonlocals['attention']
    state = hooks.native_sparse_state(attention, 'attention')
    assert state['patch'] is patch and state['block'] is block and state['block_index'] == 3

    def forged(*args, **kwargs):
        return None

    forged.__module__ = attention.__module__
    forged.__name__ = attention.__name__
    forged.__qualname__ = attention.__qualname__
    assert hooks.native_sparse_state(forged, 'attention') is None


def test_block_wrapper_keeps_unknown_attention_and_options_untouched():
    seen = []

    class Block:
        def forward(self, x, transformer_options=None, attention=None):
            seen.append((x, transformer_options, attention))
            return attention(x)

    block = Block()
    wrapped = hooks._make_sparse_hook_block_forward(block, 0, 1)
    options = {'patches': {'attn1_patch': [object()]}, 'user_key': 7}
    def original(value):
        return value + 2
    assert wrapped(3, transformer_options=options, attention=original) == 5
    assert seen == [(3, options, original)]
    assert 'block_index' not in options


def test_no_hooks_preserves_original_producer_callable_without_interception(monkeypatch):
    def forbidden(*args):
        raise AssertionError('No-hook calls must not inspect or replace the producer')

    monkeypatch.setattr(hooks, 'native_sparse_state', forbidden)

    class Block:
        def forward(self, x, transformer_options=None, attention=None):
            return attention(x)

    assert hooks._make_sparse_hook_block_forward(Block(), 0, 1)(
        3, transformer_options={}, attention=lambda value: value + 5) == 8


def attention_fixture():
    torch.manual_seed(5907)
    layer = minimax.Attention(8, 2, 4, 1e-6, dtype=torch.float32,
                             device="cpu", operations=comfy.ops.disable_weight_init)
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=.15)
    layer.requires_grad_(False)
    return layer, torch.randn(5, 8)


def options():
    holder = SimpleNamespace(model_options={"transformer_options": {"user_key": "retained"}})
    set_h3_attention_backend(holder, core_attention.attention_pytorch)
    return holder.model_options["transformer_options"]


@pytest.mark.parametrize("rope", [False, True], ids=["no_rope", "partial_rope"])
def test_no_hook_matches_real_native_core_without_dispatcher_substitution(rope):
    layer, x = attention_fixture()
    opts = options()
    rotation = minimax.rope_rotation_table(torch.randn(5, 2), torch.float32) if rope else None
    with torch.no_grad():
        expected = layer(x.clone(), rope_freqs=rotation, transformer_options=opts)
        actual = hooks._hooked_attention_forward(
            layer, x.clone(), rotation, opts, block_index=1, block_type="double", total_blocks=50)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    assert opts["user_key"] == "retained" and "block_index" not in opts


@pytest.mark.parametrize("mapping", [False, True], ids=["tuple_hook", "mapping_hook"])
def test_qkv_and_output_hooks_match_independent_dense_attention(mapping):
    layer, x = attention_fixture()
    opts = options()
    seen = []

    def qkv_hook(q, k, v, extra_options):
        seen.append(("qkv", extra_options["block_index"], extra_options["n_heads"], q.shape))
        q, k, v = q + .125, k * .75, v - .2
        return {"q": q, "k": k, "v": v} if mapping else (q, k, v)

    def output_hook(output, extra_options):
        seen.append(("output", extra_options["block_type"], extra_options["total_blocks"]))
        return output + .3

    opts["patches"] = {"attn1_patch": [qkv_hook], "attn1_output_patch": [output_hook]}
    with torch.no_grad():
        q, k, v = layer.qkv_proj(x).split(8, dim=-1)
        q = layer.q_norm(q.reshape(5, 2, 4)) + .125
        k = layer.k_norm(k.reshape(5, 2, 4)) * .75
        v = v.reshape(5, 2, 4) - .2
        tensors = [value.transpose(0, 1).unsqueeze(0) for value in (q, k, v)]
        dense = F.scaled_dot_product_attention(*tensors).transpose(1, 2).reshape(5, 8)
        expected = layer.out_proj(dense + .3)
        actual = hooks._hooked_attention_forward(
            layer, x.clone(), None, opts, block_index=7, block_type="double", total_blocks=50)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    assert seen == [("qkv", 7, 2, torch.Size([1, 5, 8])), ("output", "double", 50)]
    assert opts["patches"]["attn1_patch"] == [qkv_hook]
    assert opts["user_key"] == "retained" and "block_index" not in opts


def test_real_core_capability_report_does_not_invent_hook_support(record_property):
    layer, _ = attention_fixture()
    diffusion = minimax.MiniMaxH3Model.__new__(minimax.MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    block = torch.nn.Module()
    block.attn = layer
    diffusion.blocks = torch.nn.ModuleList([block])
    report = hooks.probe_native_attention_hooks(diffusion)
    record_property("native_hook_capabilities", json.dumps(report, sort_keys=True))
    assert report["main_block_count"] == 1
    assert report["available"] == (report["attention_hooks"] and report["block_metadata"])


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, InterruptProcessingException])
def test_hook_abort_leaves_input_and_backend_usable_for_retry(error):
    layer, x = attention_fixture()
    original = x.clone()
    opts = options()
    installed = opts["optimized_attention_override"]
    state = {"calls": 0}

    def abort_once(q, k, v, extra_options):
        state["calls"] += 1
        if state["calls"] == 1:
            raise error("controlled hook abort")
        return q, k, v

    opts["patches"] = {"attn1_patch": [abort_once]}
    with torch.no_grad():
        expected = layer(x.clone(), transformer_options={"optimized_attention_override": installed})
        with pytest.raises(error, match="controlled hook abort"):
            hooks._hooked_attention_forward(
                layer, x, None, opts, block_index=0, block_type="double", total_blocks=50)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        assert opts["optimized_attention_override"] is installed and "block_index" not in opts
        actual = hooks._hooked_attention_forward(
            layer, x, None, opts, block_index=0, block_type="double", total_blocks=50)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    assert state["calls"] == 2
