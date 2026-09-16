"""Opt-in random tiny VSA kernel tests; not full-checkpoint media qualification."""
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from h3_audio_t8_pkg import fast_h3_v2_advanced as v2

pytestmark = pytest.mark.skipif(os.environ.get('T8_V2_CUDA') != '1', reason='Explicit serial V2 CUDA probe only')


def _attention():
    from comfy.ldm.minimax.model import Attention
    torch.manual_seed(26091601)
    return Attention(384, 4, 128, 1e-6, gate_compress=True,
                     dtype=torch.bfloat16, device='cuda', operations=nn)


def _invoke(attn, groups, *, gates=True):
    from comfy.ldm.minimax.model import PackedLayout
    sparse = v2._core_sparse()
    assert sparse.ck.sol_attn_is_available(torch.device('cuda'))
    layout = PackedLayout(65, 5, 3, 5, 3)
    torch.manual_seed(26091602)
    x = torch.randn(layout.seq_len, 384, device='cuda', dtype=torch.bfloat16)
    # H3 carries2x2 rotation matrices, not bare(cos,sin) pairs.
    rope = torch.eye(2, device='cuda').expand(1, layout.seq_len, 32, 2, 2).contiguous()
    patch = sparse.SparseAttnPatch(1., .2, True, 1., 0., 0, set(), 'exact_kv_and_rows', 0, False)
    runtime = v2._V2Runtime('trained_vsa_exp', sparse, patch, groups)
    if not gates:
        with torch.no_grad():
            attn.to_gate_compress.weight.zero_()
    options = {'minimax_h3_layout': layout, 'sigmas': torch.tensor([.9], device='cuda')}
    hook = runtime.block_patch(SimpleNamespace(attn=attn), 0)
    runtime.dit = {('double_block', 0): hook}
    options['patches_replace'] = {'dit': runtime.dit}
    args = {'img': x, 'rope_freqs': rope, 'transformer_options': options}
    def original(values):
        assert 'attention' in values  # No Dense output masquerading as a VSA result.
        return values['attention'](values['img'], rope_freqs=rope, transformer_options=options)
    with torch.inference_mode():
        output = hook(args, {'original_block': original})
    torch.cuda.synchronize()
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert runtime.counts['vsa'] == 1 and not runtime.counts['dense']
    assert len(patch.pooled) == groups
    return output, attn


def test_real_kernel_preserves_head_group_gate_math_and_shared_projection_weights():
    attn = _attention()
    source = attn.to_gate_compress.weight.detach().clone()
    single, _ = _invoke(attn, 1)
    grouped, _ = _invoke(attn, 4)
    assert torch.equal(attn.to_gate_compress.weight, source)
    # Different GEMM/kernel group sizes may change rounding; never claim bit identity.
    torch.testing.assert_close(grouped, single, rtol=.02, atol=.02)


def test_learned_gate_has_real_effect_in_compiled_vsa_coarse_branch():
    attn = _attention()
    enabled, _ = _invoke(attn, 1)
    disabled, _ = _invoke(attn, 1, gates=False)
    assert (enabled.float() - disabled.float()).abs().max().item() > 1e-4
