"""CPU launch/math invariants; real CUDA receipts are separate artifacts."""
import math

import pytest
import torch

from h3_audio_t8_pkg.scoped_sage_triton import (
    _TwoStageLaunch, _clone_function, _natural_bias_to_log2, build_scoped_mask_kernel,
)


def test_natural_log_bias_equation_matches_exp2_logits():
    logits = torch.tensor([[-1., 0., 1., 2.]])
    bias = torch.tensor([[-2., -1., -.5, 0.]])
    expected = torch.softmax(logits + bias, -1)
    scaled = logits * math.log2(math.e) + _natural_bias_to_log2(bias, torch.float32)
    actual = torch.exp2(scaled) / torch.exp2(scaled).sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    wrong = torch.exp2(logits * math.log2(math.e) + bias)
    wrong = wrong / wrong.sum(-1, keepdim=True)
    assert (wrong - expected).abs().max() > .01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_float_bias_conversion_preserves_input_and_requested_dtype(dtype):
    bias = torch.tensor([[-2., 0., float('-inf')]], dtype=dtype)
    saved = bias.clone()
    output = _natural_bias_to_log2(bias, dtype)
    assert output.dtype == dtype
    torch.testing.assert_close(bias, saved)
    torch.testing.assert_close(output, (bias.float() * math.log2(math.e)).to(dtype))


def test_boolean_mask_is_not_rescaled():
    mask = torch.tensor([[True, False]])
    assert _natural_bias_to_log2(mask, torch.bfloat16) is mask


def test_launch_changes_only_pipeline_stages_without_mutating_caller():
    calls = []

    class Jit:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return "launched"
            return launch

    proxy = _TwoStageLaunch(Jit())
    grid = (1, 2, 1)
    options = {"BLOCK_M": 128, "BLOCK_N": 64, "HEAD_DIM": 128, "num_stages": 4, "num_warps": 8}
    payload = object()
    assert proxy[grid](payload, **options) == "launched"
    assert options["num_stages"] == 4
    assert calls == [(grid, (payload,), {**options, "num_stages": 2})]


def test_function_clone_changes_only_private_globals_and_keeps_defaults():
    def delegate(x=3, *, scale=2):
        return x * scale + torch.tensor(0).item()

    private_globals = dict(delegate.__globals__)
    cloned = _clone_function(delegate, private_globals)
    assert cloned.__globals__ is not delegate.__globals__
    assert cloned.__code__ is delegate.__code__
    assert cloned() == delegate() == 6
    private_globals['fixture_marker'] = True
    assert 'fixture_marker' not in delegate.__globals__


def test_unknown_outer_kernel_has_no_scoped_adapter():
    assert build_scoped_mask_kernel(lambda *a, **k: None) is None
