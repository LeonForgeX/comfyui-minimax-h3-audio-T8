from contextlib import contextmanager

import pytest
import torch

from h3_audio_t8_pkg import vdn_sdpa_backend as backend


def test_cpu_does_not_probe_cuda_and_matches_exact_reference(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: pytest.fail("CPU queried CUDA"))
    torch.manual_seed(22)
    q, k, v = torch.randn(3, 6, 2, 4, dtype=torch.float64)
    result = backend.exact_sdpa_rows(q, k, v, 0.5)
    expected = torch.einsum("hqk,khd->qhd", (torch.einsum("qhd,khd->hqk", q, k) * 0.5).softmax(-1), v)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)


def test_sm120_selects_safe_exact_backends_on_tensor_device(monkeypatch):
    import torch.nn.attention
    seen = []
    def capability(device):
        seen.append(device)
        return (12, 0)
    @contextmanager
    def kernel(backends):
        seen.append(backends)
        yield
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", kernel)
    with backend.vdn_sdpa_context(torch.device("cuda:1")):
        pass
    kinds = torch.nn.attention.SDPBackend
    assert seen == [torch.device("cuda:1"), [kinds.CUDNN_ATTENTION, kinds.FLASH_ATTENTION]]


def test_ada_keeps_existing_dispatch(monkeypatch):
    import torch.nn.attention
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda *_: pytest.fail("Ada changed"))
    with backend.vdn_sdpa_context("cuda:0"):
        pass


def test_backend_error_propagates_without_unsafe_retry(monkeypatch):
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("No available kernel")
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", failing)
    q = torch.zeros(2, 1, 4)
    with pytest.raises(RuntimeError, match="No available kernel"):
        backend.exact_sdpa_rows(q, q, q, 0.5)
    assert calls == [1]
