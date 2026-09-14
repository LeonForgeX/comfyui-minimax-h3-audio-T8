"""Independent NumPy/SVD oracles; no Core, trained model, or CUDA inference."""

import math

import numpy as np
import pytest
import torch

from h3_audio_t8_pkg import tst_math as tst


def oracle(a):
    positive = a > 0
    terms = np.zeros_like(a)
    terms[positive] = a[positive] * np.log(a[positive])
    row = -terms.sum(axis=-1).mean() / math.log(a.shape[-1])
    singular = np.linalg.svd(a, compute_uv=False)
    probabilities = singular**2 / np.sum(singular**2)
    probabilities = probabilities[probabilities > 0]
    spectral = -np.sum(probabilities * np.log(probabilities)) / math.log(a.shape[-1])
    return row, spectral, row - spectral


@pytest.mark.parametrize('frames', [2, 3, 7])
def test_extremes_and_permutation(frames):
    identity = torch.eye(frames, dtype=torch.float64)
    uniform = torch.full((frames, frames), 1 / frames, dtype=torch.float64)
    permutation = identity.flip(0)
    result = tst.transport_statistics(torch.stack((identity, uniform, permutation)))
    torch.testing.assert_close(result['tension'], torch.tensor([-1., 1., -1.], dtype=torch.float64),
                               rtol=0, atol=1e-12)


def test_non_symmetric_operator_matches_independent_svd():
    a = np.array([[.8, .1, .1], [.2, .3, .5], [.05, .85, .1]], dtype=np.float64)
    expected = oracle(a)
    actual = tst.transport_statistics(torch.from_numpy(a))
    for key, value in zip(('row_entropy', 'spectral_entropy', 'tension'), expected):
        assert float(actual[key]) == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize('fault', ['negative', 'nan', 'rows', 'rectangular', 'single', 'integer'])
def test_invalid_transport_rejected(fault):
    a = torch.eye(3)
    if fault == 'negative':
        a[0, 1] = -.1
    elif fault == 'nan':
        a[0, 0] = float('nan')
    elif fault == 'rows':
        a[0, 0] = .5
    elif fault == 'rectangular':
        a = a[:, :2]
    elif fault == 'single':
        a = torch.ones(1, 1)
    else:
        a = a.long()
    with pytest.raises(ValueError):
        tst.transport_statistics(a)


def test_full_schedule_cosine_endpoints_and_resume():
    kwargs = dict(tau=.2, layer_count=50, total_steps=8)
    values = [tst.effective_strength(layer_index=49, step_index=i, **kwargs) for i in range(8)]
    assert values[0] == .2 and values[-1] == 0
    assert all(a > b for a, b in zip(values, values[1:]))
    high_after_resume = [tst.effective_strength(layer_index=49, step_index=i, **kwargs) for i in range(4, 8)]
    assert high_after_resume == values[4:] and high_after_resume[0] < values[0]
    shallow = tst.effective_strength(layer_index=0, step_index=0, **kwargs)
    assert 0 < shallow < values[0]


@pytest.mark.parametrize('change', [dict(tau=float('nan')), dict(tau=-1), dict(tau=3),
    dict(layer_index=True), dict(layer_index=50), dict(layer_count=0),
    dict(step_index=-1), dict(step_index=8), dict(total_steps=1)])
def test_invalid_clock_or_strength_rejected(change):
    args = dict(tau=.2, layer_index=49, layer_count=50, step_index=0, total_steps=8)
    args.update(change)
    with pytest.raises(ValueError):
        tst.effective_strength(**args)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_query_correction_matches_batched_headwise_oracle_and_preserves_other_inputs(dtype, strided):
    rng = torch.Generator().manual_seed(133)
    q = torch.randn(2, 3, 11, 4, dtype=dtype, generator=rng)
    k = torch.randn(2, 3, 11, 4, dtype=dtype, generator=rng)
    if strided:
        q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
    saved_q, saved_k = q.clone(), k.clone()
    result, report = tst.correct_video_queries(q, k, video_start=2, frames=3, spatial_tokens=2,
        mode='apply_exp', tau=.2, layer_index=2, layer_count=3)
    assert report['applied'] and result is not q
    expected = q.double().numpy().copy()
    for batch in range(2):
        for head in range(3):
            qv = q[batch, head, 2:8].double().numpy().reshape(3, 2, 4).mean(axis=1)
            kv = k[batch, head, 2:8].double().numpy().reshape(3, 2, 4).mean(axis=1)
            logits = qv @ kv.T / 2
            a = np.exp(logits - logits.max(axis=-1, keepdims=True))
            a /= a.sum(axis=-1, keepdims=True)
            gamma = np.exp(.2 * oracle(a)[2])
            expected[batch, head, 2:8] *= gamma
            assert float(report['gamma'][batch, head]) == pytest.approx(gamma, abs=2e-7)
    torch.testing.assert_close(result, torch.from_numpy(expected).to(dtype), rtol=2e-6, atol=2e-7)
    assert torch.equal(q, saved_q) and torch.equal(k, saved_k)
    assert torch.equal(result[:, :, :2], q[:, :, :2])
    assert torch.equal(result[:, :, 8:], q[:, :, 8:])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_report_only_and_zero_schedule_return_original_query(dtype):
    q = torch.ones(1, 2, 8, 4, dtype=dtype)
    args = dict(video_start=2, frames=2, spatial_tokens=2)
    result, report = tst.correct_video_queries(q, q, mode='report_only', **args)
    assert result is q and not report['applied'] and torch.isfinite(report['tension']).all()
    for kwargs in (dict(tau=0), dict(step_index=7)):
        result, report = tst.correct_video_queries(q, q, mode='apply_exp', **args, **kwargs)
        assert result is q and not report['applied']


def test_disabled_is_transparent_without_tensor_access():
    sentinel = object()
    result, report = tst.correct_video_queries(sentinel, None, video_start=None, frames=None, spatial_tokens=None)
    assert result is sentinel and report['estimated_tensor_bytes'] == 0


@pytest.mark.parametrize('fragmented', [False, True])
def test_signed_correction_changes_query_temperature_in_both_directions(fragmented):
    video = torch.eye(3, dtype=torch.float64) * 10 if fragmented else torch.ones(3, 3, dtype=torch.float64)
    q = video.repeat_interleave(2, dim=0)[None, None]
    result, report = tst.correct_video_queries(q, q, video_start=0, frames=3, spatial_tokens=2,
                                              mode='apply_exp')
    gamma = float(report['gamma'])
    assert (gamma < 1) if fragmented else (gamma > 1)
    assert not torch.equal(q, result)
    assert 'operator' not in report
    assert all(value.numel() <= 1 for value in report.values() if isinstance(value, torch.Tensor))


def test_half_precision_overflow_rejected_without_mutating_query():
    q = torch.full((1, 1, 4, 2), 60000., dtype=torch.float16)
    before = q.clone()
    with pytest.raises(ValueError, match='overflowed'):
        tst.correct_video_queries(q, q, video_start=0, frames=2, spatial_tokens=2, mode='apply_exp')
    assert torch.equal(q, before) and torch.isfinite(q).all()


@pytest.mark.parametrize('change', [dict(video_start=-1), dict(video_start=7), dict(frames=1),
    dict(frames=True), dict(spatial_tokens=0), dict(mode='unknown'), dict(max_workspace_mib=0)])
def test_bad_span_or_budget_config_rejected(change):
    q = torch.ones(1, 2, 8, 4)
    args = dict(video_start=2, frames=2, spatial_tokens=2, mode='apply_exp')
    args.update(change)
    with pytest.raises(ValueError):
        tst.correct_video_queries(q, q, **args)


def test_workspace_rejection_precedes_quadratic_operator(monkeypatch):
    q = torch.zeros(1, 1, 512, 1024)
    monkeypatch.setattr(torch.linalg, 'eigvalsh', lambda *_: pytest.fail('budget allowed eigensolver'))
    with pytest.raises(ValueError, match='workspace budget'):
        tst.correct_video_queries(q, q, video_start=0, frames=2, spatial_tokens=256,
                                  mode='apply_exp', max_workspace_mib=1)


@pytest.mark.parametrize('fault', ['nan', 'shape', 'dtype'])
def test_invalid_queries_rejected(fault):
    q, k = torch.ones(1, 2, 8, 4), torch.ones(1, 2, 8, 4)
    if fault == 'nan':
        k[0, 0, 0, 0] = float('nan')
    elif fault == 'shape':
        k = k[:, :, :7]
    else:
        k = k.double()
    with pytest.raises(ValueError):
        tst.correct_video_queries(q, k, video_start=2, frames=2, spatial_tokens=2, mode='apply_exp')
