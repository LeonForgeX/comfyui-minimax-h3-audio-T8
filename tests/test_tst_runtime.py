"""TST local owner contracts. No model/backend/GPU execution is implied."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.tst_math import correct_video_queries
from h3_audio_t8_pkg.tst_runtime import TSTQueryRuntime


SIGMAS = [1., .9, .8, .7, .6, .4, .2, .1, 0.]


def runtime(**kwargs):
    options = dict(stage_start=0, stage_end=4, layer_count=2, frames=3, spatial_tokens=2)
    options.update(kwargs)
    return TSTQueryRuntime(SIGMAS, **options)


def options(layer=0, spans=None):
    return dict(block_index=layer, minimax_h3_layout=SimpleNamespace(seq_len=10,
        segments=spans if spans is not None else [(0, 2, 'text'), (2, 4, 'audio'), (4, 10, 'video')]))


@pytest.mark.parametrize('mode', ['report_only', 'apply_exp'])
def test_actual_full_clock_and_all_layers_match_math_not_phase_restart(mode):
    owner = runtime(stage_start=4, stage_end=8, mode=mode)
    q = torch.ones(1, 2, 10, 4)
    before = q.clone()
    for step in range(4, 8):
        with owner.forward(SIGMAS[step]):
            # The native token refiner is not a main layer.
            short = q[:, :, :2]
            assert owner.transform(short, short, 2, options()) is short
            for layer in range(2):
                expected, _ = correct_video_queries(q, q, video_start=4, frames=3, spatial_tokens=2,
                    layer_index=layer, layer_count=2, step_index=step, total_steps=8, mode=mode)
                actual = owner.transform(q, q, 2, options(layer))
                assert torch.equal(actual, expected)
                if mode == 'report_only' or step == 7:
                    assert actual is q
    report = owner.snapshot(require_complete=True)
    assert report['completed'] and not report['model_or_backend_calls_verified']
    assert [item['step_index'] for item in report['forwards']] == [4, 5, 6, 7]
    assert all(item['query_transform_calls'] == 2 for item in report['forwards'])
    # Uniform transport gives strictly positive tension: report actual extrema,
    # not artificial min(1,gamma), even in diagnostic-only mode.
    assert report['forwards'][0]['gamma_min'] > 1
    assert report['forwards'][-1]['gamma_min'] == report['forwards'][-1]['gamma_max'] == 1
    assert torch.equal(q, before)
    report['config']['tau'] = 99
    report['forwards'].clear()
    assert owner.snapshot()['config']['tau'] == .2
    assert len(owner.snapshot()['forwards']) == 4
    with pytest.raises(RuntimeError, match='outside'):
        with owner.forward(0.):
            pytest.fail('exhausted runtime admitted another forward')


@pytest.mark.parametrize('change', [dict(stage_start=-1), dict(stage_end=9), dict(stage_start=4),
    dict(layer_count=0), dict(frames=1), dict(spatial_tokens=0), dict(mode='disabled'),
    dict(tau=True), dict(max_workspace_mib=0)])
def test_configuration_rejection(change):
    with pytest.raises(ValueError):
        runtime(**change)


@pytest.mark.parametrize('schedule', [[1., 0.], [1., .5, .5, 0.], [1., -.1, 0.],
    [1., float('nan'), 0.], [True, .5, 0.], [1., .5, .1], [1.1, .5, 0.]])
def test_bad_full_schedules(schedule):
    with pytest.raises(ValueError):
        TSTQueryRuntime(schedule, stage_start=0, stage_end=1, layer_count=1, frames=2, spatial_tokens=1)


def test_outside_forward_and_wrong_sigma_are_rejected_without_tensor_mutation():
    owner = runtime()
    q = torch.ones(1, 2, 10, 4)
    with pytest.raises(RuntimeError, match='active owning'):
        owner.transform(q, q, 2, options())
    with pytest.raises(RuntimeError, match='sigma'):
        with owner.forward(.9):
            pytest.fail('wrong full-schedule index was admitted')
    assert not owner.snapshot()['forwards'] and torch.equal(q, torch.ones_like(q))


@pytest.mark.parametrize('fault', ['duplicate', 'layer', 'head', 'layout', 'drift', 'thread', 'nested', 'cancel', 'config'])
def test_forward_faults_clear_active_owner_and_require_new_runtime(fault):
    owner = runtime()
    q = torch.ones(1, 2, 10, 4)
    with pytest.raises(RuntimeError):
        with owner.forward(1.):
            owner.transform(q, q, 2, options(0))
            if fault == 'duplicate':
                owner.transform(q, q, 2, options(0))
            elif fault == 'layer':
                owner.transform(q, q, 2, options(2))
            elif fault == 'head':
                owner.transform(q, q, 1, options(1))
            elif fault == 'layout':
                owner.transform(q, q, 2, options(1, [(0, 2, 'text'), (3, 4, 'audio'), (4, 10, 'video')]))
            elif fault == 'drift':
                owner.transform(q, q, 2, options(1, [(0, 2, 'text'), (2, 8, 'video'), (8, 10, 'audio')]))
            elif fault == 'thread':
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(owner.transform, q, q, 2, options(1)).result()
            elif fault == 'nested':
                with owner.forward(1.):
                    pytest.fail('nested runtime admitted')
            elif fault == 'cancel':
                raise RuntimeError('cancel test')
            else:
                owner.config['tau'] = .3
                owner.transform(q, q, 2, options(1))
    assert owner.failed and owner._active is None and not owner._records
    with pytest.raises(RuntimeError):
        owner.snapshot()
    with pytest.raises(RuntimeError):
        with owner.forward(1.):
            pytest.fail('failed runtime reused')
    assert torch.equal(q, torch.ones_like(q))


def test_missing_query_owner_is_reported_and_does_not_abort_native_forward():
    owner = runtime()
    q = torch.ones(1, 2, 10, 4)
    with owner.forward(1.):
        owner.transform(q, q, 2, options(0))
    record = owner.snapshot()['forwards'][0]
    assert record['query_coverage_verified'] is False
    assert record['bypassed_layers'] == [1]
    assert owner.failed is False and owner._active is None


def test_keyboard_interrupt_also_clears_owner():
    owner = runtime()
    with pytest.raises(KeyboardInterrupt):
        with owner.forward(1.):
            raise KeyboardInterrupt
    assert owner._active is None and owner.failed


def test_incomplete_stage_and_snapshot_inside_forward_rejected():
    owner = runtime()
    with pytest.raises(RuntimeError, match='not completed'):
        owner.snapshot(require_complete=True)
    with owner.forward(1.):
        with pytest.raises(RuntimeError, match='inside a forward'):
            owner.snapshot()
        q = torch.ones(1, 2, 10, 4)
        for layer in range(2):
            owner.transform(q, q, 2, options(layer))
    assert len(owner.snapshot()['forwards']) == 1


def test_relay_dense_bias_oracle_and_eav_receive_same_corrected_query(monkeypatch):
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    from h3_audio_t8_pkg import prompt_relay_advanced as relay
    from h3_audio_t8_pkg.tst_runtime import TST_RUNTIME_KEY
    from comfy.ldm.modules import attention

    rng = torch.Generator().manual_seed(8)
    q, k, v = [torch.randn(1, 2, 10, 4, generator=rng) for _ in range(3)]
    originals = [tensor.clone() for tensor in (q, k, v)]
    owner = runtime(stage_end=1, layer_count=1, mode='apply_exp')
    event = dict(text_key_start=0, text_key_end=2, midpoint=1., window=.5, sigma=.75)
    times = torch.tensor([0., 0., 1., 1., 2., 2.])
    route = dict(seq_len=10, events=[event], query_segments=[dict(start=4, end=10, query_times=times)])
    opts = {**options(), TST_RUNTIME_KEY: owner, relay.PROMPT_RELAY_RUNTIME_KEY: route,
            eav.EAV_RUNTIME_KEY: dict(seq_len=10)}
    expected_q, _ = correct_video_queries(q, k, video_start=4, frames=3, spatial_tokens=2, mode='apply_exp')
    bias = torch.zeros(10, 10)
    bias[4:] = relay.make_prompt_relay_bias(times, 10, [event], dtype=q.dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(expected_q, k, v, attn_mask=bias)
    expected = expected.transpose(1, 2).reshape(1, 10, 8)
    seen = []

    def check_eav(actual_q, actual_k, output, eav_route):
        assert torch.equal(actual_q, expected_q) and actual_k is k
        seen.append(True)
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
        return output

    monkeypatch.setattr(attention, 'optimized_attention', attention.attention_pytorch)
    monkeypatch.setattr(eav, '_apply_eav_output_gain', check_eav)
    with owner.forward(1.):
        actual = eav.route_eav_prompt_relay_attention(q, k, v, 2, skip_reshape=True,
            transformer_options=opts, query_chunk_rows=2)
    assert seen == [True]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert owner.snapshot(require_complete=True)['forwards'][0]['query_transform_calls'] == 1
    assert opts[TST_RUNTIME_KEY] is owner
    for original, tensor in zip(originals, (q, k, v), strict=True):
        assert torch.equal(original, tensor)
