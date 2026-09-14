"""Public TST-before-Long first-window order; native tiny Euler, CPU doubles."""
import json

import pytest
import torch

from h3_audio_t8_pkg import progressive_first_segment as first
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.tst_model import TST_MODEL_KEY, build_tst_model, detach_tst_model
from test_progressive_first_segment import plan, prepare
from test_progressive_job import job
from test_progressive_checkpoint import same
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from test_relay_kj_memory import memory_nodes  # noqa: F401
from test_relay_kj_backend import kj  # noqa: F401


@pytest.mark.parametrize('mode', ['report_only', 'apply_exp'])
@pytest.mark.parametrize('backend', ['core', 'kj_memory_ffn'])
@pytest.mark.usefixtures('stub_lifter')
def test_public_first_binding_matches_explicit_relay_then_tst(backend, mode, memory_nodes):  # noqa: F811
    value = job(prompt_relay_plan=plan())
    base = value.models[0]
    if backend == 'kj_memory_ffn':
        lowmem, sage, _ = memory_nodes
        base = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(base).result[0]
        base = lowmem.MiniMaxChunkFeedForward.execute(base, 2, 256).result[0]
    options = dict(mode=mode, tau=.2, max_workspace_mib=128)
    wrapped = build_tst_model(base, value.sigmas, **options)[0]
    result, projected = prepare(value)
    actual, positive, negative, projection = first.bind_first_relay(
        wrapped, result, projected, NativeLikeFakeClip(), 64)
    plain, expected_positive, expected_negative, expected_projection = first.bind_first_relay(
        base, result, projected, NativeLikeFakeClip(), 64)
    expected = build_tst_model(plain, value.sigmas, **options)[0]
    assert detach_tst_model(actual)[1] == detach_tst_model(wrapped)[1]
    assert projection == expected_projection
    high = build_tst_model(base.clone(), value.sigmas, **options)[0]

    def sample(model, pos, neg, callback=None):
        return runtime.sample_progressive_h3(model, pos, neg, result[1], value.sampler,
            value.sigmas, model_hires=high, seed=8, upscaler_model='test', low_evaluations=4,
            input_mode='initialized_av_exp', task='t2va', eav_mode='apply_exp', eav_tau=.2,
            callback=callback)

    expected_output, _ = sample(expected, expected_positive, expected_negative)
    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('intentional HIGH cancellation')
    with pytest.raises(InterruptedError, match='intentional HIGH'):
        sample(actual, positive, negative, callback=cancel)
    output, text = sample(actual, positive, negative)
    same(output, expected_output)
    report = json.loads(text)
    assert report['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    for phase, start in [('low', 0), ('high', 4)]:
        assert [item['step_index'] for item in report['tst'][phase]['forwards']] == list(range(start, start+4))
        assert report['prompt_relay'][phase]['completed_calls']['forward'] == 4
        assert report['eav'][phase]['active_stage_forwards'] == 4
    wrapped.get_attachment(TST_MODEL_KEY).verify(wrapped)
    assert wrapped.get_attachment(TST_MODEL_KEY).active is None


@pytest.mark.usefixtures('stub_lifter')
def test_changed_tst_owner_is_not_stripped_for_relay():
    value = job(prompt_relay_plan=plan())
    wrapped = build_tst_model(value.models[0], value.sigmas, mode='apply_exp')[0]
    result, projected = prepare(value)
    wrapped.model_options['transformer_options']['optimized_attention_override'] = lambda *a, **k: torch.zeros(1)
    with pytest.raises(RuntimeError, match='owner/backend changed'):
        first.bind_first_relay(wrapped, result, projected, NativeLikeFakeClip(), 64)


@pytest.mark.usefixtures('stub_lifter')
def test_unknown_override_is_still_refused_without_tst():
    value = job(prompt_relay_plan=plan())
    base = value.models[0].clone()
    base.model_options['transformer_options']['optimized_attention_override'] = lambda *a, **k: torch.zeros(1)
    result, projected = prepare(value)
    with pytest.raises(RuntimeError, match='optimized_attention_override'):
        first.bind_first_relay(base, result, projected, NativeLikeFakeClip(), 64)


@pytest.mark.usefixtures('stub_lifter')
def test_single_event_keeps_original_tst_identity():
    value = job(prompt_relay_plan=plan('One continuous event.'))
    wrapped = build_tst_model(value.models[0], value.sigmas, mode='apply_exp')[0]
    result, projected = prepare(value)
    actual, _, _, report = first.bind_first_relay(wrapped, result, projected, NativeLikeFakeClip(), 64)
    assert actual is wrapped and report['status'] == 'passthrough'
