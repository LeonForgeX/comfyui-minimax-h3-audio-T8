"""Real tiny native sampling; encoder/lifter boundaries explicitly doubled."""

import json

import pytest
import torch

from h3_audio_t8_pkg import progressive_first_segment as first
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.long_video import LONG_VIDEO_CONDITIONING_KEY
from test_progressive_job import job, bind_parent
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401
from test_progressive_continuation import accepted  # noqa: F401
from test_progressive_checkpoint import same
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from helpers import FakeVideoVAE, FakeAudioVAE, make_audio


def plan(events='Walk.\nStop.', route='joint_av_exp'):
    result = relay.build_prompt_relay_plan('Scene.', events, 193, 'auto_equal', '',
                                          'paper_v1', .1, False, False)[0]
    result['query_route'] = route
    result.pop('plan_hash')
    result['plan_hash'] = relay._sha256_json(result)
    return result


def prepare(value, **kwargs):
    return first.prepare_first_segment(clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), chain_id='chain', segment=value.segments[0], width=value.width,
        height=value.height, condition_options=value.condition_options.get(0),
        prompt_relay_plan=value.relay_plan, **kwargs)


@pytest.mark.parametrize('kind', ['plain', 'i2va', 'locked_audio', 'relay'])
def test_first_builder_retains_native_conditions_audio_and_projection(stub_lifter, kind):  # noqa: F811
    options = {}
    if kind == 'i2va':
        options['first_frame'] = torch.linspace(0, 1, 32 * 64 * 3).reshape(1, 32, 64, 3)
    if kind == 'locked_audio':
        options.update(drive_audio=make_audio(6), audio_mode='lock_source', add_source_as_reference=False)
    value = job(segment_condition_options={0: options}, prompt_relay_plan=plan() if kind == 'relay' else None)
    result, projected = prepare(value)
    assert LONG_VIDEO_CONDITIONING_KEY not in result[0][0][1]
    assert result[6]['resolved_task'] == ('i2va' if kind == 'i2va' else 't2va')
    assert result[6]['context_active'] is False
    if kind == 'locked_audio':
        assert result[2] is options['drive_audio']
        assert result[1]['noise_mask'] is not None
    if projected is not None:
        assert result[3] == projected['compiled_prompt']
        assert projected['long_video_projection']['context_frames'] == 0
        assert projected['long_video_projection']['render_start_frame'] == 0
        assert projected['events'] == [{**event, 'global_start_frame': event['start_frame'],
            'global_end_frame_exclusive': event['end_frame_exclusive']} for event in value.relay_plan['events']]


def test_first_does_not_remove_foreign_metadata(stub_lifter, monkeypatch):  # noqa: F811
    value = job()
    build = first.build_long_video_conditioning
    def changed(*args, **kwargs):
        result = build(*args, **kwargs)
        result[0][0][1]['foreign_payload'] = 1
        return result
    monkeypatch.setattr(first, 'build_long_video_conditioning', changed)
    # The factory signature itself changed and therefore cannot expand allowed
    # media options; with no options it still preserves unknown metadata.
    result, _ = prepare(value)
    assert result[0][0][1]['foreign_payload'] == 1
    with pytest.raises(ValueError, match='Reference/area/hook'):
        runtime.prepare_stage_conditioning(result[0], value.plan, positive=True)


def test_single_event_passthrough_and_locked_joint_rejection(stub_lifter):  # noqa: F811
    value = job(prompt_relay_plan=plan('Walk.'))
    result, projected = prepare(value)
    model, _, _, report = first.bind_first_relay(value.models[0], result, projected, NativeLikeFakeClip(), 64)
    assert model is value.models[0] and report['status'] == 'passthrough'
    value = job(prompt_relay_plan=plan(), segment_condition_options={0:
        dict(drive_audio=make_audio(6), audio_mode='lock_source', add_source_as_reference=False)})
    result, projected = prepare(value)
    with pytest.raises(ValueError, match='locked source audio'):
        first.bind_first_relay(value.models[0], result, projected, NativeLikeFakeClip(), 64)


@pytest.mark.parametrize('use_relay', [False, True])
def test_bound_first_actual_euler_cancel_low_restore_and_completed_cache(
        tmp_path, stub_lifter, monkeypatch, use_relay):  # noqa: F811
    value = job(prompt_relay_plan=plan() if use_relay else None,
                sampling_options={'eav_mode': 'apply_exp', 'eav_tau': .2})
    original = first.prepare_first_segment
    fake_clip = NativeLikeFakeClip()
    calls = []
    def prepared(**kwargs):
        for key, component in value.producers.components.items():
            assert kwargs[key] is component
        calls.append(kwargs['segment'].index)
        kwargs.update(clip=fake_clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE())
        return original(**kwargs)
    bind = first.bind_first_relay
    def bound(model, result, projected, clip, chunks):
        assert clip is value.producers.components['clip']
        return bind(model, result, projected, fake_clip, chunks)
    monkeypatch.setattr(first, 'prepare_first_segment', prepared)
    monkeypatch.setattr(first, 'bind_first_relay', bound)
    with value.exclusive(tmp_path / 'baseline'):
        baseline, baseline_text = value.sample_first()
    assert json.loads(baseline_text)['counts']['actual_forwards'] == {'low': 4, 'high': 4}
    if not use_relay:
        # Independent existing empty-input sampler oracle, not another job
        # invocation. The adapter must not change the plain native trajectory.
        prepared_result, _ = original(clip=fake_clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
            chain_id='chain', segment=value.segments[0], width=value.width, height=value.height)
        direct, _ = runtime.sample_progressive_h3(value.models[0], prepared_result[0], prepared_result[0],
            prepared_result[1], value.sampler, value.sigmas, model_hires=value.models[1],
            seed=value.segments[0].seed, **value.options)
        same(baseline, direct)
    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('HIGH interrupted')
    with value.exclusive(tmp_path / 'resumed'):
        with pytest.raises(InterruptedError):
            value.sample_first(callback=cancel)
    callbacks = []
    with value.exclusive(tmp_path / 'resumed'):
        output, text = value.sample_first(callback=lambda step, *_: callbacks.append(step))
    assert callbacks == [4, 5, 6, 7]
    same(output, baseline)
    report = json.loads(text)
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    if use_relay:
        assert report['prompt_relay']['high']['completed_calls']['forward'] == 4
    with value.exclusive(tmp_path / 'resumed'):
        output, text = value.sample_first(callback=lambda *_: pytest.fail('Cached HIGH sampled again'))
    same(output, baseline)
    assert json.loads(text)['counts']['actual_forwards'] == {'low': 0, 'high': 0}
    assert calls == [0, 0, 0]


def test_job_binds_first_guide_policy(stub_lifter):  # noqa: F811
    value = job(guide_resize='preserve_mean')
    assert value.identity['guide_resize'] == 'preserve_mean'
    value.guide_resize = 'legacy_bilinear'
    with pytest.raises(ValueError, match='contract changed'):
        value.verify()


def test_first_requires_lock_and_rejects_already_accepted(accepted, stub_lifter):  # noqa: F811
    value = job()
    with pytest.raises(RuntimeError, match='lock'):
        value.sample_first()
    bind_parent(accepted, value.sha256)
    with value.exclusive(accepted.root):
        with pytest.raises(ValueError, match='already accepted'):
            value.sample_first()
