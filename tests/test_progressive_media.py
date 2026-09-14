"""Actual timeline windows and delivery precedence; no model/GPU execution."""

import pytest
import torch

from h3_audio_t8_pkg.progressive_media import resolve_segment_options, select_delivery_audio
from h3_audio_t8_pkg.long_video_orchestration import build_long_video_chain_plan
from test_progressive_job import job
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def audio(samples=300, rate=24):
    return {'waveform': torch.arange(samples, dtype=torch.float32).reshape(1, 1, -1) / samples,
            'sample_rate': rate}


def segments(context=22):
    return build_long_video_chain_plan(chain_id='media', total_duration_seconds=8,
        render_window_frames=124, context_frames=context, global_prompt='Walk.',
        segment_prompts_json='', base_seed=0, seed_policy='increment')


@pytest.mark.parametrize('context', [22, 39])
@pytest.mark.parametrize('rate', [24, 32000, 44100, 48000])
def test_shared_audio_includes_exact_context_without_restart(context, rate):
    source = audio(6 * rate, rate)
    snapshot = source['waveform'].clone()
    first, second = segments(context)
    start = round((second.plan.timeline_start_seconds - context / 24) * rate)
    selected = resolve_segment_options({'drive_audio': source}, {}, second)['drive_audio']
    expected = torch.zeros(1, 1, round(second.plan.render_frames / 24 * rate))
    count = min(expected.shape[-1], snapshot.shape[-1] - start)
    expected[..., :count] = snapshot[..., start:start + count]
    assert torch.equal(selected['waveform'], expected)
    assert torch.equal(source['waveform'], snapshot)
    initial = resolve_segment_options({'drive_audio': source}, {}, first)['drive_audio']
    assert torch.equal(initial['waveform'], snapshot[..., :initial['waveform'].shape[-1]])


def test_reference_audio_and_reference_video_are_not_timeline_sliced():
    reference = audio(12)
    video = {'reference': torch.ones(2, 32, 32, 3)}
    frame = torch.ones(1, 32, 32, 3)
    shared = dict(audio_mode='reference_only', drive_audio=reference,
                  ref_videos=video, first_frame=frame, last_frame=frame)
    first, second = segments()
    early = resolve_segment_options(shared, {}, first)
    late = resolve_segment_options(shared, {}, second)
    assert early['drive_audio'] is late['drive_audio'] is reference
    assert early['ref_videos'] is late['ref_videos'] is video
    assert late['first_frame'] is frame
    assert 'last_frame' not in early and late['last_frame'] is frame
    assert shared['last_frame'] is frame


def test_local_already_windowed_override_is_not_cropped_again():
    source, local = audio(), audio(19)
    first, second = segments()
    selected = resolve_segment_options({'drive_audio': source, 'final_audio': source},
                                      {'drive_audio': local, 'final_audio': None}, second)
    assert selected['drive_audio'] is local
    assert selected['final_audio'] is None
    frame = torch.ones(1, 32, 32, 3)
    assert resolve_segment_options({}, {'last_frame': frame}, first)['last_frame'] is frame


@pytest.mark.parametrize('mode', ['native', 'lock_source', 'reference_only', 'remix_source'])
@pytest.mark.parametrize('override', [False, True])
def test_delivery_precedence_preserves_original_audio_semantics(mode, override):
    source, final, generated = audio(20), audio(21), audio(22)
    options = dict(audio_mode=mode, drive_audio=source, final_audio=final if override else None)
    selected, origin = select_delivery_audio(options, generated)
    expected = (final, 'final_audio') if override else (
        (source, 'drive_audio') if mode in {'native', 'lock_source'} else (generated, 'generated_audio'))
    assert selected is expected[0] and origin == expected[1]


@pytest.mark.parametrize('options', [{'width': 128}, {'unknown': None}, {1: None},
    {'audio_mode': 'bogus'}, {'audio_mode': 'lock_source'}, {'audio_mode': None}])
def test_invalid_or_owned_options_rejected(options):
    with pytest.raises(ValueError):
        resolve_segment_options(options, {}, segments()[0])


def test_shared_source_is_bound_to_job_and_delivery_survives_reentry(tmp_path, stub_lifter):  # noqa: F811
    source = audio()
    value = job(shared_condition_options={'drive_audio': source})
    with pytest.raises(RuntimeError, match='lock'):
        value.resolved_condition_options(1)
    with value.exclusive(tmp_path):
        selected = value.resolved_condition_options(1)['drive_audio']
        delivered, report = value.delivery_audio(1, audio(1))
        assert torch.equal(delivered['waveform'], selected['waveform'])
        assert report['origin'] == 'drive_audio'
        saved = delivered['waveform'].clone()
        for invalid in (-1, 2, True):
            with pytest.raises(ValueError, match='planned segment'):
                value.resolved_condition_options(invalid)
    with value.exclusive(tmp_path):
        restored, new_report = value.delivery_audio(1, audio(1))
        assert torch.equal(restored['waveform'], saved) and new_report == report
        source['waveform'][..., 0] += .1
        with pytest.raises(ValueError, match='contract changed'):
            value.resolved_condition_options(1)
