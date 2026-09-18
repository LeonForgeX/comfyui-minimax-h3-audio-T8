"""Real conditioning routes with fake encoders; no pretrained quality claim."""
import json

import pytest
import torch

from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from h3_audio_t8_pkg.long_video import LONG_VIDEO_SCHEMA, MOTION_AUDIO_END_FRAME, build_long_video_conditioning
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def context(width, height, segment):
    if segment == 0:
        return {'schema': LONG_VIDEO_SCHEMA, 'empty': True}
    return {'schema': LONG_VIDEO_SCHEMA, 'empty': False,
        'video_tail': torch.zeros(1, 24, 12, height // 16, width // 16),
        'audio_tail': torch.full((1, 32, 2, 65), .125),
        'metadata': {'source_segment_index': segment - 1, 'target_segment_index': segment,
            'max_context_frames': 39, 'audio_overhang': 1 / 3}}


def arguments(width, height, segment, reference):
    return dict(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        segment_index=segment, context_frames=22 if segment else 0,
        context_audio='video_and_audio', prompt='S1 uses <Audio 1> voice and says <d>你终于回来了。</d>',
        width=width, height=height, length=124, audio_mode='native', task_type='Ref2VA',
        add_source_as_reference=False, ref_images={'ref_image_0': torch.zeros(1, height, width, 3)},
        ref_audios={'ref_audio_0': reference})


@pytest.mark.parametrize('segment', [0, 1])
@pytest.mark.parametrize('width,height', [(128, 192), (256, 384)])
def test_reference_only_survives_both_stage_conditions_and_next_segment(monkeypatch, segment, width, height):
    reference = make_audio(2)
    before = reference['waveform'].clone()
    state = context(width, height, segment)
    audio_before = None if segment == 0 else state['audio_tail'].clone()
    # MODEL patch installation is orthogonal here. The real condition builder,
    # packed AV/reference layout and media map run without replacing their math.
    monkeypatch.setattr(runner, 'patch_long_video_model', lambda model: model)
    model = object()
    engine = runner.DualModelSegmentRunner(model, model, contract={},
        low_width=128, low_height=192, upscaler_model='fixture', coarse_steps=20)
    _, positive, latent, mux, prompt, _, _ = engine._conditions(model, state,
        arguments(width, height, segment, reference), None)
    assert mux is None and '<Audio 1>' in prompt
    assert any(item['kind'] == 'audio' and item['ref_audio_t'] == 80
        for item in positive[0][1]['minimax_refs'])
    assert torch.equal(reference['waveform'], before)
    if segment:
        assert torch.equal(state['audio_tail'], audio_before)
        # This genuine builder uses the original timeline motion-audio guide,
        # not a newly invented noise_mask. HIGH overlap masking is separate.
        motion = next(item for item in positive[0][1]['minimax_refs'] if MOTION_AUDIO_END_FRAME in item)
        assert torch.all(motion['audio_latent'] == .125)
    # Complete Stock20 audio remains the existing second-pass auto policy;
    # introducing a voice reference does not revive unfinished4/8 zero-lock.
    assert engine.audio_policy['first_pass_complete_trajectory'] is True


def test_video_soundtrack_changes_public_voice_ordinal_without_mux():
    reference = make_audio(2)
    values = arguments(128, 192, 0, reference)
    values.update(prompt='S1 uses <Audio 2> voice and says <d>你好。</d>',
        ref_videos={'ref_video_1': torch.zeros(48, 64, 64, 3)},
        ref_video_audios={'ref_video_audio_1': make_audio(2)})
    _, _, mux, prompt, media, _ = build_long_video_conditioning(
        context=context(128, 192, 0), **values)
    mapping = json.loads(media)
    assert mapping['audios'] == {'1': 'ref_video_audio_1', '2': 'ref_audio_1'}
    assert mapping['source_audio_ordinal'] is None
    assert '<Audio 2>' in prompt and mux is None
