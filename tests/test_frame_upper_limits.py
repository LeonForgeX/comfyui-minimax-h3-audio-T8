"""Long-frame admission only; CPU fixtures are not video quality qualification."""
import asyncio
import ast
import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.core import align_frame_count, reference_video_frame_warnings
from h3_audio_t8_pkg.long_video import build_long_video_conditioning
from h3_audio_t8_pkg.multikeyframe_advanced import (
    _build_advanced_conditioning, append_keyframe_plan, resolve_keyframe_plan,
)
from h3_audio_t8_pkg.prompt_relay_packet_advanced import build_prompt_relay_plan_from_packet
from h3_audio_t8_pkg.prompt_relay_preview_advanced import preview_prompt_relay_plan
from h3_audio_t8_pkg.prepared_generation_contract import validate_geometry
from h3_audio_t8_pkg.speech import render_frame_count
from h3_audio_t8_pkg.studio_advanced import build_studio_timeline, compile_prompt_packet
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE


FRAME_INPUTS = {
    'length', 'expected_length', 'video_frame_count', 'frame_count',
    'render_window_frames', 'minimum_render_frames', 'min_render_frames',
    'max_render_frames', 'window_frame_count', 'reference_video_frames_each',
    'frame_index', 'current_frame', 'frame_in_shot', 'total_frames',
    'context_before_frames', 'context_after_frames', 'hold_frames',
    'repair_context_frames', 'minimum_hot_frames', 'max_expanded_frames',
    'absolute_start_frame',
}
DURATION_INPUTS = {
    'scene_duration_seconds', 'source_duration_seconds', 'duration_seconds',
    'target_duration_seconds', 'total_duration_seconds', 'new_duration_seconds',
    'default_duration_seconds', 'render_seconds', 'min_scene_seconds',
    'target_scene_seconds', 'max_scene_seconds',
    'duration',
}


def test_every_public_schema_has_no_project_frame_admission_maximum():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [cls.define_schema().node_id for cls in classes]
    assert len(ids) == 340 and len(set(ids)) == 340
    assert ids[-1] == 'SolAttnMiniMax'
    checked = []
    for cls in classes:
        for item in cls.define_schema().inputs or []:
            if item.id in FRAME_INPUTS | DURATION_INPUTS:
                assert getattr(item, 'max', None) is None, (cls.__name__, item.id)
                # None must disappear from the actual object_info/v1 field.
                assert 'max' not in item.as_dict()
                checked.append((cls.__name__, item.id))
    assert len(checked) >= 60


def test_schema_source_inventory_cannot_reintroduce_an_upper_cap():
    for source in (Path(__file__).resolve().parents[1] / 'h3_t8').glob('nodes*.py'):
        for call in ast.walk(ast.parse(source.read_text(encoding='utf-8-sig'))):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'Input' and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value in FRAME_INPUTS | DURATION_INPUTS):
                continue
            for keyword in call.keywords:
                if keyword.arg == 'max':
                    assert isinstance(keyword.value, ast.Constant) and keyword.value.value is None


@pytest.mark.parametrize('route', ['stable', 'middle_keyframe', 'continuation'])
@pytest.mark.parametrize('count', [379, 869, 3626])
def test_all_three_reference_video_paths_allow_long_inputs_with_warning(route, count):
    aligned = align_frame_count(count)
    video_vae = FakeVideoVAE()
    args = dict(clip=FakeClip(), video_vae=video_vae, audio_vae=FakeAudioVAE(),
                prompt='A stable actor. No speech.', width=32, height=32,
                length=aligned, audio_mode='native', add_source_as_reference=False,
                ref_videos={'ref_video_1': torch.zeros(count, 32, 32, 3)})
    if route == 'middle_keyframe':
        keyframes = append_keyframe_plan(
            None, torch.zeros(1, 32, 32, 3), 'frame', 17, .999, 'center_crop', True)
        args.update(resolved_middle=resolve_keyframe_plan(keyframes, aligned),
                    task_type='hybrid', first_frame=torch.zeros(1, 32, 32, 3),
                    last_frame=torch.zeros(1, 32, 32, 3), audio_denoise_strength=.35,
                    prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
                    ref_image_size='match', reference_video_policy='official_2_to_15s',
                    first_frame_noise_aug=.999, last_frame_noise_aug=.999,
                    reference_visual_noise_aug=.999)
        result = _build_advanced_conditioning(**args)
    elif route == 'continuation':
        args.update(context={'schema': 1, 'empty': False,
                    'video_tail': torch.zeros(1, 24, 12, 2, 2),
                    'audio_tail': torch.zeros(1, 32, 2, 65),
                    'metadata': {'source_segment_index': 0, 'target_segment_index': 1,
                                 'max_context_frames': 39, 'audio_overhang': 1 / 3}},
                    segment_index=1, context_frames=22, context_audio='video_and_audio')
        result = build_long_video_conditioning(**args)
    else:
        result = build_conditioning(**args)
    assert any(f'has {count} frames' in str(value) and 'not an upper limit' in str(value) for value in result)
    # Existing fitting/alignment is unchanged, rather than silently cutting at360.
    assert max(batch.shape[0] for batch in video_vae.encode_calls) == count - (count - 5) % 17


def test_reference_minimum_and_training_threshold_remain_separate():
    with pytest.raises(ValueError, match='at least48|at least 48'):
        reference_video_frame_warnings(47, 1, 'official_2_to_15s')
    assert reference_video_frame_warnings(48, 1, 'official_2_to_15s') == []
    assert reference_video_frame_warnings(869, 1, 'allow_short_exp') == []


def test_packet_preview_and_speech_cross_old_upper_limits():
    packet = compile_prompt_packet('Stable actor.', 'minimax_h3', 200., '2:3', '')
    plan, *_ = build_prompt_relay_plan_from_packet(
        packet, '[]', 'auto_equal', 'paper_v1', .1, False, False)
    assert plan['frame_count'] == align_frame_count(4800)
    assert preview_prompt_relay_plan(plan)[1] is True
    assert render_frame_count(40.) == align_frame_count(960)


def test_explicit_studio_no_split_keeps_whole_duration():
    result = build_studio_timeline('long', json.dumps([{'prompt': 'Continuous shot.',
        'duration_seconds': 40.}]), 'minimax_h3', 5., '2:3', 1, 'fixed', False, False)
    assert len(result['shots']) == 1
    assert result['shots'][0]['frame_count'] == align_frame_count(960)


def test_prepared_ltx_no_arbitrary_192_frame_limit_but_keeps_grid_and_tokens():
    assert validate_geometry({'frames': 385, 'width': 32, 'height': 32, 'fps': 24})['frames'] == 385
    with pytest.raises(ValueError, match='8n'):
        validate_geometry({'frames': 386, 'width': 32, 'height': 32, 'fps': 24})
    with pytest.raises(ValueError, match='envelope'):
        validate_geometry({'frames': 385, 'width': 8192, 'height': 8192, 'fps': 24})
