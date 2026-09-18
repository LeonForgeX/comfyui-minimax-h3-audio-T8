"""Pure guard for one task-owned timed-out single-model probe; no sampling math."""
import copy
from pathlib import Path
import re


def owned_resume_chain_root(previous, chain_id):
    """Validate containment before opening any manifest from recipe-supplied text."""
    previous = Path(previous).resolve(strict=True)
    if not re.fullmatch(r'owned_voice_long_[0-9a-f]{16}', chain_id):
        raise ValueError('An exact owned chain name is required before file access')
    state_root = (previous / 'output/minimax_h3_t8_long_video').resolve(strict=True)
    if not state_root.is_relative_to(previous):
        raise ValueError('Owned state folder escaped the prior probe')
    root = (state_root / chain_id).resolve(strict=True)
    if not root.is_relative_to(state_root):
        raise ValueError('Owned chain escaped the state folder')
    return root


def prepare_owned_single_resume(recipe, manifest, process_receipt):
    process = process_receipt.get('process', {})
    if process_receipt.get('status') != 'failed' or process.get('status') != 'timeout' or process.get('active_after_cleanup') != 0:
        raise ValueError('Resume only a confirmed timed-out and cleaned-up owned Job')
    return _validated_fixed_recipe_and_first(recipe, manifest, relay_owner='12')


def prepare_owned_window_resume(recipe, manifest, process_receipt, terminal, *, text_policy='accepted_window_text_exp'):
    if text_policy not in ('accepted_window_text_exp', 'dialogue_start_owner_exp'):
        raise ValueError('Known explicit experimental text policy required')
    process = process_receipt.get('process', {})
    if (process_receipt.get('status') != 'owned_worker_exit0_cleanup0'
            or process.get('status') != 'complete' or process.get('exit_code') != 0
            or process.get('active_after_cleanup') != 0
            or terminal.get('status') != 'controlled_native_voice_window_first_committed_cancelled_not_complete8s'
            or terminal.get('actual_forwards') != 20 or terminal.get('accepted_count') != 1
            or terminal.get('sources_unchanged') is not True
            or terminal.get('window_text_policy') != text_policy):
        raise ValueError('Only resume the confirmed cleaned-up controlled window probe')
    if recipe.get('13') != dict(class_type='MiniMaxH3PromptRelayWindowTextEXPT8',
                                inputs=dict(prompt_relay_plan=['12', 0], text_policy=text_policy)):
        raise ValueError('Preserve the explicit window text owner')
    return _validated_fixed_recipe_and_first(recipe, manifest, relay_owner='13')


def _validated_fixed_recipe_and_first(recipe, manifest, *, relay_owner):
    target = recipe['8']['inputs']
    expected = dict(steps=20, shift_video=12., shift_audio=3.,
                    sampler_name='dual_clock_euler', scheduler='native_flow',
                    task_type='Ref2VA', audio_mode='native',
                    total_duration_seconds=8., width=512, height=768,
                    render_window_frames=124, context_frames=22,
                    resume_existing=True, model=['11', 0], prompt_relay_plan=[relay_owner, 0],
                    add_source_as_reference=False, prompt_primary_audio_ordinal=0,
                    **{'ref_images.ref_image_0': ['5', 0], 'ref_audios.ref_audio_0': ['6', 0]})
    if (recipe['8']['class_type'] != 'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'
            or any(target.get(key) != value for key, value in expected.items())
            or any(key in target for key in ('drive_audio', 'final_audio', 'upscaler_model', 'model_pass1', 'model_pass2'))):
        raise ValueError('Do not change this fixed reference-only Stock20 recipe or its owners')
    if recipe.get('12') != dict(class_type='MiniMaxH3PromptRelayQueryRouteT8Advanced',
                               inputs=dict(prompt_relay_plan=['7', 0], query_route='joint_av_exp')):
        raise ValueError('An explicit existing joint-AV connection is required')
    chain_id = target.get('chain_id', '')
    if (not re.fullmatch(r'owned_voice_long_[0-9a-f]{16}', chain_id)
            or manifest.get('chain_id') != chain_id
            or manifest.get('schema') != 2
            or manifest.get('format') != 'minimax_h3_t8_accepted_manifest'
            or manifest.get('invalidated') != []):
        raise ValueError('Use only this non-invalidated owned manifest')
    segments = manifest.get('segments', [])
    if len(segments) != 1:
        raise ValueError('Exactly one mechanically accepted first segment is required')
    first = segments[0]
    expected_first = dict(index=0, frame_count=124, width=512, height=768, fps=24,
                          timeline_start_frame=0, timeline_end_frame=124,
                          is_final_segment=False, strict_decode_validated=True)
    if (any(first.get(key) != value for key, value in expected_first.items())
            or not first.get('sampling_summary', '').startswith('20-step dual_clock_euler/native_flow shift12/3;')):
        raise ValueError('Keep the original completed first segment and sampling policy')
    for key in ('video', 'context'):
        if not re.fullmatch(r'[0-9a-f]{64}', first.get(key + '_sha256', '')):
            raise ValueError('A full accepted source SHA256 is required')
    return copy.deepcopy(recipe), copy.deepcopy(first)
