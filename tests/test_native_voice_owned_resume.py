import copy

import pytest

from tools.native_voice_owned_resume import owned_resume_chain_root, prepare_owned_single_resume
from tools.native_voice_owned_resume import prepare_owned_window_resume


def test_chain_path_checked_before_manifest_access(tmp_path):
    name = 'owned_voice_long_' + 'a' * 16
    root = tmp_path / 'output/minimax_h3_t8_long_video' / name
    root.mkdir(parents=True)
    assert owned_resume_chain_root(tmp_path, name) == root
    for bad in ('../../private', 'user_chain', 'C:/private', name + '/nested'):
        with pytest.raises(ValueError):
            owned_resume_chain_root(tmp_path, bad)


def test_resolved_state_folder_escape_rejected_before_manifest_access(monkeypatch, tmp_path):
    from pathlib import Path
    resolve = Path.resolve
    def escaped(path, *, strict=False):
        if path.name == 'minimax_h3_t8_long_video':
            return tmp_path.parent / 'outside_probe'
        return resolve(path, strict=strict)
    monkeypatch.setattr(Path, 'resolve', escaped)
    with pytest.raises(ValueError):
        owned_resume_chain_root(tmp_path, 'owned_voice_long_' + 'a' * 16)


def fixture():
    target = dict(steps=20, shift_video=12., shift_audio=3.,
                  sampler_name='dual_clock_euler', scheduler='native_flow',
                  task_type='Ref2VA', audio_mode='native', total_duration_seconds=8.,
                  width=512, height=768, render_window_frames=124, context_frames=22,
                  resume_existing=True, model=['11', 0], prompt_relay_plan=['12', 0],
                  add_source_as_reference=False, prompt_primary_audio_ordinal=0,
                  chain_id='owned_voice_long_' + 'a' * 16,
                  **{'ref_images.ref_image_0': ['5', 0], 'ref_audios.ref_audio_0': ['6', 0]})
    recipe = {'8': dict(class_type='MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced', inputs=target),
              '12': dict(class_type='MiniMaxH3PromptRelayQueryRouteT8Advanced',
                         inputs=dict(prompt_relay_plan=['7', 0], query_route='joint_av_exp'))}
    first = dict(index=0, frame_count=124, width=512, height=768, fps=24,
                 timeline_start_frame=0, timeline_end_frame=124, is_final_segment=False,
                 strict_decode_validated=True, sampling_summary='20-step dual_clock_euler/native_flow shift12/3;',
                 video_sha256='b' * 64, context_sha256='c' * 64)
    manifest = dict(chain_id=target['chain_id'], schema=2,
                    format='minimax_h3_t8_accepted_manifest', invalidated=[], segments=[first])
    process = dict(status='failed', process=dict(status='timeout', active_after_cleanup=0))
    return recipe, manifest, process


def test_owned_resume_is_exact_copy_not_sampler_or_audio_reconfiguration():
    inputs = fixture()
    frozen = copy.deepcopy(inputs)
    recipe, first = prepare_owned_single_resume(*inputs)
    assert inputs == frozen
    assert recipe == inputs[0] and recipe is not inputs[0]
    assert first == inputs[1]['segments'][0] and first is not inputs[1]['segments'][0]


@pytest.mark.parametrize('fault', ['live', 'other_failure', 'second_segment', 'bad_hash',
                                  'foreign_chain', 'changed_steps', 'source_mux', 'route', 'invalidated'])
def test_resume_refuses_unknown_or_changed_evidence_without_mutation(fault):
    recipe, manifest, process = fixture()
    if fault == 'live':
        process['process']['active_after_cleanup'] = 1
    elif fault == 'other_failure':
        process['process']['status'] = 'child_failed'
    elif fault == 'second_segment':
        manifest['segments'].append(copy.deepcopy(manifest['segments'][0]))
    elif fault == 'bad_hash':
        manifest['segments'][0]['video_sha256'] = 'abc'
    elif fault == 'foreign_chain':
        recipe['8']['inputs']['chain_id'] = 'user_chain'
    elif fault == 'changed_steps':
        recipe['8']['inputs']['steps'] = 8
    elif fault == 'source_mux':
        recipe['8']['inputs']['final_audio'] = ['6', 0]
    elif fault == 'route':
        recipe['12']['inputs']['query_route'] = 'video_only_paper'
    else:
        manifest['invalidated'].append({'index': 0})
    frozen = copy.deepcopy((recipe, manifest, process))
    with pytest.raises(ValueError):
        prepare_owned_single_resume(recipe, manifest, process)
    assert (recipe, manifest, process) == frozen


@pytest.mark.parametrize('fault', [None, 'live', 'unexpected_stop', 'too_many_calls', 'changed_owner'])
def test_controlled_window_resume_requires_real_first_segment_and_cleaned_job(fault):
    recipe, manifest, _ = fixture()
    recipe['8']['inputs']['prompt_relay_plan'] = ['13', 0]
    recipe['13'] = dict(class_type='MiniMaxH3PromptRelayWindowTextEXPT8',
                         inputs=dict(prompt_relay_plan=['12', 0], text_policy='accepted_window_text_exp'))
    process = dict(status='owned_worker_exit0_cleanup0',
                   process=dict(status='complete', exit_code=0, active_after_cleanup=0))
    terminal = dict(status='controlled_native_voice_window_first_committed_cancelled_not_complete8s',
                    actual_forwards=20, accepted_count=1, sources_unchanged=True,
                    window_text_policy='accepted_window_text_exp')
    if fault == 'live':
        process['process']['active_after_cleanup'] = 1
    elif fault == 'unexpected_stop':
        terminal['status'] = 'failed'
    elif fault == 'too_many_calls':
        terminal['actual_forwards'] = 35
    elif fault == 'changed_owner':
        recipe['13']['inputs']['prompt_relay_plan'] = ['foreign', 0]
    frozen = copy.deepcopy((recipe, manifest, process, terminal))
    if fault:
        with pytest.raises(ValueError):
            prepare_owned_window_resume(recipe, manifest, process, terminal)
    else:
        result, first = prepare_owned_window_resume(recipe, manifest, process, terminal)
        assert result == recipe and result is not recipe and first == manifest['segments'][0]
    assert (recipe, manifest, process, terminal) == frozen


@pytest.mark.parametrize('fault', [None, 'different_policy', 'legacy_receipt', 'unknown'])
def test_dialogue_owner_resume_does_not_reuse_another_experiment(fault):
    recipe, manifest, _ = fixture()
    recipe['8']['inputs']['prompt_relay_plan'] = ['13', 0]
    recipe['13'] = dict(class_type='MiniMaxH3PromptRelayWindowTextEXPT8',
                       inputs=dict(prompt_relay_plan=['12', 0], text_policy='dialogue_start_owner_exp'))
    process = dict(status='owned_worker_exit0_cleanup0',
                   process=dict(status='complete', exit_code=0, active_after_cleanup=0))
    terminal = dict(status='controlled_native_voice_window_first_committed_cancelled_not_complete8s',
                    actual_forwards=20, accepted_count=1, sources_unchanged=True,
                    window_text_policy='dialogue_start_owner_exp')
    if fault == 'different_policy':
        recipe['13']['inputs']['text_policy'] = 'accepted_window_text_exp'
    elif fault == 'legacy_receipt':
        terminal['window_text_policy'] = 'accepted_window_text_exp'
    policy = 'unknown' if fault == 'unknown' else 'dialogue_start_owner_exp'
    frozen = copy.deepcopy((recipe, manifest, process, terminal))
    if fault:
        with pytest.raises(ValueError):
            prepare_owned_window_resume(recipe, manifest, process, terminal, text_policy=policy)
    else:
        result, first = prepare_owned_window_resume(recipe, manifest, process, terminal, text_policy=policy)
        assert result == recipe and result is not recipe and first == manifest['segments'][0]
        with pytest.raises(ValueError):
            prepare_owned_window_resume(recipe, manifest, process, terminal)
    assert (recipe, manifest, process, terminal) == frozen
