"""Single-window text-only diagnostic guards, not voice-quality certification."""
import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.core import nested_av_parts
from h3_audio_t8_pkg.long_video import LONG_VIDEO_SCHEMA, MOTION_AUDIO_END_FRAME
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan, configure_prompt_relay_query_route, _validate_plan, _sha256_json
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import configure_long_video_window_text, project_prompt_relay_plan_to_long_video_window, build_prompt_relay_long_video_conditioning
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip, _model_patcher, _allow_fixture_core_contract
from tools.qualify_native_voice_continuation_text_gpu import BEFORE, AFTER, validate_parent, visual_only_crossing_text


def projected_plan():
    plan = build_prompt_relay_plan(
        global_prompt='S1 uses <Audio 1> timbre, not target words. Clear newly generated Mandarin dialogue.',
        local_prompts='S1 calmly says <d>[Mandarin]你终于回来了。</d>\n'
            'S1 speaks gently with relief and restrained joy: <d>[Mandarin]这次别再走了。</d>\n'
            'S1 listens silently with a small smile. No additional speech.',
        length=193, timing_mode='percent', time_ranges='0-35\n35-70\n70-100',
        math_profile='paper_v1', epsilon=.1, allow_gaps=False, allow_overlaps=False)[0]
    plan = configure_prompt_relay_query_route(plan, 'joint_av_exp')[0]
    plan = configure_long_video_window_text(plan, 'dialogue_start_owner_exp')[0]
    return project_prompt_relay_plan_to_long_video_window(plan, 1, 90, 22, 124/24, 192/24)[0]


def test_actual_owner_projection_retains_positive_speech_despite_removing_literal():
    before = projected_plan()
    assert before['events'][0]['local_prompt'] == BEFORE
    assert 'speaks gently' in before['compiled_prompt'] and '<d>' not in before['compiled_prompt']
    # The rejected language hypothesis is independently evidenced, not inferred.
    assert 'Clear newly generated Mandarin dialogue' in before['compiled_prompt']
    assert [e['event_index'] for e in before['events']] == [2, 3]
    assert before['events'][0]['global_start_frame'] == 73
    assert before['events'][0]['global_end_frame_exclusive'] == 146


def test_single_intervention_changes_only_known_local_text_spans_and_authentication():
    before = projected_plan()
    frozen = copy.deepcopy(before)
    after = visual_only_crossing_text(before, _validate_plan, _sha256_json)
    assert before == frozen
    assert before['plan_hash'] != after['plan_hash']
    excluded = {'events', 'compiled_prompt', 'plan_hash'}
    assert {k:v for k,v in before.items() if k not in excluded} == {k:v for k,v in after.items() if k not in excluded}
    for a, b in zip(after['events'], before['events']):
        text_keys = {'local_prompt', 'prompt_char_start', 'prompt_char_end'}
        assert {k:v for k,v in a.items() if k not in text_keys} == {k:v for k,v in b.items() if k not in text_keys}
        assert after['compiled_prompt'][a['prompt_char_start']:a['prompt_char_end']] == f"Event {a['event_index']}: {a['local_prompt']}"
    assert after['events'][0]['local_prompt'] == AFTER
    assert after['events'][1]['local_prompt'] == before['events'][1]['local_prompt']
    assert 'speaks gently' not in after['compiled_prompt'] and '<d>' not in after['compiled_prompt']


@pytest.mark.parametrize('fault', ['wrong_hash', 'other_policy', 'other_route', 'new_dialogue_owner', 'different_window', 'foreign_local', 'missing_language', 'duplicate_event'])
def test_intervention_rejects_other_recipes_and_never_mutates_them(fault):
    source = projected_plan()
    if fault == 'wrong_hash':
        source['plan_hash'] = '0' * 64
    elif fault == 'other_policy':
        source['long_video_window_text_policy'] = 'accepted_window_text_exp'
    elif fault == 'other_route':
        source['query_route'] = 'video_only_paper'
    elif fault == 'new_dialogue_owner':
        source['long_video_projection']['dialogue_owner_event_indices'] = [2]
    elif fault == 'different_window':
        source['long_video_projection']['accepted_start_frame'] = 226
    elif fault == 'foreign_local':
        source['events'][0]['local_prompt'] = 'User-supplied instructions must not be overwritten.'
    elif fault == 'missing_language':
        source['global_prompt'] = 'Other global prompt.'
    else:
        source['events'].append(copy.deepcopy(source['events'][0]))
    if fault != 'wrong_hash':
        source.pop('plan_hash')
        source['plan_hash'] = _sha256_json(source)
    frozen = copy.deepcopy(source)
    with pytest.raises(ValueError):
        visual_only_crossing_text(source, _validate_plan, _sha256_json)
    assert source == frozen


def parent_receipts():
    entry = {'video_sha256': 'a' * 64, 'context_sha256': 'b' * 64}
    shared = dict(window_text_policy='dialogue_start_owner_exp', sources_unchanged=True,
                  production_source_sha256={'h3_t8/fixture.py': 'c' * 64})
    first = dict(**shared, status='controlled_native_voice_window_first_committed_cancelled_not_complete8s',
                 actual_forwards=20, accepted_count=1, first_segment=entry)
    full = dict(**shared, status='actual_native_reference_single_model_two_segment20_8s_complete_not_human',
                actual_forwards=20, all_jobs_actual_forwards=40, completed_segments=2,
                discarded_timeout_partial_forwards=0, reused_first_segment=entry,
                reused_first_segment_bytes_unchanged=True, assets_and_Core_unchanged=True)
    processes = [dict(status='owned_worker_exit0_cleanup0',
        process=dict(status='complete', exit_code=0, active_after_cleanup=0, job_assigned_before_task=True)) for _ in (1, 2)]
    return first, full, processes


def test_parent_terminal_guards_do_not_adopt_the_first_into_a_new_chain():
    first, full, processes = parent_receipts()
    entry = validate_parent(first, full, processes)
    assert entry == first['first_segment'] and entry is not first['first_segment']
    entry['video_sha256'] = 'changed'
    assert first['first_segment']['video_sha256'] == 'a' * 64


@pytest.mark.parametrize('fault', ['live', 'unclean', 'no_job', 'partial', 'changed_prefix', 'other_policy', 'changed_source', 'failed'])
def test_parent_rejects_nonterminal_partial_foreign_or_mutated_evidence(fault):
    first, full, processes = parent_receipts()
    if fault == 'live':
        processes[1]['process']['status'] = 'running'
    elif fault == 'unclean':
        processes[1]['process']['active_after_cleanup'] = 1
    elif fault == 'no_job':
        processes[0]['process']['job_assigned_before_task'] = False
    elif fault == 'partial':
        full['all_jobs_actual_forwards'] = 35
    elif fault == 'changed_prefix':
        full['reused_first_segment'] = {'video_sha256': 'd' * 64}
    elif fault == 'other_policy':
        full['window_text_policy'] = 'accepted_window_text_exp'
    elif fault == 'changed_source':
        full['production_source_sha256'] = {}
    else:
        full['status'] = 'failed'
    with pytest.raises(ValueError):
        validate_parent(first, full, processes)


def test_real_native_condition_builder_retains_reference_and_known_AV_context(monkeypatch):
    _allow_fixture_core_contract(monkeypatch)
    reference = make_audio(2)
    reference_before = reference['waveform'].clone()
    context = {'schema': LONG_VIDEO_SCHEMA, 'empty': False,
        'video_tail': torch.full((1,24,12,8,8), .03125), 'audio_tail': torch.full((1,32,2,65), .125),
        'metadata': {'source_segment_index':0, 'target_segment_index':1, 'max_context_frames':39, 'audio_overhang':1/3}}
    tails_before = {k: context[k].clone() for k in ('video_tail', 'audio_tail')}
    before = projected_plan()
    after = visual_only_crossing_text(before, _validate_plan, _sha256_json)
    results = []
    for plan in (before, after):
        results.append(build_prompt_relay_long_video_conditioning(
            model=_model_patcher(), clip=NativeLikeFakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
            context=context, prompt_relay_plan=plan, segment_index=1, context_frames=22,
            context_audio='video_and_audio', width=128, height=128, length=90, task_type='Ref2VA',
            audio_mode='native', audio_denoise_strength=.35, add_source_as_reference=False,
            prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size='match',
            reference_video_policy='official_2_to_15s', execution_mode='apply_exp', query_chunk_rows=64,
            ref_images={'ref_image_0': torch.zeros(1,128,128,3)}, ref_audios={'ref_audio_0': reference}))
    for result in results:
        _, positive, latent, mux, _, mapping, report = result
        assert mux is None and json.loads(mapping)['audios'] == {'1': 'ref_audio_1'}
        assert json.loads(report)['context_frames'] == 22
        refs = positive[0][1]['minimax_refs']
        assert any(r['kind'] == 'audio' and r['ref_audio_t'] == 80 for r in refs)
        motion = next(r for r in refs if MOTION_AUDIO_END_FRAME in r)
        assert torch.all(motion['audio_latent'] == .125)
    for a, b in zip(nested_av_parts(results[0][2]), nested_av_parts(results[1][2])):
        assert torch.equal(a, b)
    assert torch.equal(reference['waveform'], reference_before)
    assert all(torch.equal(context[k], v) for k, v in tails_before.items())


def test_worker_does_not_import_GPU_at_module_scope_or_submit_user_queue():
    path = Path(__file__).resolve().parents[1] / 'tools/qualify_native_voice_continuation_text_gpu.py'
    parsed = ast.parse(path.read_text(encoding='utf8'))
    imports = [n for n in parsed.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(not (getattr(n, 'module', '') or '').startswith(('comfy', 'torch')) for n in imports)
    assert all(not a.name.startswith(('comfy', 'torch')) for n in imports for a in n.names)
    source = path.read_text(encoding='utf8')
    assert 'SerialProbeLease' in source and '@wraps(original)' in source
    assert 'interrupt_current_processing' not in source and 'accept_long_video_candidate(' not in source
    assert 'compose_accepted_long_video(' not in source and 'source_audio_mux=False' in source
    for function in ('_bind_loop_relay_conditioning', '_sample_prepared_segment', 'decode_av_latent', 'setup_dual_clock_sampling'):
        assert f'torch.inference_mode()({function})' in source
