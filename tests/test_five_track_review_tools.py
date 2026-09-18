import ast
from types import SimpleNamespace
import json

import pytest
from PIL import Image

from tools import build_candidate_combined_review as builder
from tools.build_five_track_review import review_manifest_name, validate_single_model_terminal, voice_review_configuration
from tools.build_five_track_review import validate_window_text_terminal, validate_window_words
from tools.build_five_track_review import validate_dialogue_owner_terminal


def test_review_v3_requires_explicit_bound_cohort_and_preserves_prior_names():
    assert review_manifest_name(None, None) == 'five-track-review-input-v1.json'
    assert review_manifest_name('native-voice-ablation-media-audit-v1', None) == 'five-track-review-input-v2.json'
    assert review_manifest_name('native-voice-ablation-media-audit-v1', 'native-voice-dual-joint-media-audit-v1') == 'five-track-review-input-v3.json'
    assert review_manifest_name('native-voice-ablation-media-audit-v1', 'native-voice-dual-joint-media-audit-v2') == 'five-track-review-input-v3.json'
    with pytest.raises(ValueError):
        review_manifest_name(None, 'native-voice-dual-joint-media-audit-v1')
    with pytest.raises(ValueError):
        review_manifest_name('native-voice-ablation-media-audit-v1', '../unbound')


def test_original_review_configuration_remains_immutable():
    manifest, clips = voice_review_configuration(None)
    assert manifest == 'five-track-review-input-v1.json'
    assert len(clips) == 2 and clips[0][1] == 'voice_neutral_00001_'


def test_single_model_review_uses_fresh_bound_v4_and_never_prior_manifest():
    assert review_manifest_name('native-voice-ablation-media-audit-v1',
                                'native-voice-dual-joint-media-audit-v2',
                                'native-voice-long-media-audit-v1') == 'five-track-review-input-v4.json'
    for ablation, joint, single in [(None, None, 'native-voice-long-media-audit-v1'),
                                    ('native-voice-ablation-media-audit-v1', None, 'native-voice-long-media-audit-v1'),
                                    ('native-voice-ablation-media-audit-v1', 'native-voice-dual-joint-media-audit-v2', '../unbound')]:
        with pytest.raises(ValueError):
            review_manifest_name(ablation, joint, single)


def test_single_model_review_keeps_timeout_and_discarded_compute_not_false40_pass():
    receipt = dict(status='actual_native_reference_single_model_two_segment20_8s_complete_not_human',
                   single_model=True, query_route='joint_av_exp', completed_segments=2,
                   human_qualified=False, sources_unchanged=True, actual_forwards=40)
    assert validate_single_model_terminal(receipt) is False
    receipt.update(actual_forwards=20, resumed_accepted_segments_before=1,
                   reused_first_segment_bytes_unchanged=True, assets_and_Core_unchanged=True,
                   prior_job_actual_forwards=35, all_jobs_actual_forwards=55,
                   discarded_timeout_partial_forwards=15)
    assert validate_single_model_terminal(receipt) is True
    receipt['all_jobs_actual_forwards'] = 40
    with pytest.raises(ValueError):
        validate_single_model_terminal(receipt)
    receipt['all_jobs_actual_forwards'] = 55
    receipt['reused_first_segment_bytes_unchanged'] = False
    with pytest.raises(ValueError):
        validate_single_model_terminal(receipt)


def test_reference_ablation_is_new_bound_review_not_overwrite():
    manifest, clips = voice_review_configuration('native-voice-ablation-media-audit-v1')
    assert manifest == 'five-track-review-input-v2.json'
    assert len(clips) == 3
    assert [clip[1] for clip in clips] == ['voice_reference_with_00001_', 'voice_emotion_00001_', 'voice_reference_without_00001_']
    with pytest.raises(ValueError):
        voice_review_configuration('../unbound')


def test_window_text_cohort_is_fresh_v5_and_keeps_old_v4_immutable():
    args = ('native-voice-ablation-media-audit-v1', 'native-voice-dual-joint-media-audit-v2',
            'native-voice-long-media-audit-v1')
    assert review_manifest_name(*args) == 'five-track-review-input-v4.json'
    assert review_manifest_name(*args, 'native-voice-window-media-audit-v1') == 'five-track-review-input-v5.json'
    with pytest.raises(ValueError):
        review_manifest_name(*args, '../foreign')
    with pytest.raises(ValueError):
        review_manifest_name(None, None, None, 'native-voice-window-media-audit-v1')


@pytest.mark.parametrize('fault', [None, 'owner', 'missing_window', 'missing_long', 'missing_joint', 'missing_ablation'])
def test_dialogue_owner_requires_fresh_explicit_v6_not_inheriting_window_cohort(fault):
    args = ['native-voice-ablation-media-audit-v1', 'native-voice-dual-joint-media-audit-v2',
            'native-voice-long-media-audit-v1', 'native-voice-window-media-audit-v1',
            'native-voice-dialogue-owner-media-audit-v1']
    if fault == 'owner':
        args[-1] = '../unknown'
    elif fault:
        args[{'missing_window':3, 'missing_long':2, 'missing_joint':1, 'missing_ablation':0}[fault]] = None
    if fault:
        with pytest.raises(ValueError):
            review_manifest_name(*args)
    else:
        assert review_manifest_name(*args) == 'five-track-review-input-v6.json'
        assert review_manifest_name(*args[:-1]) == 'five-track-review-input-v5.json'


@pytest.mark.parametrize('fault', [None, 'policy', 'all_calls', 'controlled', 'first_bytes', 'Core', 'partial'])
def test_dialogue_owner_review_has_own_policy_and_real_complete20_plus20_evidence(fault):
    receipt = dict(status='actual_native_reference_single_model_two_segment20_8s_complete_not_human',
                   single_model=True, query_route='joint_av_exp', completed_segments=2,
                   human_qualified=False, sources_unchanged=True, actual_forwards=20,
                   resumed_accepted_segments_before=1, reused_first_segment_bytes_unchanged=True,
                   assets_and_Core_unchanged=True, prior_job_actual_forwards=20,
                   all_jobs_actual_forwards=40, discarded_timeout_partial_forwards=0,
                   prior_job_controlled_stop=True, window_text_policy='dialogue_start_owner_exp')
    for key, value in {'policy':('window_text_policy', 'accepted_window_text_exp'),
                       'all_calls':('all_jobs_actual_forwards', 55),
                       'controlled':('prior_job_controlled_stop', False),
                       'first_bytes':('reused_first_segment_bytes_unchanged', False),
                       'Core':('assets_and_Core_unchanged', False),
                       'partial':('completed_segments', 1)}.items():
        if fault == key:
            receipt[value[0]] = value[1]
    if fault:
        with pytest.raises(ValueError):
            validate_dialogue_owner_terminal(receipt)
    else:
        validate_dialogue_owner_terminal(receipt)
        with pytest.raises(ValueError):
            validate_window_text_terminal(receipt)


@pytest.mark.parametrize('fault', [None, 'policy', 'timeout', 'changed_first', 'not_controlled', 'partial_only'])
def test_window_review_requires_controlled_full_av_evidence(fault):
    row = dict(status='actual_native_reference_single_model_two_segment20_8s_complete_not_human',
               single_model=True, query_route='joint_av_exp', completed_segments=2,
               human_qualified=False, sources_unchanged=True, actual_forwards=20,
               resumed_accepted_segments_before=1, reused_first_segment_bytes_unchanged=True,
               assets_and_Core_unchanged=True, prior_job_actual_forwards=20, all_jobs_actual_forwards=40,
               discarded_timeout_partial_forwards=0, prior_job_controlled_stop=True,
               window_text_policy='accepted_window_text_exp')
    if fault == 'policy':
        row['window_text_policy'] = 'preserve_all'
    elif fault == 'timeout':
        row.update(prior_job_actual_forwards=35, all_jobs_actual_forwards=55, discarded_timeout_partial_forwards=15)
    elif fault == 'changed_first':
        row['reused_first_segment_bytes_unchanged'] = False
    elif fault == 'not_controlled':
        row['prior_job_controlled_stop'] = False
    elif fault == 'partial_only':
        row['completed_segments'] = 1
    if fault:
        with pytest.raises(ValueError):
            validate_window_text_terminal(row)
    else:
        validate_window_text_terminal(row)


@pytest.mark.parametrize('fault', [None, 'sha', 'hint', 'human', 'unfinished', 'ambiguous'])
def test_window_words_are_bound_observations_not_quality_acceptance(fault):
    row = dict(status='complete_offline_CPU_ASR_observations_not_human_acceptance',
               input_and_model_bytes_unchanged=True, human_qualified=False, GPU_used=False,
               configuration=dict(initial_prompt=None, prefix=None, hotwords=None),
               media=[dict(sha256='a'*64, reference_recording=False,
                           observations=[dict(text='observed auto'), dict(text='observed zh')])])
    if fault == 'sha':
        row['media'][0]['sha256'] = 'b'*64
    elif fault == 'hint':
        row['configuration']['initial_prompt'] = 'target sentence'
    elif fault == 'human':
        row['human_qualified'] = True
    elif fault == 'unfinished':
        row['status'] = 'incomplete'
    elif fault == 'ambiguous':
        row['media'].append(row['media'][0].copy())
    if fault:
        with pytest.raises(ValueError):
            validate_window_words(row, 'a'*64)
    else:
        assert validate_window_words(row, 'a'*64) == ['observed auto', 'observed zh']


@pytest.mark.parametrize('name', ['qualify_native_voice_dual_gpu.py', 'qualify_native_voice_gpu.py',
                                 'qualify_avatar_gpu.py', 'diagnose_taeh3_gpu.py'])
def test_native_forward_counters_preserve_introspectable_signature(name):
    tree = ast.parse((builder.PROJECT / 'tools' / name).read_text(encoding='utf8'))
    counters = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'counted']
    assert len(counters) == 1
    assert any(isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == 'wraps'
               for dec in counters[0].decorator_list)


def fixture(monkeypatch, tmp_path):
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    media, poster, receipt = artifacts / 'input.mp4', artifacts / 'poster.png', artifacts / 'audit.json'
    media.write_bytes(b'audited fixture video')
    Image.new('RGB', (32, 48)).save(poster)
    receipt.write_text(json.dumps({'sha256': builder.digest(media)}))
    clip = dict(label='Fixture', path='artifacts/input.mp4', sha256=builder.digest(media),
                audit='artifacts/audit.json', audit_sha256=builder.digest(receipt),
                poster=dict(path='artifacts/poster.png', sha256=builder.digest(poster)))
    manifest = artifacts / 'input.json'
    manifest.write_text(json.dumps({'groups': [dict(id='one', title='<untrusted>', note='fixture', clips=[clip])]}))
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'five_track_review.html').write_text('<script>MANIFEST_JSON</script>')
    monkeypatch.setattr(builder, 'PROJECT', tmp_path)
    monkeypatch.setattr(builder.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=json.dumps({
        'streams': [dict(codec_type='video', codec_name='h264', pix_fmt='yuv420p',
                         width=32, height=48, duration='3.0')]})))
    return manifest, poster, artifacts / 'output'


def test_bound_real_frame_copy_and_safe_inline_data(monkeypatch, tmp_path):
    manifest, poster, output = fixture(monkeypatch, tmp_path)
    result = builder.build(manifest, output, template_name='five_track_review.html')
    clip = result['groups'][0]['clips'][0]
    assert builder.digest(output / 'public' / clip['poster']['url']) == builder.digest(poster)
    assert '<untrusted>' not in (output / 'public/review.html').read_text()


@pytest.mark.parametrize('failure', ['changed', 'wrong_dimensions', 'template_traversal'])
def test_invalid_bound_review_frame_rejected(monkeypatch, tmp_path, failure):
    manifest, poster, output = fixture(monkeypatch, tmp_path)
    template = 'five_track_review.html'
    if failure == 'changed':
        poster.write_bytes(b'changed')
    elif failure == 'wrong_dimensions':
        Image.new('RGB', (48, 32)).save(poster)
        value = json.loads(manifest.read_text())
        value['groups'][0]['clips'][0]['poster']['sha256'] = builder.digest(poster)
        manifest.write_text(json.dumps(value))
    else:
        template = '../private.html'
    with pytest.raises(ValueError):
        builder.build(manifest, output, template_name=template)
    assert not output.exists()
