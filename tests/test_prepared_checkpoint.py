"""Synthetic evidence tests, separate from actual pilot migration tool runs."""
from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from h3_audio_t8_pkg import prepared_checkpoint as checkpoint
from h3_audio_t8_pkg.prepared_generation_contract import BACKEND
from tests.test_prepared_generation_contract import bundle  # noqa: F401


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


@pytest.fixture
def case(bundle, tmp_path):  # noqa: F811
    gen_dir, dec_dir, media_dir = [tmp_path / name for name in ('generation', 'decode', 'recovered')]
    for directory in (gen_dir, dec_dir, media_dir):
        directory.mkdir()
    for name, revision, suffixes in [('LTX', 'd151147788a9284cca791edc6ce898007e727fe6',
            ['packages/ltx-core/src', 'packages/ltx-pipelines/src']),
            ('Sana', '144085566a866f9784f3798d4c8d1603f3adbccf', ['models/minimax_h3/Sol-H3-Spark'])]:
        root = tmp_path / name
        for suffix in suffixes:
            (root / suffix).mkdir(parents=True)
        bundle['source_revisions'].append({'path': str(root), 'revision': revision})
    bundle['generation']['isolated_paths'] = [str(tmp_path / 'LTX/packages/ltx-core/src'),
        str(tmp_path / 'LTX/packages/ltx-pipelines/src'), str(tmp_path / 'Sana/models/minimax_h3/Sol-H3-Spark')]
    bundle['generation']['prompt'] = '4K, refined, high quality, cinematic detail, clean textures, natural motion.'
    old_gen = deepcopy(bundle['generation'])
    old_gen.update(schema='t8-ltx-real-joint-refinement-v1', seed=8301)
    old_gen['identities'] = {a['path']: a['sha256'] for a in bundle['assets']}
    old_gen['model_identities'] = [a for a in bundle['assets'] if a['path'] in (old_gen['base'], old_gen['lora'])]
    for name in ['ltx_dequantized_lora.py', 'ltx_lora_contract.py', 'ltx_gpu_fusion_gate.py',
            'ltx_text_weight_offload.py', 'taomate_weight_offload.py']:
        old_gen['identities'][str(BACKEND / name)] = checkpoint.sha(BACKEND / name)
    latent = gen_dir / 'refined-latents.safetensors'
    # Deliberately header-only synthetic data, not a qualified real tensor.
    data = json.dumps({'video': {'shape': [1, 128, 10, 32, 64], 'dtype': 'BF16', 'data_offsets': [0, 4]}}).encode()
    latent.write_bytes(struct.pack('<Q', len(data)) + data + b'fake')
    rgb = dec_dir / 'refined-rgb.safetensors'
    rgb.write_bytes(b'synthetic RGB source')
    movie = media_dir / 'h3-learned2x-ltx-refined-original-audio.mp4'
    movie.write_bytes(b'synthetic media, not decode evidence')
    old_dec = deepcopy(bundle['decode'])
    old_dec.update(latent=str(latent), identities={a['path']: a['sha256'] for a in bundle['assets']})
    generated = {'status': 'real_three_update_LTX_joint_refinement_latents_pass', 'seed': 8301,
        'sigmas': [.909375, .725, .421875, 0], 'lora_strength': .8, 'joint_forwards': 3, 'block_calls': 144,
        'output_sha256': checkpoint.sha(latent), 'video_shape': [1, 128, 10, 32, 64]}
    media = {'status': 'real_LTX_refined_media_pass_CPU_reencode_pending_review',
        'source_decode_report': str(dec_dir / 'report.json'), 'GPU_rerun': False,
        'media': {'full_decode_pass': True, 'audio_packets_and_timestamps_exact': True},
        'rgb_sha256': checkpoint.sha(rgb), 'video_sha256': checkpoint.sha(movie)}
    terminal = {'status': 'LTX_joint_refinement_latents_complete_decode_pending',
        'cleanup': {'exit_code': 0, 'owned_children_remaining': []}}
    write(gen_dir / 'request.json', old_gen)
    write(gen_dir / 'report.json', generated)
    write(gen_dir / 'terminal.json', terminal)
    write(dec_dir / 'request.json', old_dec)
    write(dec_dir / 'report.json', {'status': 'failed', 'error': 'synthetic old MP4 failure'})
    write(media_dir / 'report.json', media)
    return bundle, gen_dir, dec_dir, media_dir


def migrate(case, seed=8301):
    prepared, gen, dec, media = case
    return checkpoint.migration(prepared, gen, dec, seed=seed, media_directory=media)


def test_synthetic_provenance_is_explicit_not_new_gpu(case):
    migrated, report = migrate(case)
    assert report['GPU_rerun'] is False and report['new_wrapper_GPU_qualified'] is False
    assert len(report['math_helpers_identical']) == 5
    assert set(migrated['checkpoint']['stages']) == {'generation', 'decode'}
    assert report['human_review'] == 'pending' and 'checkpoint' not in case[0]


def test_wrong_seed_rejected(case):
    with pytest.raises(ValueError, match='seed/sampling'):
        migrate(case, seed=8302)


@pytest.mark.parametrize('field,value', [('sigmas', [.9, .7, 0]), ('lora_strength', .7),
    ('joint_forwards', 2), ('block_calls', 100), ('status', 'failed')])
def test_original_generation_must_have_completed_expected_math(case, field, value):
    path = case[1] / 'report.json'
    report = checkpoint.read_json(path)
    report[field] = value
    write(path, report)
    with pytest.raises(ValueError):
        migrate(case)


@pytest.mark.parametrize('field,value', [('exit_code', 1), ('owned_children_remaining', [12345])])
def test_unclean_old_job_not_qualified(case, field, value):
    path = case[1] / 'terminal.json'
    report = checkpoint.read_json(path)
    report['cleanup'][field] = value
    write(path, report)
    with pytest.raises(ValueError, match='clean up'):
        migrate(case)


@pytest.mark.parametrize('field', ['inputs', 'text_cache', 'base', 'lora'])
def test_changed_prepared_input_or_model_identity_rejected(case, field):
    path = case[0]['generation'][field]
    for asset in case[0]['assets']:
        if asset['path'] == path:
            asset['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash differs|identity differs'):
        migrate(case)


def test_actual_import_root_must_have_the_original_pin(case):
    for pin in case[0]['source_revisions']:
        if Path(pin['path']).name == 'LTX':
            pin['revision'] = 'd' * 40
    with pytest.raises(ValueError, match='actual imported source'):
        migrate(case)


@pytest.mark.parametrize('target', ['latent', 'movie', 'rgb'])
def test_corrupted_artifact_is_never_migrated(case, target):
    path = {'latent': case[1] / 'refined-latents.safetensors',
        'movie': case[3] / 'h3-learned2x-ltx-refined-original-audio.mp4',
        'rgb': case[2] / 'refined-rgb.safetensors'}[target]
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        migrate(case)


def test_failed_original_mp4_is_not_the_recovered_movie(case):
    prepared, gen, dec, _ = case
    with pytest.raises(ValueError, match='CPU-recovered'):
        checkpoint.migration(prepared, gen, dec, seed=8301)


def test_wrong_recovery_audio_policy_rejected(case):
    path = case[3] / 'report.json'
    report = checkpoint.read_json(path)
    report['media']['audio_packets_and_timestamps_exact'] = False
    write(path, report)
    with pytest.raises(ValueError, match='media audit'):
        migrate(case)


def test_wrong_packaged_math_helper_is_rejected(case):
    path = case[1] / 'request.json'
    request = checkpoint.read_json(path)
    key = str(BACKEND / 'ltx_gpu_fusion_gate.py')
    request['identities'][key] = '0' * 64
    write(path, request)
    with pytest.raises(ValueError, match='math helper'):
        migrate(case)
