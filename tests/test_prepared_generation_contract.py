"""CPU manifest and real local-file checks; no model or GPU qualification."""
from copy import deepcopy
import os
import subprocess

import pytest

from h3_audio_t8_pkg.prepared_generation_contract import validate_bundle, generation_request, validate_geometry
from h3_audio_t8_pkg import prepared_generation_contract as contract
from h3_audio_t8_pkg.prepared_generation_runtime import verify_assets
from h3_audio_t8_pkg.prepared_identity import inventory
from h3_audio_t8_pkg.prepared_backend.resource_guard import file_identity


@pytest.fixture
def bundle(tmp_path):
    files = {}
    for name in ('base', 'lora', 'inputs', 'text_cache', 'vae', 'original_video'):
        file = tmp_path / (name + '.fixture')
        file.write_bytes(b'Not model weights: ' + name.encode())
        files[name] = str(file)
    geometry = {'frames': 73, 'width': 2048, 'height': 1024, 'fps': 24}
    source = tmp_path / 'source'
    source.mkdir()
    return {'schema': 't8_prepared_generation_bundle_v1', 'kind': 'ltx_refine',
        'generation': {'schema': 't8-ltx-prepared-refinement-v1',
            **{k: files[k] for k in ('base', 'lora', 'inputs', 'text_cache')},
            'isolated_paths': [str(source / 'src')], 'prompt': 'A synthetic manifest test.',
            'geometry': geometry, 'normalization': 'normalized_ltx_av', 'reference_prefix_frames': 0},
        'decode': {'schema': 't8-ltx-refined-decode-v1', 'core': str(source),
            'geometry': deepcopy(geometry), **{k: files[k] for k in ('vae', 'original_video')}},
        'assets': [file_identity(p) for p in files.values()],
        'source_revisions': [{'path': str(source), 'revision': 'b' * 40}], 'trees': []}


def test_valid_manifest_builds_controller_bound_hashes(bundle):
    assert validate_bundle(bundle) is bundle
    request = generation_request(bundle, 8301, 'SYNTHETIC-GPU', {})
    assert request['seed'] == 8301 and len(request['model_identities']) == 2
    assert request['inputs_sha256'] == request['identities'][request['inputs']]
    assert request['text_cache_sha256'] == request['identities'][request['text_cache']]
    assert 'seed' not in bundle['generation']


@pytest.mark.parametrize('field', ['worker', 'command', 'python', 'script', 'latent', 'gpu_uuid', 'seed', 'identities', 'model_identities', 'other'])
def test_unknown_or_controller_bound_request_field_rejected(bundle, field):
    bundle['generation'][field] = 'not permitted'
    with pytest.raises(ValueError, match='request fields'):
        validate_bundle(bundle)


@pytest.mark.parametrize('field', ['base', 'lora', 'inputs', 'text_cache'])
def test_model_or_prepared_input_without_identity_rejected(bundle, field):
    bundle['assets'] = [a for a in bundle['assets'] if a['path'] != bundle['generation'][field]]
    with pytest.raises(ValueError, match='identity'):
        validate_bundle(bundle)


def test_actual_core_path_must_be_pinned(bundle, tmp_path):
    bundle['decode']['core'] = str(tmp_path / 'different_core')
    with pytest.raises(ValueError, match='Core'):
        validate_bundle(bundle)


def test_arbitrary_import_root_not_covered_by_unrelated_pin(bundle, tmp_path):
    bundle['generation']['isolated_paths'].append(str(tmp_path / 'unbound_runtime'))
    with pytest.raises(ValueError, match='Isolated imports'):
        validate_bundle(bundle)


def test_runtime_inventory_binds_actual_members(bundle, tmp_path):
    root = tmp_path / 'runtime'
    root.mkdir()
    module = root / 'package.py'
    module.write_text('VALUE = 1\n')
    bundle['generation']['isolated_paths'].append(str(root))
    bundle['trees'] = [{'path': str(root), 'files': [str(module)]}]
    with pytest.raises(ValueError, match='asset identities'):
        validate_bundle(bundle)
    bundle['assets'].append(file_identity(module))
    validate_bundle(bundle)
    bundle['trees'][0]['files'].append(bundle['assets'][0]['path'])
    with pytest.raises(ValueError, match='leaves'):
        validate_bundle(bundle)


@pytest.mark.parametrize('field,value', [('frames', 124), ('frames', 194), ('frames', 1),
    ('fps', 30), ('width', 2049), ('height', 1023), ('width', True), ('frames', 73.0), ('width', 8192)])
def test_invalid_geometry_is_rejected_before_imports(bundle, field, value):
    bundle['generation']['geometry'][field] = value
    bundle['decode']['geometry'] = deepcopy(bundle['generation']['geometry'])
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_decode_geometry_cannot_disagree(bundle):
    bundle['decode']['geometry']['width'] = 1024
    with pytest.raises(ValueError, match='agree'):
        validate_bundle(bundle)


@pytest.mark.parametrize('value', [False, 0.0, 8])
def test_reference_prefix_is_exact_zero_integer(bundle, value):
    bundle['generation']['reference_prefix_frames'] = value
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_geometry_envelope_is_import_free_and_does_not_pad():
    values = {'frames': 73, 'width': 2048, 'height': 1024, 'fps': 24}
    before = deepcopy(values)
    assert validate_geometry(values) == before and values == before


def test_real_directory_addition_is_detected_before_hashing(tmp_path):
    root = tmp_path / 'teacher'
    root.mkdir()
    member = root / 'request_00.fixture'
    member.write_bytes(b'CPU-only inventory fixture')
    bundle = {'trees': [{'path': str(root), 'files': inventory(root)}],
        'assets': [file_identity(member)], 'source_revisions': []}
    verify_assets(bundle, lambda: None)
    (root / 'new_config.json').write_text('{}')
    with pytest.raises(ValueError, match='inventory changed'):
        verify_assets(bundle, lambda: None)


def test_same_size_mtime_content_mutation_is_detected(tmp_path):
    member = tmp_path / 'weights.fixture'
    member.write_bytes(b'aaaa')
    identity = file_identity(member)
    bundle = {'trees': [], 'assets': [identity], 'source_revisions': []}
    member.write_bytes(b'bbbb')
    os.utime(member, ns=(identity['mtime_ns'], identity['mtime_ns']))
    with pytest.raises(ValueError, match='asset changed'):
        verify_assets(bundle, lambda: None)


def test_pyc_cache_does_not_change_directory_identity(tmp_path):
    (tmp_path / 'module.py').write_text('VALUE = 1\n')
    expected = inventory(tmp_path)
    (tmp_path / '__pycache__').mkdir()
    (tmp_path / '__pycache__/module.pyc').write_bytes(b'generated cache')
    assert inventory(tmp_path) == expected


def test_real_source_revision_and_tracked_changes_checked(tmp_path):
    source = tmp_path / 'git-source'
    source.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True, stderr=subprocess.STDOUT).strip()
    git('init', '-q')
    module = source / 'module.py'
    module.write_text('VALUE = 1\n')
    git('add', 'module.py')
    git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@invalid.local', 'commit', '-qm', 'fixture')
    bundle = {'trees': [], 'assets': [file_identity(module)],
        'source_revisions': [{'path': str(source), 'revision': git('rev-parse', 'HEAD')}]}
    verify_assets(bundle, lambda: None)
    module.write_text('VALUE = 2\n')
    bundle['assets'] = [file_identity(module)]
    with pytest.raises(ValueError, match='source revision or working tree'):
        verify_assets(bundle, lambda: None)


def test_empty_python_init_file_has_valid_identity(bundle, tmp_path):
    empty = tmp_path / '__init__.py'
    empty.touch()
    bundle['assets'].append(file_identity(empty))
    validate_bundle(bundle)


def test_dependency_change_invalidates_cache_without_loading_torch(bundle, monkeypatch):
    monkeypatch.setattr(contract, 'environment_identity', lambda: {'torch': 'A'})
    first = contract.fingerprint(bundle, 8301, {})
    monkeypatch.setattr(contract, 'environment_identity', lambda: {'torch': 'B'})
    assert contract.fingerprint(bundle, 8301, {}) != first
