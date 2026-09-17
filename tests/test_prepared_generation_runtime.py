"""Synthetic cache/state tests. These do not qualify model generation or media."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg import prepared_generation_runtime as runtime
from h3_audio_t8_pkg.prepared_generation_contract import fingerprint
from h3_audio_t8_pkg.prepared_backend.resource_guard import file_identity


@pytest.fixture
def setup(tmp_path, monkeypatch):
    asset = tmp_path / 'prepared-fixture.bin'
    asset.write_bytes(b'synthetic prepared input')
    trees, assets = [], [file_identity(asset)]
    for relative in ('base/FL2VA/transformer', 'adapter', 'teacher'):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        member = directory / 'fixture.bin'
        member.write_bytes(b'synthetic directory member')
        trees.append({'path': str(directory), 'files': [str(member)]})
        assets.append(file_identity(member))
    bundle = {'schema': 't8_prepared_generation_bundle_v1', 'kind': 'tao5s',
        'generation': {'schema': 't8-taomate-prepared-first-request-v1', 'source': str(tmp_path),
            'source_revision': 'a' * 40, 'audio_seed': 8301, 'base': str(tmp_path / 'base'),
            'adapter': str(tmp_path / 'adapter'), 'teacher': str(tmp_path / 'teacher'),
            **{key: str(asset) for key in ('milestones', 'text_features', 'download_receipt', 'cpu_receipt')}},
        'decode': {'schema': 't8-taomate-video-decode-v1', 'source': str(tmp_path),
            'milestones': str(asset), 'audio': str(asset), 'vae': str(asset), 'core': str(tmp_path)},
        'assets': assets, 'source_revisions': [{'path': str(tmp_path), 'revision': 'a' * 40}], 'trees': trees}
    monkeypatch.setattr(runtime, 'engine_sources', lambda: {'synthetic_engine': 'b' * 64})
    # Contract CPU tests below check actual identities separately. Here isolate
    # state transitions; no source repository, NVML or GPU worker is fabricated
    # as evidence of real generation.
    monkeypatch.setattr(runtime, 'verify_assets', lambda *args: None)
    class Reader:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def sample(self):
            return {'gpu_uuid': 'SYNTHETIC-no-GPU'}
    class Guard:
        def __init__(self, *args):
            pass
        def observe(self, *args, **kwargs):
            return None
    monkeypatch.setattr(runtime, 'NvmlResourceReader', Reader)
    monkeypatch.setattr(runtime, 'ResourceGuard', Guard)
    calls = []
    def worker(stage, spec, request, *args):
        stage.mkdir()
        calls.append((spec[0], deepcopy(request)))
        path = stage / spec[2]
        path.write_bytes(b'SYNTHETIC ONLY: ' + spec[0].encode())
        return runtime.artifact_record(path, {'synthetic': True})
    monkeypatch.setattr(runtime, 'run_worker', worker)
    kwargs = dict(output_directory=tmp_path, chain_id='fixture', noise_seed=8301, resume_existing=True,
        lease_path=tmp_path / 'gpu.lock', interrupt=lambda: None)
    return bundle, kwargs, calls, worker


def state_path(kwargs):
    return Path(kwargs['output_directory']) / 'MiniMaxH3-Prepared' / kwargs['chain_id'] / 'state.json'


def test_synthetic_serial_generation_then_decode_and_resume(setup):
    bundle, kwargs, calls, _ = setup
    movie, report = runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 2 and 'video_seed' in calls[0][1] and 'latent' in calls[1][1]
    assert report['generation_ran'] and report['decode_ran'] and report['cache_hits'] == []
    again, cached = runtime.run_prepared(bundle, **kwargs)
    assert again == movie and len(calls) == 2
    assert not cached['generation_ran'] and not cached['decode_ran']
    assert set(cached['cache_hits']) == {'generation', 'decode'}
    assert cached['human_review'] == 'pending'


@pytest.mark.parametrize('change', ['seed', 'engine', 'resume', 'bundle'])
def test_changed_settings_never_reuse_old_chain(setup, monkeypatch, change):
    bundle, kwargs, calls, _ = setup
    runtime.run_prepared(bundle, **kwargs)
    if change == 'seed':
        kwargs['noise_seed'] += 1
    elif change == 'engine':
        monkeypatch.setattr(runtime, 'engine_sources', lambda: {'synthetic_engine': 'c' * 64})
    elif change == 'resume':
        kwargs['resume_existing'] = False
    else:
        bundle['generation']['audio_seed'] = 44
    with pytest.raises(ValueError, match='new chain_id'):
        runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 2


def test_first_failure_still_binds_seed(setup, monkeypatch):
    bundle, kwargs, calls, _ = setup
    def fail(*args):
        raise RuntimeError('synthetic initial worker failure')
    monkeypatch.setattr(runtime, 'run_worker', fail)
    with pytest.raises(RuntimeError, match='initial worker'):
        runtime.run_prepared(bundle, **kwargs)
    assert json.loads(state_path(kwargs).read_text())['stages'] == {}
    kwargs['noise_seed'] += 1
    with pytest.raises(ValueError, match='new chain_id'):
        runtime.run_prepared(bundle, **kwargs)
    assert calls == []


def test_partial_generation_resumes_only_decode(setup, monkeypatch):
    bundle, kwargs, calls, real_fixture = setup
    def fail_decode(stage, spec, *args):
        if 'decode' in spec[0]:
            raise RuntimeError('synthetic decode failure')
        return real_fixture(stage, spec, *args)
    monkeypatch.setattr(runtime, 'run_worker', fail_decode)
    with pytest.raises(RuntimeError, match='decode failure'):
        runtime.run_prepared(bundle, **kwargs)
    assert set(json.loads(state_path(kwargs).read_text())['stages']) == {'generation'}
    monkeypatch.setattr(runtime, 'run_worker', real_fixture)
    _, report = runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 2 and report['cache_hits'] == ['generation']
    assert not report['generation_ran'] and report['decode_ran']


@pytest.mark.parametrize('corruption', ['unknown_stage', 'decode_only', 'wrong_schema', 'wrong_kind', 'escape', 'changed_bytes'])
def test_corrupt_state_fails_closed(setup, corruption):
    bundle, kwargs, calls, _ = setup
    movie, _ = runtime.run_prepared(bundle, **kwargs)
    path = state_path(kwargs)
    state = json.loads(path.read_text())
    if corruption == 'unknown_stage':
        state['stages']['unexpected'] = state['stages']['generation']
    elif corruption == 'decode_only':
        del state['stages']['generation']
    elif corruption == 'wrong_schema':
        state['schema'] = 2
    elif corruption == 'wrong_kind':
        state['kind'] = 'ltx_refine'
    elif corruption == 'escape':
        state['stages']['generation'] = runtime.artifact_record(bundle['assets'][0]['path'])
    else:
        movie.write_bytes(b'corrupted movie fixture')
    runtime.write_json(path, state)
    with pytest.raises(ValueError):
        runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 2


def checkpoint_bundle(bundle, kwargs):
    source = Path(kwargs['output_directory'])
    generation, decode = source / 'known-latents.fixture', source / 'known-movie.fixture'
    generation.write_bytes(b'known synthetic latent')
    decode.write_bytes(b'known synthetic movie')
    result = deepcopy(bundle)
    result['checkpoint'] = {'fingerprint': fingerprint(bundle, kwargs['noise_seed'], runtime.engine_sources()),
        'stages': {'generation': runtime.artifact_record(generation), 'decode': runtime.artifact_record(decode)}}
    return result


def test_explicit_checkpoint_is_reported_as_adoption_not_generation(setup):
    bundle, kwargs, calls, _ = setup
    bundle = checkpoint_bundle(bundle, kwargs)
    _, report = runtime.run_prepared(bundle, **kwargs)
    assert calls == [] and not report['generation_ran'] and not report['decode_ran']
    assert set(report['cache_hits']) == {'generation', 'decode'}
    state = json.loads(state_path(kwargs).read_text())
    assert state['stages']['decode']['report']['cache_origin'] == 'explicit_verified_checkpoint'


def test_checkpoint_cannot_adopt_decode_without_generation(setup):
    bundle, kwargs, calls, _ = setup
    bundle = checkpoint_bundle(bundle, kwargs)
    del bundle['checkpoint']['stages']['generation']
    with pytest.raises(ValueError, match='requires'):
        runtime.run_prepared(bundle, **kwargs)
    assert calls == []


def test_changed_checkpoint_rejected_before_copying_any_stage(setup):
    bundle, kwargs, calls, _ = setup
    bundle = checkpoint_bundle(bundle, kwargs)
    Path(bundle['checkpoint']['stages']['decode']['path']).write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='content changed'):
        runtime.run_prepared(bundle, **kwargs)
    assert not list(state_path(kwargs).parent.glob('adopted-*')) and calls == []


def test_changed_backend_during_stage_is_never_promoted(setup, monkeypatch):
    bundle, kwargs, calls, real_fixture = setup
    def mutate(*args):
        record = real_fixture(*args)
        monkeypatch.setattr(runtime, 'engine_sources', lambda: {'synthetic_engine': 'c' * 64})
        return record
    monkeypatch.setattr(runtime, 'run_worker', mutate)
    with pytest.raises(RuntimeError, match='backend changed'):
        runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 1 and json.loads(state_path(kwargs).read_text())['stages'] == {}


@pytest.mark.parametrize('chain', ['../escape', '/absolute', 'a/b', '', 'x' * 97])
def test_bad_chain_has_no_execution(setup, chain):
    bundle, kwargs, calls, _ = setup
    kwargs['chain_id'] = chain
    with pytest.raises(ValueError):
        runtime.run_prepared(bundle, **kwargs)
    assert calls == []


@pytest.fixture
def stream_setup(setup):
    bundle, kwargs, calls, worker = setup
    gen, dec = bundle['generation'], bundle['decode']
    asset = gen['text_features']
    bundle['kind'] = 'tao_stream'
    bundle['generation'] = {key: gen[key] for key in (
        'source','source_revision','base','adapter','teacher','download_receipt')}
    bundle['generation'].update(schema='t8-taomate-prepared-stream-v1', stream_requests=[
        dict(request_index=i,text_features=asset,milestones=asset,audio_seed=8301+i)
        for i in range(3)])
    bundle['decode'] = dict(schema='t8-taomate-stream-decode-v1',core=dec['core'],
        source=dec['source'],video_vae=asset,audio_vae=asset,request_count=3)
    return bundle, kwargs, calls, worker


def test_stream_public_runtime_ordered_request_binding_and_resume(stream_setup):
    bundle, kwargs, calls, _ = stream_setup
    movie, report = runtime.run_prepared(bundle, **kwargs)
    assert [name for name, _ in calls] == ['taomate_stream_worker.py','taomate_stream_decode_worker.py']
    assert [item['video_seed'] for item in calls[0][1]['stream_requests']] == [8301,8302,8303]
    assert calls[1][1]['request_count'] == 3 and report['kind'] == 'tao_stream'
    again, reused = runtime.run_prepared(bundle, **kwargs)
    assert again == movie and len(calls) == 2 and not reused['generation_ran']


@pytest.mark.parametrize('change',['video_seed','teacher_seed','prompt_asset','order'])
def test_stream_changed_request_invalidates_entire_retained_history(stream_setup, change):
    bundle, kwargs, calls, _ = stream_setup
    runtime.run_prepared(bundle, **kwargs)
    items = bundle['generation']['stream_requests']
    if change == 'video_seed':
        kwargs['noise_seed'] += 1
    elif change == 'teacher_seed':
        items[-1]['audio_seed'] += 1
    elif change == 'prompt_asset':
        # Another valid identity, not an unknown file bypassing the manifest.
        items[-1]['text_features'] = bundle['assets'][1]['path']
    else:
        items.reverse()
    with pytest.raises(ValueError):
        runtime.run_prepared(bundle, **kwargs)
    assert len(calls) == 2


def test_stream_decode_failure_resumes_without_repeating_sampling(stream_setup, monkeypatch):
    bundle, kwargs, calls, real_fixture = stream_setup
    def fail_decode(stage, spec, *args):
        if spec[0] == 'taomate_stream_decode_worker.py':
            raise RuntimeError('synthetic stream decode failure')
        return real_fixture(stage, spec, *args)
    monkeypatch.setattr(runtime,'run_worker',fail_decode)
    with pytest.raises(RuntimeError,match='stream decode failure'):
        runtime.run_prepared(bundle, **kwargs)
    assert set(json.loads(state_path(kwargs).read_text())['stages']) == {'generation'}
    monkeypatch.setattr(runtime,'run_worker',real_fixture)
    _, report = runtime.run_prepared(bundle, **kwargs)
    assert not report['generation_ran'] and report['decode_ran'] and len(calls) == 2
