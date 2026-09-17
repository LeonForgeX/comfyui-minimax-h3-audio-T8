"""Synthetic lifecycle negatives; not a GPU streaming/quality qualification."""
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.prepared_backend.taomate_request_state import OwnedRequestState


def runtime():
    return SimpleNamespace(_request_index=0, _native_frame_offset=0,
        _video_latent_offset=0, _audio_latent_offset=0, _running=False,
        executions=[], cache=[])


def commit(value):
    entry = SimpleNamespace(native_frame_offset=value._native_frame_offset,
        video_latent_offset=value._video_latent_offset, audio_latent_offset=value._audio_latent_offset,
        published_native_frames=120, published_video_latents=35, published_audio_latents_per_channel=200)
    value.executions.append(entry)
    value._request_index += 1
    value._native_frame_offset += 120
    value._video_latent_offset += 35
    value._audio_latent_offset += 200
    value.cache.append('committed-fixture')
    return 'output'


def test_two_requests_preserve_committed_state_until_close():
    owner, value, model = OwnedRequestState(), runtime(), object()
    for i in range(2):
        assert owner.execute(value, model, lambda: commit(value), value.cache.clear) == 'output'
        assert len(value.cache) == i + 1 and len(owner.records) == i + 1
    assert owner.records[1]['before'] == owner.records[0]['after']
    owner.close(value.cache.clear)
    owner.close(lambda: pytest.fail('closed owner must be idempotent'))
    assert owner.state == 'closed' and not value.cache and owner._model is None
    with pytest.raises(RuntimeError, match='new runtime'):
        owner.execute(value, model, lambda: commit(value), value.cache.clear)


def test_different_model_rejected_without_erasing_valid_stream():
    owner, value, model = OwnedRequestState(), runtime(), object()
    owner.execute(value, model, lambda: commit(value), value.cache.clear)
    with pytest.raises(ValueError, match='model owner'):
        owner.execute(value, object(), lambda: pytest.fail('not called'), value.cache.clear)
    assert owner.state == 'ready' and value.cache
    owner.execute(value, model, lambda: commit(value), value.cache.clear)


@pytest.mark.parametrize('phase', ['before', 'partial', 'after'])
def test_cancel_or_failure_poison_stream_and_next_task_is_fresh(phase):
    owner, value, model = OwnedRequestState(), runtime(), object()
    checks = []
    def interrupt():
        checks.append(1)
        if phase == 'before' or phase == 'after' and len(checks) == 2:
            raise InterruptedError('test cancellation')
    def call():
        value.cache.append('partial-fixture')
        if phase == 'partial':
            raise RuntimeError('test phase failure')
        return commit(value)
    with pytest.raises((InterruptedError, RuntimeError)):
        owner.execute(value, model, call, value.cache.clear, interrupt)
    assert owner.state == 'failed' and not value.cache and owner._model is None
    with pytest.raises(RuntimeError, match='failed'):
        owner.execute(value, model, call, value.cache.clear)
    fresh, other = OwnedRequestState(), runtime()
    fresh.execute(other, model, lambda: commit(other), other.cache.clear)
    assert fresh.records[0]['before']['_request_index'] == 0


def test_cleanup_error_does_not_mask_original_failure():
    owner, value = OwnedRequestState(), runtime()
    def fail():
        raise ValueError('original')
    def cleanup():
        raise RuntimeError('cleanup')
    with pytest.raises(ValueError, match='original') as caught:
        owner.execute(value, object(), fail, cleanup)
    assert 'cleanup' in caught.value.__notes__[0]
    assert owner.state == 'failed' and owner._model is None


@pytest.mark.parametrize('field', ['_request_index', '_native_frame_offset', '_video_latent_offset', '_audio_latent_offset'])
def test_invalid_commit_is_not_reusable(field):
    owner, value = OwnedRequestState(), runtime()
    def invalid():
        commit(value)
        setattr(value, field, getattr(value, field) + 1)
    with pytest.raises(RuntimeError):
        owner.execute(value, object(), invalid, value.cache.clear)
    assert owner.state == 'failed' and not value.cache


def test_concurrent_run_and_close_rejected_without_mutating_owner():
    owner, value, model = OwnedRequestState(), runtime(), object()
    def call():
        with pytest.raises(RuntimeError, match='active request'):
            owner.execute(value, model, lambda: None, value.cache.clear)
        with pytest.raises(RuntimeError, match='active streaming'):
            owner.close(value.cache.clear)
        assert owner.state == 'running'
        return commit(value)
    owner.execute(value, model, call, value.cache.clear)
    assert owner.state == 'ready'


def test_failed_close_stays_closed():
    owner = OwnedRequestState()
    def fail():
        raise OSError('fixture cleanup error')
    with pytest.raises(OSError):
        owner.close(fail)
    assert owner.state == 'closed'


def test_cleanup_preserves_exception_without_python311_add_note():
    class OlderException(Exception):
        add_note = None
    owner, value = OwnedRequestState(), runtime()
    def fail():
        raise OlderException('original')
    def cleanup():
        raise RuntimeError('cleanup')
    with pytest.raises(OlderException, match='original'):
        owner.execute(value, object(), fail, cleanup)
    assert len(owner.cleanup_errors) == 1 and 'cleanup' in owner.cleanup_errors[0]
