"""Parallel CPU scanning preserves the existing exact byte contract."""
from collections import OrderedDict
import threading

import pytest
import torch

from h3_audio_t8_pkg.progressive_weight_digest import weight_state_identity
from h3_audio_t8_pkg import long_video_dual_identity as identity


def state():
    return OrderedDict((f'weight_{n}', torch.full((6 * 1024 * 1024,), .5 + n, dtype=torch.float16))
                       for n in range(4))


def test_original_sha_schema_and_fresh_mutations_are_preserved(monkeypatch):
    values = state()
    expected = identity.content_identity(values)
    threads = set()
    original = identity.content_identity
    def observe(value):
        threads.add(threading.current_thread().name)
        return original(value)
    monkeypatch.setattr(identity, 'content_identity', observe)
    assert weight_state_identity(values) == expected
    assert any(name.startswith('h3-weight-digest') for name in threads)
    values['weight_0'][0] = .25
    assert weight_state_identity(values) != expected
    values['weight_1'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        weight_state_identity(values)


def test_effectful_tensor_reader_is_kept_on_the_caller_thread(monkeypatch):
    values = state()
    original = values['weight_0'].detach
    calls = []
    def detach():
        calls.append(threading.current_thread().name)
        return original()
    values['weight_0'].detach = detach
    native = identity.content_identity
    def record(value):
        calls.append(threading.current_thread().name)
        return native(value)
    monkeypatch.setattr(identity, 'content_identity', record)
    weight_state_identity(values)
    assert calls and set(calls) == {threading.current_thread().name}


def test_small_state_does_not_create_worker_threads(monkeypatch):
    original = identity.content_identity
    threads = []
    def observe(value):
        threads.append(threading.current_thread().name)
        return original(value)
    monkeypatch.setattr(identity, 'content_identity', observe)
    values = {'weight': torch.ones(12)}
    assert weight_state_identity(values) == original(values)
    assert set(threads) == {threading.current_thread().name}
