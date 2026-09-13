"""Actual Windows child-tree tests with synthetic CPU workers, no model reruns."""
import json
import os
from pathlib import Path
import time

import psutil
import pytest

from h3_audio_t8_pkg import prepared_process as process
from h3_audio_t8_pkg.prepared_generation_runtime import write_json
from h3_audio_t8_pkg import prepared_generation_runtime as runtime

TASK = Path(__file__).parent / 'fixtures/prepared_process_task.py'
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object contract')


def test_missing_python311_add_note_preserves_original_cancel(tmp_path):
    class LegacyCancel(InterruptedError):
        add_note = None  # Simulate the missing 3.11 API, not a full 3.10 test.
    original = LegacyCancel('legacy cancel')
    request = tmp_path / 'request.json'
    write_json(request, {'mode': 'complete'})
    def cancel():
        raise original
    def broken_write(*args):
        raise OSError('legacy receipt failure')
    with pytest.raises(LegacyCancel) as caught:
        process.execute_owned(TASK, request, timeout=5, observe=lambda: None,
            interrupt=cancel, write_json=broken_write)
    assert caught.value is original
    assert 'legacy receipt failure' in str(original.__notes__)


def test_broken_custom_note_method_cannot_replace_original():
    class CustomCancel(InterruptedError):
        def add_note(self, message):
            raise OSError('broken custom diagnostics')
    original = CustomCancel('keep me')
    process.note_original(original, 'cleanup detail')
    assert str(original) == 'keep me'


def call(tmp_path, mode, *, timeout=5, interrupt=lambda: None, observe=lambda: None):
    request = tmp_path / 'request.json'
    write_json(request, {'mode': mode})
    return process.execute_owned(TASK, request, timeout=timeout, interrupt=interrupt,
        observe=observe, write_json=write_json)


def gone(tmp_path):
    receipt = json.loads((tmp_path / 'process.json').read_text())
    assert receipt['cleanup']['active_after_cleanup'] == 0
    assert receipt['cleanup']['errors'] == []
    pids = [receipt['pid']]
    child = tmp_path / 'child.json'
    if child.exists():
        pids.append(json.loads(child.read_text())['pid'])
    deadline = time.monotonic() + 3
    while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not any(psutil.pid_exists(pid) for pid in pids)
    return receipt


def test_real_owned_success(tmp_path):
    receipt = call(tmp_path, 'complete')
    assert receipt['status'] == 'complete' and receipt['job_assigned_before_task']
    assert 'FIXTURE_COMPLETE' in (tmp_path / 'stdout.log').read_text()
    gone(tmp_path)


@pytest.mark.parametrize('mode', ['hang', 'tree'])
def test_real_timeout_stops_tree(tmp_path, mode):
    with pytest.raises(TimeoutError):
        call(tmp_path, mode, timeout=1)
    if mode == 'tree':
        assert (tmp_path / 'child.json').exists()
    gone(tmp_path)


@pytest.mark.parametrize('mode,match', [('error', 'failed'), ('orphan', 'descendants')])
def test_real_failure_or_orphan_never_passes(tmp_path, mode, match):
    with pytest.raises(RuntimeError, match=match):
        call(tmp_path, mode)
    gone(tmp_path)


def test_real_cancel_keeps_original_exception_and_stops_tree(tmp_path):
    original = InterruptedError('Comfy cancel fixture')
    def interrupt():
        if (tmp_path / 'child.json').exists():
            raise original
    with pytest.raises(InterruptedError) as caught:
        call(tmp_path, 'tree', interrupt=interrupt)
    assert caught.value is original
    gone(tmp_path)


def test_cancel_before_start_never_creates_worker(tmp_path):
    def cancel():
        raise InterruptedError('before start')
    with pytest.raises(InterruptedError):
        call(tmp_path, 'complete', interrupt=cancel)
    receipt = json.loads((tmp_path / 'process.json').read_text())
    assert 'pid' not in receipt
    assert receipt['cleanup']['active_after_cleanup'] == 0
    assert not (tmp_path / 'ready').exists()


def test_failed_receipt_write_cannot_mask_original(tmp_path):
    request = tmp_path / 'request.json'
    write_json(request, {'mode': 'complete'})
    original = InterruptedError('original cancel')
    def cancel():
        raise original
    def broken_write(*args):
        raise OSError('receipt storage failure fixture')
    with pytest.raises(InterruptedError) as caught:
        process.execute_owned(TASK, request, timeout=5, observe=lambda: None,
            interrupt=cancel, write_json=broken_write)
    assert caught.value is original
    assert 'receipt storage failure' in str(original.__notes__)


def test_cleanup_error_does_not_mask_worker_failure(tmp_path, monkeypatch):
    real_stop = process.stop_owned
    def injected(*args, **kwargs):
        result = real_stop(*args, **kwargs)
        result['errors'].append('synthetic cleanup diagnostic')
        return result
    monkeypatch.setattr(process, 'stop_owned', injected)
    with pytest.raises(RuntimeError, match='worker failed') as caught:
        call(tmp_path, 'error')
    assert 'synthetic cleanup diagnostic' in str(caught.value.__notes__)


def test_cleanup_uncertainty_prevents_success(tmp_path, monkeypatch):
    real_stop = process.stop_owned
    def injected(*args, **kwargs):
        result = real_stop(*args, **kwargs)
        result['active_after_cleanup'] = None
        return result
    monkeypatch.setattr(process, 'stop_owned', injected)
    with pytest.raises(RuntimeError, match='cleanup incomplete'):
        call(tmp_path, 'complete')


def test_cleanup_continues_after_query_and_termination_errors():
    calls = []
    class BrokenJob:
        def active(self):
            calls.append('query')
            raise OSError('synthetic query failure')
        def terminate(self):
            calls.append('terminate')
            raise OSError('synthetic termination failure')
        def close(self):
            calls.append('close')
    result = process.stop_owned(BrokenJob(), None, assigned=True)
    assert calls == ['query', 'terminate', 'query', 'close']
    assert result['active_after_cleanup'] is None and len(result['errors']) == 3


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf'), 7201])
def test_invalid_timeout_has_no_side_effect(tmp_path, timeout):
    with pytest.raises(ValueError):
        call(tmp_path, 'complete', timeout=timeout)
    assert not (tmp_path / 'process.json').exists()


@pytest.mark.parametrize('mode', ['artifact', 'bad_hash', 'bad_status'])
def test_actual_worker_controller_checks_report_and_digest(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(runtime, 'BACKEND', TASK.parent)
    class Reader:
        def sample(self):
            return {'synthetic': True}
    class Guard:
        def observe(self, row):
            assert row == {'synthetic': True}
            return None
    stage = tmp_path / 'stage'
    spec = (TASK.name, 'fixture_pass', 'fixture-output.safetensors', 5)
    if mode == 'artifact':
        record = runtime.run_worker(stage, spec, {'mode': mode}, Reader(), Guard(), lambda: None)
        assert Path(record['path']).read_bytes().startswith(b'SYNTHETIC')
    else:
        with pytest.raises(RuntimeError, match='disagree|required completion'):
            runtime.run_worker(stage, spec, {'mode': mode}, Reader(), Guard(), lambda: None)
    gone(stage)
    assert json.loads((stage / 'terminal.json').read_text())['status'] == ('complete' if mode == 'artifact' else 'failed')


def test_real_resource_guard_exception_stops_owned_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'BACKEND', TASK.parent)
    stage = tmp_path / 'stage'
    class Reader:
        def sample(self):
            return {'synthetic': True}
    class Guard:
        def observe(self, row):
            return 'synthetic_low_memory' if (stage / 'child.json').exists() else None
    spec = (TASK.name, 'fixture_pass', 'fixture-output.safetensors', 5)
    with pytest.raises(RuntimeError, match='resource guard: synthetic_low_memory'):
        runtime.run_worker(stage, spec, {'mode': 'tree'}, Reader(), Guard(), lambda: None)
    gone(stage)
