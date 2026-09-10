from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.dlss_fi_backend import process


def clock(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(process.time,'monotonic',lambda:ticks[0])
    monkeypatch.setattr(process.time,'sleep',lambda seconds:ticks.__setitem__(0,ticks[0]+seconds))
    return ticks


def test_lagging_job_accounting_does_not_report_left_descendant(monkeypatch):
    ticks = clock(monkeypatch)
    counts = iter((1,1,0))
    assert process._settled_active_processes(SimpleNamespace(active=lambda:next(counts))) == 0
    assert ticks[0] == pytest.approx(.02)


def test_live_descendant_still_returned_after_bounded_grace(monkeypatch):
    ticks = clock(monkeypatch)
    assert process._settled_active_processes(SimpleNamespace(active=lambda:1)) == 1
    assert ticks[0] == pytest.approx(.25)


def test_already_empty_job_does_not_wait(monkeypatch):
    ticks = clock(monkeypatch)
    assert process._settled_active_processes(SimpleNamespace(active=lambda:0)) == 0
    assert ticks[0] == 0


def test_exit_confirmation_respects_resource_guard_and_cancel(monkeypatch):
    clock(monkeypatch)
    def fail_guard():
        raise RuntimeError('resource guard')
    with pytest.raises(RuntimeError,match='resource guard'):
        process._settled_active_processes(SimpleNamespace(active=lambda:1),check=fail_guard)
    with pytest.raises(InterruptedError):
        process._settled_active_processes(SimpleNamespace(active=lambda:1),cancel=SimpleNamespace(is_set=lambda:True))
