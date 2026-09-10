from fractions import Fraction
from pathlib import Path
import struct
import sys
import threading
import time

import pytest

from tools.dlss_fi_transport import BinarySession, frame_packet, runtime_identity

FIXTURE = Path(__file__).parent / "fixtures/dlss_fi_fake_worker.py"


def session(mode="fragment", **kw):
    return BinarySession([sys.executable, "-u", str(FIXTURE), mode], cwd=FIXTURE.parent,
        width=kw.pop("width", 2), height=kw.pop("height", 2), frame_count=3, timeout=kw.pop("timeout", 2), **kw)


def test_fragmented_handshake_and_reset_payload_are_fully_consumed():
    worker = session()
    with worker:
        assert worker.frame(bytes(16), bytes(16), Fraction(3, 2), reset=True)["rgba"] is None
        response = worker.frame(bytes(16), bytes(16), Fraction(37, 24))
        assert response["rgba"] == bytes([2])*16
        assert response["usable_generated_frames"] == 1
        assert worker.frame(bytes(16), bytes(16), Fraction(38, 24), reset=True)["rgba"] is None
        with pytest.raises(RuntimeError, match="beyond"):
            worker.frame(bytes(16), bytes(16), Fraction(39, 24))
    assert worker.process.poll() is not None
    assert not worker.stop_receipt["unfinished_threads"]
    first = worker.stop_receipt
    worker.close()
    assert worker.stop_receipt is first
    assert worker.stop_receipt["runtime_qualified"] is False


@pytest.mark.parametrize("mode", ["disabled", "huge_count", "truncated", "no_frame"])
def test_corrupt_or_empty_response_fails_closed_and_cleans_owned_process(mode):
    worker = session(mode)
    with worker:
        if mode == "no_frame":
            worker.frame(bytes(16), bytes(16), Fraction(0), reset=True)
        with pytest.raises((ValueError, RuntimeError)):
            worker.frame(bytes(16), bytes(16), Fraction(worker.next_index, 24), reset=worker.next_index == 0)
        assert worker.closed and worker.failed and worker.process.poll() is not None
        with pytest.raises(RuntimeError):
            worker.frame(bytes(16), bytes(16), 0, reset=True)


def test_invalid_setup_is_not_accepted():
    with pytest.raises(ValueError, match="protocol"):
        session("bad_setup")


@pytest.mark.parametrize("mode", ["hang_setup", "hang_frame", "hang_input"])
def test_pipe_timeout_is_bounded_including_blocked_writer(mode):
    start = time.perf_counter()
    if mode == "hang_setup":
        with pytest.raises(TimeoutError):
            session(mode, timeout=.8)
    else:
        worker = session(mode, timeout=.8, width=512, height=256)
        with worker, pytest.raises(TimeoutError):
            worker.frame(bytes(512*256*4), bytes(512*256*4), 0, reset=True)
        assert worker.process.poll() is not None and not worker.stop_receipt["unfinished_threads"]
    assert time.perf_counter()-start < 8


def test_cancel_during_blocked_read_kills_only_owned_worker():
    cancel = threading.Event()
    worker = session("hang_frame", cancel=cancel)
    timer = threading.Timer(.1, cancel.set)
    timer.start()
    try:
        with worker, pytest.raises(RuntimeError, match="cancelled"):
            worker.frame(bytes(16), bytes(16), 0, reset=True)
    finally:
        timer.cancel()
    assert worker.process.poll() is not None


def test_stderr_without_newlines_is_bounded_and_does_not_deadlock():
    with session("stderr_flood") as worker:
        worker.frame(bytes(16), bytes(16), 0, reset=True)
        assert sum(map(len, worker.logs)) <= 65536


def test_payload_validation_happens_before_any_write_and_preserves_index():
    with session() as worker:
        for color, flow, stamp, reset in ((b"x", bytes(16), 0, True),
                (bytes(16), b"x", 0, True), (bytes(16), bytes(16), 0.0, True),
                (bytes(16), bytes(16), 0, False)):
            with pytest.raises(ValueError):
                worker.frame(color, flow, stamp, reset=reset)
            assert worker.next_index == 0
        worker.frame(bytes(16), bytes(16), 1, reset=True)
        with pytest.raises(ValueError, match="increase"):
            worker.frame(bytes(16), bytes(16), 1)
        assert worker.next_index == 1


@pytest.mark.parametrize("stamp", [1.5, 2**63, Fraction(1, 2**63)])
def test_wire_timestamp_rejects_lossy_or_overflow(stamp):
    with pytest.raises(ValueError):
        frame_packet(0, stamp, True)


def test_wire_fraction_preserves_negative_origin():
    assert struct.unpack("<4I2q", frame_packet(3, Fraction(-1, 24), True)) == (0x31464746, 3, 1, 0, -1, 24)


def test_wrong_runtime_hash_does_not_execute(tmp_path):
    (tmp_path / "dlssg-worker.exe").write_bytes(b"MZnot_a_worker")
    with pytest.raises(ValueError, match="identity"):
        runtime_identity(tmp_path)


def test_registration_failure_stops_worker_before_handshake():
    processes = []
    def fail(process):
        processes.append(process)
        raise ValueError('registration rejected')
    with pytest.raises(ValueError, match='registration rejected'):
        session(on_start=fail)
    assert len(processes) == 1 and processes[0].poll() is not None
