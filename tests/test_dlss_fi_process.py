import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import psutil
import pytest

from tools.dlss_fi_process import IsolatedTaskError, WindowsJob, run_isolated

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object route')
TASK = Path(__file__).parent/'fixtures/dlss_fi_isolation_task.py'


def assert_gone(receipt):
    assert receipt['active_after_cleanup'] == 0
    pids = [receipt['pid'], *map(int, re.findall(r'CHILD=(\d+)', receipt['stdout_tail']))]
    # Job accounting reaches zero before Windows removes every exited process
    # object from enumeration. Observe bounded eventual disappearance; never kill
    # by the recorded PID (it can already have been recycled).
    deadline = time.monotonic()+3
    while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.02)
    assert not any(psutil.pid_exists(pid) for pid in pids)


def test_normal_task_is_owned_before_execution():
    receipt = run_isolated(TASK, ['complete'], timeout=10)
    assert receipt['status'] == 'complete' and receipt['job_assigned_before_task']
    assert 'OWNED_TASK_COMPLETE' in receipt['stdout_tail']
    assert_gone(receipt)


@pytest.mark.parametrize('mode', ['hang', 'tree_hang'])
def test_codec_hang_and_grandchild_have_hard_deadline(mode):
    with pytest.raises(IsolatedTaskError) as error:
        run_isolated(TASK, [mode], timeout=1)
    assert error.value.receipt['status'] == 'timeout'
    assert error.value.receipt['wall_seconds'] < 8
    if mode == 'tree_hang':
        assert 'CHILD=' in error.value.receipt['stdout_tail']
    assert_gone(error.value.receipt)


def test_cancel_before_launch_does_not_create_process():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(IsolatedTaskError) as error:
        run_isolated(TASK, ['hang'], cancel=cancel)
    assert error.value.receipt['status'] == 'cancelled_before_start'
    assert error.value.receipt['pid'] is None


def test_active_cancel_cleans_whole_tree():
    cancel = threading.Event()
    timer = threading.Timer(1, cancel.set)
    timer.start()
    try:
        with pytest.raises(IsolatedTaskError) as error:
            run_isolated(TASK, ['tree_hang'], timeout=10, cancel=cancel)
    finally:
        timer.cancel()
    assert error.value.receipt['status'] == 'cancelled'
    assert_gone(error.value.receipt)


@pytest.mark.parametrize('mode,status', [('orphan', 'child_left_descendants'), ('error', 'child_failed')])
def test_failed_or_orphaned_task_cannot_pass(mode, status):
    with pytest.raises(IsolatedTaskError) as error:
        run_isolated(TASK, [mode], timeout=10)
    assert error.value.receipt['status'] == status
    assert_gone(error.value.receipt)


def test_stdout_and_stderr_flood_are_bounded_and_drained():
    receipt = run_isolated(TASK, ['flood'], timeout=10)
    assert len(receipt['stdout_tail']) <= 131072 and len(receipt['stderr_tail']) <= 131072
    assert_gone(receipt)


def test_closing_job_handle_kills_its_gated_child_without_explicit_terminate():
    job = WindowsJob()
    process = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import sys;sys.stdin.buffer.read();sys.exit(37)'],
        stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        job.assign(process)
        assert job.active() == 1
        job.close()
        # Windows kill-on-close can use exit code zero; EOF would exit 37.
        assert process.wait(timeout=5) != 37
    finally:
        job.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()


@pytest.mark.parametrize('timeout', [0, -1, True, float('nan'), float('inf'), 3601])
def test_bad_deadline_fails_before_start(timeout):
    with pytest.raises(ValueError):
        run_isolated(TASK, ['complete'], timeout=timeout)
