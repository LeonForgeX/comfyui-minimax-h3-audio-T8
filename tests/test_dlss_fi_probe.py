import json
import os
import subprocess
import sys

import pytest

from tools.probe_dlss_fi_runtime import collect, parse_probe
from tools.run_progressive_pilot import OwnedServer


def response(**kw):
    return json.dumps({"available": True, "multi_frame_count_max": 1,
        "runtime_version": "fixture", "worker_version": "fixture", "detail": "synthetic", **kw}).encode()


def owner(tmp_path, script):
    obj = OwnedServer(tmp_path, 0, True)
    obj.process = subprocess.Popen([sys.executable, '-u', '-c', script], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return obj


def test_report_is_not_actual_device_or_generation_qualification():
    report = parse_probe(b'diagnostic line\n'+response())
    assert report['supports_native_2x_by_report'] is True
    assert not report['actual_device_qualified'] and not report['generation_qualified'] and not report['quality_qualified']
    assert not parse_probe(response(available=False))['supports_native_2x_by_report']


@pytest.mark.parametrize('data', [b'', b'x'*65537, b'[]', b'{}', b'\xff', response(available='true'),
    response(multi_frame_count_max=True), response(multi_frame_count_max=999), response(runtime_version=None)],
    ids=['empty','oversize','array','missing','encoding','bool-string','bool-count','large-count','missing-version'])
def test_malformed_probe_does_not_qualify(data):
    with pytest.raises((ValueError, UnicodeError)):
        parse_probe(data)


def test_real_cpu_process_collects_stdout_and_bounded_stderr_and_cleans(tmp_path):
    obj = owner(tmp_path, "import os;os.write(2,b'x'*200000);print('fixture')")
    result = collect(obj, lambda: None, timeout=3)
    assert result['stdout'].strip() == b'fixture' and len(result['stderr_tail']) <= 65536
    assert obj.process.poll() == 0 and obj.probe_output['stdout'].strip() == 'fixture'


@pytest.mark.parametrize('script,exception', [
    ("import time;time.sleep(60)", TimeoutError),
    ("import os;os.write(1,b'x'*200000)", ValueError),
    ("import sys;print('reason',file=sys.stderr);raise SystemExit(3)", RuntimeError)])
def test_timeout_flood_and_failure_stop_owned_process_and_keep_diagnostic(tmp_path, script, exception):
    obj = owner(tmp_path, script)
    with pytest.raises(exception):
        collect(obj, lambda: None, timeout=.8)
    assert obj.process.poll() is not None
    assert hasattr(obj, 'probe_output')


def test_resource_guard_failure_cleans_probe(tmp_path):
    obj = owner(tmp_path, "import time;time.sleep(60)")
    def fail():
        raise RuntimeError('resource guard')
    with pytest.raises(RuntimeError, match='resource guard'):
        collect(obj, fail, timeout=3)
    assert obj.process.poll() is not None
