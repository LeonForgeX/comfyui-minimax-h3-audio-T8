"""CPU-only process ownership fixture, never a model/quality test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--request', type=Path, required=True)
request = parser.parse_args().request
mode = json.loads(request.read_text())['mode']
if mode in ('tree', 'orphan'):
    child = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(30)'],
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    request.with_name('child.json').write_text(json.dumps({'pid': child.pid}))
request.with_name('ready').write_text('CPU fixture, not generation')
if mode == 'error':
    raise RuntimeError('fixture intentional error')
if mode in ('tree', 'hang'):
    time.sleep(30)
if mode in ('artifact', 'bad_hash', 'bad_status'):
    output = request.with_name('fixture-output.safetensors')
    output.write_bytes(b'SYNTHETIC artifact; not a safetensors model')
    request.with_name('report.json').write_text(json.dumps({
        'status': 'fixture_pass' if mode != 'bad_status' else 'failed',
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest() if mode != 'bad_hash' else '0' * 64}))
print('FIXTURE_COMPLETE', flush=True)
