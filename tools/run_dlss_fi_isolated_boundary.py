"""Qualification of one real FI file inside the new outer Windows Job.

The complete fixed-case preflight, codec work and DLSSG grandchild live inside
the job. Default is a read-only CPU plan. Still not an arbitrary-file node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.dlss_fi_process import run_isolated, IsolatedTaskError  # noqa: E402
from tools.run_dlss_fi_boundaries import CASES, RESEARCH  # noqa: E402
from tools.progressive_probe_control import file_identity  # noqa: E402


def command(args, child_root):
    return ['--mode', 'gpu', '--case', args.case, '--manifest', str(args.manifest.resolve(strict=True)),
            '--runtime', str(args.runtime.resolve(strict=True)), '--run-root', str(child_root)]


def run(args):
    script = Path(__file__).with_name('run_dlss_fi_boundaries.py')
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New exclusive research root required')
    if args.mode == 'plan':
        return {'status': 'CPU_plan_no_process', 'case': args.case, 'root': str(root),
                'child_script': file_identity(script), 'deadline_seconds': 240}
    # Creating only a receipt folder, never replacing a source/output.
    root.mkdir(parents=True)
    identity = {str(p): file_identity(p) for p in (Path(__file__), script, Path(__file__).with_name('dlss_fi_process.py'))}
    receipt = {'status': 'incomplete', 'quality_qualified': False, 'identity': identity}
    try:
        owned = run_isolated(script, command(args, root/'child'), timeout=240,
                             check=lambda: (_ for _ in ()).throw(RuntimeError('STOP requested')) if (root/'STOP').exists() else None)
        child = json.loads((root/'child/terminal.json').read_text(encoding='utf8'))
        if child['status'] != 'actual_boundary_complete' or child['server_stop']['owned_children_remaining']:
            raise ValueError('Isolated child did not complete its actual FI contract')
        if any(file_identity(p) != value for p, value in identity.items()):
            raise ValueError('Isolation code changed during execution')
        receipt.update(status='actual_boundary_in_kill_on_close_job_complete', owner=owned,
                       candidate=child['candidate'], counts=child['counts'])
    except IsolatedTaskError as error:
        receipt.update(status='failed', owner=error.receipt)
        raise
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        with (root/'terminal.json').open('x', encoding='utf8') as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'gpu'), default='plan')
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
