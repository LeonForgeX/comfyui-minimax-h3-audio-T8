"""CPU-only candidate regression with durable collection and per-phase receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def source_snapshot(project):
    paths = set(project.glob('*.py'))
    paths.update(project / name for name in ('pyproject.toml', 'meta.json', 'features.json', '.comfyignore'))
    for directory in ('h3_t8', 'tools', 'tests', 'web', 'examples'):
        paths.update(path for path in (project / directory).rglob('*')
                     if path.is_file() and path.suffix.lower() in {'.py', '.js', '.json', '.html'})
    return {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select', nargs='*', default=['tests'])
    parser.add_argument('--deselect', nargs='*', default=[],
                        help='Explicit integration cases outside this candidate; retained in the receipt')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(project / 'artifacts'):
        raise ValueError('New task-owned artifacts directory required')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      PYTHONPATH=str(args.core.resolve()))
    sys.path[:0] = [str(project), str(args.core.resolve())]
    sys.argv = ['candidate-cpu-regression', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import torch
    import pytest
    torch.set_num_threads(2)
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError('Regression must not have CUDA available or initialized')
    output.mkdir(parents=True)
    frozen = source_snapshot(project)
    (output / 'sources-before.json').write_text(json.dumps(frozen, indent=2), encoding='utf8')
    started = time.monotonic()
    counts = {}

    class Durable:
        def pytest_deselected(self, items):
            (output / 'deselected.json').write_text(json.dumps([item.nodeid for item in items],
                ensure_ascii=False, indent=2), encoding='utf8')

        def pytest_collection_finish(self, session):
            (output / 'collection.json').write_text(json.dumps([item.nodeid for item in session.items],
                ensure_ascii=False, indent=2), encoding='utf8')

        def pytest_runtest_logreport(self, report):
            row = {'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome,
                   'duration': report.duration}
            if report.failed or report.skipped:
                row['detail'] = str(report.longrepr)
            with (output / 'phases.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            key = report.when + ':' + report.outcome
            counts[key] = counts.get(key, 0) + 1

    code = pytest.main([*args.select, *['--deselect=' + item for item in args.deselect],
                        '-q', '--junitxml=' + str(output / 'results.xml')], plugins=[Durable()])
    after = source_snapshot(project)
    changed = sorted(key for key in frozen.keys() | after.keys() if frozen.get(key) != after.get(key))
    receipt = {'pytest_exit_code': int(code), 'phase_counts': counts,
               'cuda_initialized': torch.cuda.is_initialized(), 'elapsed_seconds': time.monotonic() - started,
               'sources_checked': len(frozen), 'changed_sources': changed,
               'scope': args.select, 'deselected': args.deselect,
               'status': 'pass' if code == 0 else 'failed'}
    if receipt['cuda_initialized']:
        receipt['status'] = 'failed_cuda_initialized'
        code = 1
    if changed:
        receipt['status'] = 'failed_source_changed'
        code = 1
    (output / 'terminal.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
