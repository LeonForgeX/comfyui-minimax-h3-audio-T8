"""Bounded Job-owned development worker; never starts or controls a UI service."""
import argparse
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.receipt.exists() or not args.receipt.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new owned receipt within project artifacts')
    package = types.ModuleType('_owned_development')
    package.__path__ = [str(project / 'h3_t8')]
    sys.modules[package.__name__] = package
    from _owned_development.dlss_fi_backend.process import run_isolated, IsolatedTaskError
    result = dict(status='incomplete', published=False, user_frontend_changed=False)
    try:
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        result['process'] = run_isolated(args.worker, arguments, timeout=args.timeout)
        result['status'] = 'owned_worker_exit0_cleanup0'
    except IsolatedTaskError as error:
        result.update(status='failed', process=error.receipt)
        raise
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2), encoding='utf8')
        brief = {k: v for k, v in result.items() if k != 'process'}
        brief['process'] = {k: v for k, v in result.get('process', {}).items() if k not in {'stdout_tail', 'stderr_tail'}}
        print(json.dumps(brief), flush=True)


if __name__ == '__main__':
    main()
