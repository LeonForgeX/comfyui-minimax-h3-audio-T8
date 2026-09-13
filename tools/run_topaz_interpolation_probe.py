"""Explicit official Topaz frame-interpolation probe; never downloads model weights."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('install', 'definitions', 'data', 'output', 'source', 'lease'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', default='apo-8')
    parser.add_argument('--multiplier', type=int, choices=(2, 4), default=2)
    parser.add_argument('--duplicate-threshold', type=float, default=.01)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    package = importlib.util.module_from_spec(
        importlib.util.spec_from_loader('topaz_fi_probe_pkg', loader=None, is_package=True))
    package.__path__ = [str(project / 'h3_t8')]
    sys.modules['topaz_fi_probe_pkg'] = package

    from topaz_fi_probe_pkg.topaz_contract import OfficialTopaz
    from topaz_fi_probe_pkg.topaz_runtime import run_interpolation

    if args.output.exists():
        raise ValueError('Use a new output directory')
    runtime = OfficialTopaz(args.install, args.definitions, args.data)
    path, report = run_interpolation(
        runtime,
        args.source,
        args.output,
        model_id=args.model,
        multiplier=args.multiplier,
        lease_path=args.lease,
        duplicate_threshold=args.duplicate_threshold,
    )
    print(json.dumps({'status': report['status'], 'output': str(path)}))


if __name__ == '__main__':
    main()
