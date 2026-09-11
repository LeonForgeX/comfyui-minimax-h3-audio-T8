"""Explicit official Topaz audit or serial enhancement; no automatic model downloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('install', 'definitions', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--lease', type=Path)
    p.add_argument('--model', default='iris-3')
    p.add_argument('--scale', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    package = importlib.util.module_from_spec(importlib.util.spec_from_loader('topaz_probe_pkg', loader=None, is_package=True))
    package.__path__ = [str(project / 'h3_t8')]
    sys.modules['topaz_probe_pkg'] = package
    from topaz_probe_pkg.topaz_contract import OfficialTopaz
    from topaz_probe_pkg.topaz_runtime import audit_installation, model_evidence, run_regular
    runtime = OfficialTopaz(args.install, args.definitions, args.data)
    if args.output.exists():
        raise ValueError('Use a new output directory')
    if args.execute:
        if args.source is None or args.lease is None:
            raise ValueError('Explicit source and shared serial GPU lease required')
        path, report = run_regular(runtime, args.source, args.output, model_id=args.model,
            width=args.width, height=args.height, scale=args.scale, lease_path=args.lease)
        print(json.dumps({'status': report['status'], 'output': str(path)}))
    else:
        report = audit_installation(runtime)
        try:
            report['model'] = model_evidence(runtime, args.model, args.scale)
        except RuntimeError as error:
            report['model'] = {'status': 'missing_weights', 'reason': str(error)}
        args.output.mkdir(parents=True)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps({'status': report['status'], 'model': report['model']['status'],
            'neuroserver': report['neuroserver_directory_present'], 'gpu': False}))


if __name__ == '__main__':
    main()
