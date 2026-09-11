"""Inspect the selected official installation and model catalog without inference."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('install', 'definitions', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new task-owned artifact directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    package = types.ModuleType('topaz_catalog_probe')
    package.__path__ = [str(project / 'h3_t8')]
    sys.modules[package.__name__] = package
    runtime_module = importlib.import_module(package.__name__ + '.topaz_runtime')
    sources = [project / 'h3_t8' / name for name in
               ('topaz_runtime.py', 'topaz_contract.py', 'topaz_media.py')]
    before = [runtime_module.file_identity(path) for path in sources]
    runtime = runtime_module.OfficialTopaz(args.install, args.definitions, args.data)
    report = runtime_module.audit_installation(runtime)
    if before != [runtime_module.file_identity(path) for path in sources]:
        raise RuntimeError('Source changed during catalog inspection')
    report['probe_sources'] = before
    report['torch_imported'] = 'torch' in sys.modules
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    catalog = report['model_catalog']
    print(json.dumps({'status': report['status'], 'models': len(catalog['models']),
                      'nonmodels_or_errors': len(catalog['unreadable_or_nonmodel_definitions']),
                      'torch_imported': report['torch_imported'], 'report': str(output / 'report.json')}))


if __name__ == '__main__':
    main()
