"""Read-only official Topaz inventory; optional FFmpeg help, never enhancement.

No licenses, login state or authorization files are read. No downloads,
installer, Neuroserver or GPU inference are invoked by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


WEIGHT_SUFFIXES = {'.tz', '.tz3', '.onnx', '.safetensors', '.engine', '.plan', '.zip'}


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def inventory(install, definitions, data, *, probe_help=False):
    install, definitions, data = (Path(p).resolve(strict=True) for p in (install, definitions, data))
    executables = {name: identity(install / name) for name in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe')}
    models = []
    for path in sorted(definitions.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or 'backends' not in record:
            continue
        models.append({'id': path.stem, 'name': record.get('name'),
            'definition': identity(path), 'neuroserver': record.get('isNeuroserverModel', False),
            'parameters': record.get('parameters', []),
            'backends': record['backends'],
            'availability': 'definition_only_not_execution_proven'})
    weights = [{'path': str(path), 'bytes': path.stat().st_size}
               for path in sorted(data.rglob('*')) if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES]
    neuro = install / 'neuroserver'
    result = {'status': 'static_inventory_not_runtime_qualification', 'install': str(install),
        'definition_directory': str(definitions), 'weight_directory': str(data),
        'executables': executables, 'models': models, 'weight_files': weights,
        'neuroserver_directory_present': neuro.is_dir(),
        'neuroserver_version_file_present': (neuro / 'version.json').is_file(),
        'planned_child_environment': {'TVAI_MODEL_DIR': str(definitions), 'TVAI_MODEL_DATA_DIR': str(data)},
        'authorization': 'not_read_or_modified', 'downloads': False, 'gpu_inference': False}
    if probe_help:
        # Restrict to executable help, a filter is never applied to input media.
        env = os.environ.copy()
        env.update(result['planned_child_environment'])
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        completed = subprocess.run([str(install / 'ffmpeg.exe'), '-hide_banner', '-h', 'filter=tvai_up'],
            cwd=install, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=20, creationflags=flags, check=True)
        result['tvai_up_help'] = completed.stdout + completed.stderr
        result['tvai_up_advertised'] = 'Filter tvai_up' in result['tvai_up_help']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', type=Path, required=True)
    parser.add_argument('--definitions', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--help-probe', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new report, do not overwrite previous evidence')
    result = inventory(args.install, args.definitions, args.data, probe_help=args.help_probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'models': len(result['models']),
        'weights': len(result['weight_files']), 'neuroserver': result['neuroserver_directory_present'],
        'report': str(args.output)}))


if __name__ == '__main__':
    main()
