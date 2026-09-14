"""Decode/probe an existing source with formal FFprobe; no enhancement/download."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def load_module(name):
    path = Path(__file__).resolve().parents[1] / 'h3_t8' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('install', 'definitions', 'data', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output evidence directory')
    contract, media = load_module('topaz_contract'), load_module('topaz_media')
    runtime = contract.OfficialTopaz(args.install, args.definitions, args.data)
    identity = media.file_identity(args.source)
    args.output.mkdir(parents=True)
    probes = {}
    for kind in ('frames', 'packets'):
        with (args.output / (kind + '.json')).open('xb') as stdout:
            result = subprocess.run(media.probe_command(runtime, args.source, **{kind: True}),
                cwd=runtime.install, env=runtime.child_environment(os.environ), stdout=stdout,
                stderr=subprocess.PIPE, timeout=180, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            raise RuntimeError('Official ffprobe failed: ' + result.stderr.decode('utf8', 'replace')[-2000:])
        probes[kind] = json.loads((args.output / (kind + '.json')).read_text(encoding='utf8'))
    timeline = media.analyze_video(
        probes['frames'], allowed_bit_depths=media.AUTOMATIC_H264_INPUT_BIT_DEPTHS
    )
    if media.file_identity(args.source) != identity:
        raise RuntimeError('Source changed during probing')
    report = {'status': 'source_cpu_probe_pass_not_enhancement', 'source': identity,
        'timeline': timeline, 'audio': media.compare_audio_packets(probes['packets'], probes['packets'], '0'),
        'gpu_inference': False, 'downloads': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'timeline'}))


if __name__ == '__main__':
    main()
