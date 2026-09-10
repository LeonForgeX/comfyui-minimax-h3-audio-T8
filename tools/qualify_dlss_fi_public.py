"""One explicit serial public-node qualification; no browser or server changes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'artifacts/acceleration-research-20260909'


def run(output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(RESEARCH):
        raise ValueError('New research output required')
    # CPU Core mode prevents parent model allocation. Do not hide the physical
    # device from the child's CUDA LUID enumeration (which creates no context).
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    sys.path.insert(0, str(ROOT.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import folder_paths
    from comfy_api.latest import InputImpl
    spec = importlib.util.spec_from_file_location('_fi_qualify', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from _fi_qualify.nodes_dlss_fi import MiniMaxH3DLSSFrameInterpolationEXPT8 as Node
    from _fi_qualify.dlss_fi_backend.resources import file_identity
    source = json.loads((RESEARCH/'dlss-fi-file-probe-20260910-v1/source.json').read_text())['file']
    if file_identity(source['path']) != source:
        raise ValueError('Existing comparison source changed')
    runtime = RESEARCH/'upstreams/dlss-interpolation/bin/runtime/dlssg'
    output.mkdir(parents=True)
    original = folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(output))  # This isolated CPU parent only, not user's server.
    try:
        result = Node.execute(InputImpl.VideoFromFile(source['path']), str(runtime), '', 240)
        report = json.loads(result.result[3])
        old = json.loads((RESEARCH/'dlss-fi-file-probe-20260910-v1/result.json').read_text())
        report['byte_identical_to_previous_game'] = file_identity(result.result[2])['sha256'] == old['media']['file']['sha256']
        with (output/'qualification.json').open('x', encoding='utf8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps({'status': report['status'], 'same_bytes': report['byte_identical_to_previous_game'],
                          'output': result.result[2], 'isolation': report['isolation'], 'counts': report['ledger']['counts']}))
    finally:
        folder_paths.set_output_directory(original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
