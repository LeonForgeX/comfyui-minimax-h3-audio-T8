"""Import the actual extracted candidate on CPU; no GPU, installer or frontend."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def verify(root, core=None):
    root = Path(root).resolve(strict=True)
    receipt = json.loads((root/'receipt.json').read_text(encoding='utf8'))
    package = Path(receipt['extracted'])
    project = Path(__file__).resolve().parents[1]
    core = Path(core).resolve() if core is not None else next(
        (p for p in project.parents if (p / 'comfy/cli_args.py').is_file()), None)
    if core is None or not (core / 'comfy/cli_args.py').is_file():
        raise ValueError('Pass --core with the actual ComfyUI checkout for an external release worktree')
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(receipt['archive']) != receipt['archive_sha256'] or any(sha(package/n) != d for n, d in receipt['files'].items()):
        raise ValueError('Archive/extracted source changed')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(core))
    sys.argv = ['candidate-import-check', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    if torch.cuda.is_available():
        raise ValueError('Package verification must be CPU-only')
    spec = importlib.util.spec_from_file_location('progressive_candidate_pkg', package/'__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    nodes = asyncio.run(module.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in nodes]
    features = json.loads((package/'features.json').read_text(encoding='utf8'))
    if ids != features['nodes'] or len(ids) != len(set(ids)) or len(ids) != 327 or ids[318:] != [
            'MiniMaxH3ProgressiveSamplerEXPT8', 'MiniMaxH3DLSSFrameInterpolationEXPT8',
            'MiniMaxH3TRTVAECheckEXPT8', 'MiniMaxH3TRTVAEDecoderEXPT8',
            'MiniMaxH3TRTVAEFullEXPT8', 'MiniMaxH3TRTVAECompileEXPT8',
            'MiniMaxH3DualModelLongVideoEXPT8', 'MiniMaxH3TopazEnvironmentEXPT8',
            'MiniMaxH3TopazVideoEXPT8']:
        raise ValueError('Packaged node schema list/order differs')
    origins = {}
    for name, loaded in list(sys.modules.items()):
        location = getattr(loaded, '__file__', None)
        if name.startswith(spec.name) and location:
            if not Path(location).resolve().is_relative_to(package.resolve()):
                raise ValueError('Package import escaped extraction')
            origins[name] = str(location)
    workflows = list((package/'examples/workflows').rglob('*.json'))
    if len(workflows) != 236:
        raise ValueError('Packaged workflow count differs')
    for workflow in workflows:
        json.loads(workflow.read_text(encoding='utf8'))
    for task in ('T2VA', 'I2VA'):
        name = f'examples/workflows/28-progressive-sampling/2026-09-09_H3_Progressive_{task}_6plus2_EXP.json'
        if sha(project/name) != receipt['files'][name]:
            raise ValueError('Packaged workflow differs from project delivery directory')
    if not (package/'docs/PROGRESSIVE_SAMPLING_EXP.md').is_file() or (package/'tools').exists():
        raise ValueError('Required guide missing or research tools included')
    # Import and execute the shipped worker through its actual parent entry.
    # Deliberately malformed CPU fixture must fail before any device query/worker.
    from importlib import import_module
    backend = import_module(spec.name+'.dlss_fi_backend.entry')
    process = import_module(spec.name+'.dlss_fi_backend.process')
    invalid = root/'invalid-cpu-fi-input.mp4'
    with invalid.open('xb') as output:
        output.write(b'CPU import qualification: deliberately invalid media')
    try:
        backend.process_file(invalid, package, root, timeout=30)
    except process.IsolatedTaskError as error:
        fi_failure = error.receipt
        if (fi_failure['status'] != 'child_failed' or fi_failure['active_after_cleanup'] or
                'InvalidDataError' not in fi_failure['stderr_tail'] or
                Path(backend.__file__).resolve().parent != package/'h3_t8/dlss_fi_backend'):
            raise ValueError('Packaged worker did not fail at the expected isolated CPU media check') from error
    else:
        raise ValueError('Malformed fixture unexpectedly succeeded')
    if sha(receipt.get('index_path', project/'.git/index')) != receipt['main_index_sha256']:
        raise ValueError('User index changed')
    result = {'status': 'actual_archive_CPU_schema_and_workflow_pass', 'nodes': len(ids), 'workflow_json': len(workflows),
        'archive_sha256': receipt['archive_sha256'], 'package_origins': origins, 'gpu_initialized': torch.cuda.is_initialized(),
        'published': False, 'public_FI_node_included': True, 'human_qualified': False,
        'FI_packaged_worker_CPU_invalid_media_cleanup': fi_failure}
    if result['gpu_initialized']:
        raise ValueError('CPU package import initialized CUDA')
    with (root/'import-receipt.json').open('x', encoding='utf8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    return {k: v for k, v in result.items() if k != 'package_origins'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--core', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.core)))
