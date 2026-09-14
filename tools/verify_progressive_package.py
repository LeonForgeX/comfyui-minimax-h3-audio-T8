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


def validate_registry(ids, feature_ids):
    # Fixed released v1.79.6/e12d8af prefix; new nodes append, never replace it.
    prefix = hashlib.sha256(json.dumps(ids[:331], separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if (ids != feature_ids or len(ids) != len(set(ids)) or len(ids) != 334
            or prefix != 'd49dca3dadf4898dc2b1fbd607c8873b66ddce6b245e178ed25e06398085b915'
            or ids[331:] != ['MiniMaxH3TSTModelEXPT8', 'MiniMaxH3ProgressiveSetupEXPT8', 'MiniMaxH3ProgressiveLongVideoEXPT8']):
        raise ValueError('Packaged node schema differs from released prefix plus declared SelfLift additions')


def validate_workflows(names, members):
    expected = {name for name in members if name.startswith('examples/workflows/') and name.endswith('.json')}
    if not expected or set(names) != expected or len(names) != len(expected) or len(names) != 254:
        raise ValueError('Packaged workflow membership differs from frozen source receipt')
    selflift = {name.rsplit('/', 1)[-1] for name in names if name.startswith('examples/workflows/33-selflift-taomate/')}
    required = {
        '2026-09-14_H3_SelfLift_I2VA_Core_Sage_4plus4_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_EAV_4plus4_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_Guide_Mean_4plus4_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_KJ_FFN_TST_EAV_Relay_4plus4_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_KJ_Relay_Two_Segment_8s_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_Sol_4plus4_EXP.json',
        '2026-09-14_H3_SelfLift_I2VA_TST_4plus4_EXP.json',
        '2026-09-14_H3_TaoMate_T2VA_3step_EXP.json',
        '2026-09-14_H3_TaoMate_T2VA_4step_EXP.json',
    }
    if selflift != required:
        raise ValueError('Packaged SelfLift/TaoMate workflow set differs')


def verify(root, core=None):
    root = Path(root).resolve(strict=True)
    receipt = json.loads((root/'receipt.json').read_text(encoding='utf8'))
    if (receipt.get('version') != '1.80.0' or receipt.get('human_qualified') is not True
            or receipt.get('universal_quality_claim') is not False
            or receipt.get('workflow_json_count') != 254
            or len(receipt.get('selflift_workflows', {})) != 9):
        raise ValueError('Candidate receipt is not the scoped human-reviewed v1.80.0 delivery')
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
    validate_registry(ids, features['nodes'])
    origins = {}
    for name, loaded in list(sys.modules.items()):
        location = getattr(loaded, '__file__', None)
        if name.startswith(spec.name) and location:
            if not Path(location).resolve().is_relative_to(package.resolve()):
                raise ValueError('Package import escaped extraction')
            origins[name] = str(location)
    workflows = list((package/'examples/workflows').rglob('*.json'))
    validate_workflows([p.relative_to(package).as_posix() for p in workflows], receipt['files'])
    for workflow in workflows:
        json.loads(workflow.read_text(encoding='utf8'))
    pending = ('UNREVIEWED', '未人审', '尚待人审', '仍需CPU/UI复核', '未通过不晋级')
    for workflow in workflows:
        relative = workflow.relative_to(package).as_posix()
        if relative.startswith('examples/workflows/33-selflift-taomate/'):
            text = workflow.read_text(encoding='utf8')
            if any(token in text for token in pending):
                raise ValueError('Promoted workflow retains pending-review marker: '+relative)
            review = json.loads(text).get('extra', {}).get('t8_bound_review', {})
            if (review.get('status') != 'accepted_in_this_review_scope'
                    or review.get('not_universal_quality_claim') is not True):
                raise ValueError('Promoted workflow lacks scoped review metadata: '+relative)
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
    result = {'status': 'actual_archive_CPU_schema_workflow_and_bound_review_pass', 'nodes': len(ids), 'workflow_json': len(workflows),
        'archive_sha256': receipt['archive_sha256'], 'package_origins': origins, 'gpu_initialized': torch.cuda.is_initialized(),
        'published': False, 'public_FI_node_included': True, 'human_qualified': True,
        'human_review_scope': receipt['human_review_scope'], 'universal_quality_claim': False,
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
