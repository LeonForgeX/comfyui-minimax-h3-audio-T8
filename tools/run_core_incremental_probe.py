"""CPU-only incremental H3 interface checks in one exact existing Core tree."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-root', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--source-repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, repo = args.core_root.resolve(strict=True), args.source_repository.resolve(strict=True)
    if args.output.exists():
        raise ValueError('Use a new evidence directory')
    project = Path(__file__).resolve().parents[1]
    paths = ('comfy/ldm/minimax/model.py', 'comfy/model_patcher.py',
        'comfy/model_prefetch.py', 'comfy/ldm/modules/attention.py',
        'comfy_extras/nodes_model_advanced.py', 'comfy/utils.py')
    sources = {}
    for relative in paths:
        reference = subprocess.run(['git', '-C', str(repo), 'show', args.revision + ':' + relative],
            capture_output=True, check=True).stdout
        local = (root / relative).read_bytes()
        if local.replace(b'\r\n', b'\n') != reference.replace(b'\r\n', b'\n'):
            raise ValueError('Core source differs from stated revision: ' + relative)
        sources[relative] = hashlib.sha256(local).hexdigest()
    args.output.mkdir(parents=True)
    sys.path[:] = [str(root), str(project), *[p for p in sys.path
        if p not in (str(project / 'tools'), str(root), str(project))]]
    os.environ.update(PYTHONPATH=os.pathsep.join((str(root), str(project))),
        CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.argv = ['core-incremental', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import pytest
    import torch
    torch.set_num_threads(2)
    tests = ['test_h3_core_compat.py', 'test_core_final_layer_partial_schedule.py',
        'test_h3_attention_ownership.py', 'test_prompt_relay_core_compat.py',
        'test_h3_block_cache_compat.py']
    code = int(pytest.main([*[str(project / 'tests' / name) for name in tests],
        '-q', '--tb=short', '--junitxml=' + str(args.output / 'pytest.xml')]))
    modules = ('comfy.ldm.minimax.model', 'comfy.model_patcher', 'comfy.model_prefetch',
        'comfy.ldm.modules.attention', 'comfy.utils')
    origins = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
    unchanged = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in sources.items())
    valid = code == 0 and unchanged and not torch.cuda.is_initialized() and all(
        Path(path).is_relative_to(root) for path in origins.values())
    report = {'status': 'pass' if valid else 'failed', 'revision': args.revision,
        'source_sha256': sources, 'core_sources_unchanged': unchanged, 'origins': origins,
        'pytest_exit_code': code, 'cuda_initialized': torch.cuda.is_initialized(), 'tests': tests,
        'scope': 'targeted interfaces on exact Core sources; current host dependencies, not old dependency pins or GPU allocator graphs'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
