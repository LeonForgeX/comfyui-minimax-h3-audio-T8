"""Validate actual browser-exported API graphs against current Core without execution."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['T2VA', 'I2VA'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite evidence')
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = ['native-export-validation', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import nodes
    import execution
    import torch
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    spec = importlib.util.spec_from_file_location('native_exports_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    result = {'status': 'incomplete', 'cases': {}, 'registered_project_nodes': len(classes),
              'generation_queued': False, 'cuda_initialized': torch.cuda.is_initialized()}
    for task in args.cases:
        if task not in {'T2VA', 'I2VA', 'DualPlain', 'DualRelay'}:
            raise ValueError('Unsupported named UI case')
        graph = json.loads((args.audit / f'{task}.api.json').read_text(encoding='utf-8'))
        valid = asyncio.run(execution.validate_prompt('native-export-' + task, graph, None))
        if not valid[0] or valid[3]:
            raise ValueError(f'{task} rejected: {valid}')
        result['cases'][task] = valid
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    result['status'] = 'actual_browser_API_exports_Core_CPU_validation_pass'
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
