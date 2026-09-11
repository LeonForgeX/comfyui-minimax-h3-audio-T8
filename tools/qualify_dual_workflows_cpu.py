"""Actual Core schemas/validation and generated workflow roundtrip, no queue/UI."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    project = Path(__file__).resolve().parents[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_NUM_THREADS'] = '2'
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = ['dual-workflow-cpu', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import nodes
    import execution
    import torch
    spec = importlib.util.spec_from_file_location('dual_workflow_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from tools.build_dual_model_workflows import build_prompt, build_workflow, NODE, LORA, PLAN
    from tools.audit_progressive_workflows import audit_candidate
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    info = {}
    for name in ('UNETLoader', 'VAELoader', 'CLIPLoader', NODE, LORA, PLAN):
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        if hasattr(cls, 'GET_NODE_INFO_V1'):
            info[name] = cls.GET_NODE_INFO_V1()
        else:
            info[name] = {'name': name, 'input': cls.INPUT_TYPES(), 'output': cls.RETURN_TYPES,
                'output_name': getattr(cls, 'RETURN_NAMES', cls.RETURN_TYPES), 'display_name': name}
    # JSON transport is what the browser actually receives (tuples become lists).
    info = json.loads(json.dumps(info))
    args.output.mkdir(parents=True)
    report = {'status': 'incomplete', 'registered_nodes': len(classes), 'cases': {},
              'cuda_initialized': torch.cuda.is_initialized(), 'queued': False, 'browser_used': False}
    for name, relay in [('Plain', False), ('Relay', True)]:
        graph, workflow = build_prompt(info, relay=relay), build_workflow(info, relay=relay)
        valid = asyncio.run(execution.validate_prompt('dual-' + name, graph, None))
        if not valid[0]:
            raise ValueError(f'{name} API rejected: {valid}')
        audit = audit_candidate(graph, workflow, info)
        report['cases'][name] = {'validation': valid, 'serialization': audit}
        for suffix, payload in [('prompt.json', graph), ('workflow.json', workflow)]:
            (args.output / f'{name}.{suffix}').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    report['cuda_initialized'] = torch.cuda.is_initialized()
    report['cuda_available'] = torch.cuda.is_available()
    report['status'] = ('failed_cuda_guard' if report['cuda_initialized'] or report['cuda_available']
                       else 'actual_Core_API_and_serialization_pass_not_browser_roundtrip')
    (args.output / 'object-info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if report['status'] == 'failed_cuda_guard':
        raise RuntimeError('CPU-only workflow validation unexpectedly accessed CUDA')


if __name__ == '__main__':
    main()
