"""Validate existing SelfLift EAV/TST graphs in actual Core; never queue/infer."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def api_from_workflow(workflow, classes):
    """Read actual schema order and saved links, including seed-control widgets."""
    edges = {edge[0]: edge for edge in workflow['links']}
    graph = {}
    for node in workflow['nodes']:
        if node['type'] == 'MarkdownNote':
            continue
        if node.get('mode', 0) != 0:
            raise ValueError('Only active accepted control graphs are qualified')
        cls = classes[node['type']]
        specs = cls.INPUT_TYPES()
        values = iter(node.get('widgets_values', []))
        inputs = {}
        pins = {pin['name']: pin for pin in node.get('inputs', [])}
        for section in ('required', 'optional'):
            for name, spec in specs.get(section, {}).items():
                kind, options = spec[0], spec[1] if len(spec) > 1 else {}
                widget = not options.get('forceInput') and (
                    isinstance(kind, list) or kind in {'INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO'})
                if widget:
                    try:
                        value = next(values)
                    except StopIteration:
                        if section != 'optional' or 'default' not in options:
                            raise ValueError(f'Missing saved widget: {node["type"]}.{name}') from None
                        value = options['default']
                    if options.get('control_after_generate'):
                        if next(values) != 'fixed':
                            raise ValueError('Control recipe seed must remain fixed')
                    inputs[name] = value
                edge_id = pins.get(name, {}).get('link')
                if edge_id is not None:
                    edge = edges[edge_id]
                    inputs[name] = [str(edge[1]), edge[2]]
        # A core widget (e.g. LoadImage's upload control) is not an execution
        # input. It must be explicitly known, never confused with a new option.
        leftovers = list(values)
        if leftovers and not (node['type'] == 'LoadImage' and leftovers == ['image']):
            raise ValueError(f'Unaccounted saved widgets: {node["type"]}: {leftovers}')
        graph[str(node['id'])] = {'class_type': node['type'], 'inputs': inputs}
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(project / 'artifacts') or destination.exists():
        raise ValueError('Choose a new JSON within project artifacts')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path[:0] = [str(args.core), str(project)]
    sys.argv = ['selflift-interfaces-cpu', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import execution
    import torch
    from comfy_extras import nodes_preview_any
    nodes.NODE_CLASS_MAPPINGS.update(nodes_preview_any.NODE_CLASS_MAPPINGS)
    spec = importlib.util.spec_from_file_location('selflift_core_qualification', project / '__init__.py',
                                                submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    reports = []
    for suffix in ('EAV', 'TST', 'Guide_Mean'):
        path = project / 'examples/workflows/33-selflift-taomate' / (
            f'2026-09-14_H3_SelfLift_I2VA_{suffix}_4plus4_EXP.json')
        payload = path.read_bytes()
        graph = api_from_workflow(json.loads(payload), nodes.NODE_CLASS_MAPPINGS)
        result = asyncio.run(execution.validate_prompt('selflift-contract-cpu', graph, None))
        if not result[0]:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        reports.append({'workflow': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                        'execution_nodes': len(graph), 'validation': result})
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization')
    report = {'status': 'actual_Core_saved_SelfLift_graphs_pass_not_inference_or_browser',
              'registered_nodes': len(classes), 'graphs': reports,
              'cuda_initialized': False, 'queued': False, 'browser_used': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
