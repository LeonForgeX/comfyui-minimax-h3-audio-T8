"""Actual Core validation of the saved plain-backend chunked4+4 graph; no inference/UI."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
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
    project = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Use a new artifact directory inside this project')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path[:0] = [str(args.core), str(project), str(project / 'tools')]
    sys.argv = ['chunked-workflow-cpu', '--cpu']
    import comfy.options
    comfy.options.enable_args_parsing()
    import nodes
    import execution
    import torch
    from comfy_extras import nodes_custom_sampler, nodes_lora_debug, nodes_preview_any, nodes_video
    spec = importlib.util.spec_from_file_location('chunked_workflow_pkg', project / '__init__.py',
        submodule_search_locations=[str(project)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    for module in (nodes_preview_any, nodes_video, nodes_custom_sampler, nodes_lora_debug):
        if hasattr(module, 'comfy_entrypoint'):
            extension = asyncio.run(module.comfy_entrypoint())
            for cls in asyncio.run(extension.get_node_list()):
                nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
        else:
            nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    for cls in classes:
        nodes.NODE_CLASS_MAPPINGS[cls.define_schema().node_id] = cls
    from run_chunked_parity_probe import build_graph
    from audit_progressive_workflows import audit_candidate
    from run_progressive_pilot import write_json
    graph = build_graph()
    graph.pop('26')  # Explicitly validate the formal plain MODEL route, no KJ.
    graph['8']['inputs']['model'] = ['23',0]
    workflow_path = project / 'examples/workflows/13-latent-upscale/2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json'
    w = json.loads(workflow_path.read_text(encoding='utf-8'))
    # Canonicalize IDs only in the audit copy, never modify the saved workflow.
    remap = {int(key): i+1 for i,key in enumerate(graph)}
    checked = deepcopy(w)
    checked['nodes'] = [n for n in checked['nodes'] if n['type'] != 'MarkdownNote']
    for node in checked['nodes']:
        node['id'] = remap[node['id']]
    for edge in checked['links']:
        edge[1], edge[3] = remap[edge[1]], remap[edge[3]]
    info = {}
    for node in graph.values():
        name = node['class_type']
        cls = nodes.NODE_CLASS_MAPPINGS[name]
        info[name] = cls.GET_NODE_INFO_V1() if hasattr(cls,'GET_NODE_INFO_V1') else {
            'input': cls.INPUT_TYPES(), 'output': cls.RETURN_TYPES,
            'output_name': getattr(cls,'RETURN_NAMES',cls.RETURN_TYPES)}
    info = json.loads(json.dumps(info))
    # SaveVideo's nested DynamicCombo serializes as four scalar widgets, not
    # one JSON object. Flatten only the selected real schema for this audit;
    # actual Core validation below receives the exact nested API object.
    audit_graph, audit_info = deepcopy(graph), deepcopy(info)
    save = graph['25']['inputs']['format']
    def option(spec, key):
        return next(item for item in spec[1]['options'] if item['key']==key)
    format_spec = info['SaveVideo']['input']['required']['format']
    codec_spec = option(format_spec, save['format'])['inputs']['required']['codec']
    encoding_spec = option(codec_spec, save['codec']['codec'])['inputs']['optional']['encoding']
    if option(encoding_spec, save['codec']['encoding']['encoding'])['inputs'].get('required'):
        raise ValueError('Additional dynamic encoding widgets are not qualified')
    audit_info['SaveVideo']['input']['required']['format'] = [
        [item['key'] for item in format_spec[1]['options']], {'default':'auto'}]
    audit_info['SaveVideo']['input']['optional'] = {
        'codec': [[item['key'] for item in codec_spec[1]['options']], {'default':'auto'}],
        'encoding': [[item['key'] for item in encoding_spec[1]['options']], {'default':'auto'}]}
    audit_graph['25']['inputs'].update(format=save['format'], codec=save['codec']['codec'],
        encoding=save['codec']['encoding']['encoding'])
    serialization = audit_candidate(audit_graph, checked, audit_info)
    valid = asyncio.run(execution.validate_prompt('chunked-parity-plain-cpu', graph, None))
    if not valid[0] or torch.cuda.is_initialized():
        raise RuntimeError(f'CPU graph validation failed: {valid}')
    args.output.mkdir(parents=True)
    report = {'status':'actual_Core_API_and_serialization_pass_not_browser_roundtrip',
        'registered_nodes':len(classes),
        'registered_node_ids':[cls.define_schema().node_id for cls in classes],
        'validation':valid, 'serialization':serialization,
        'cuda_initialized':torch.cuda.is_initialized(), 'queued':False, 'browser_used':False}
    write_json(args.output / 'report.json',report)
    write_json(args.output / 'prompt.json',graph)
    write_json(args.output / 'object-info.json',info)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
