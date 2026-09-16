"""Build candidate native workflows without opening or exporting the user's canvas."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_to_frontend_workflow import convert  # noqa: E402
from audit_progressive_workflows import audit_candidate  # noqa: E402
from run_fast_h3_v2_probe import MODEL, REVISION, build_graph  # noqa: E402


def recipes(reference="t8_h4c2_bund_korean_mv_ref_20260916.png"):
    cases = {
        'FastH3_V2_Trained_VSA_T2VA_EXP': dict(profile='trained_vsa_exp', frames=124),
        'FastH3_V2_Dense_T2VA_EXP': dict(profile='dense_compat_exp', frames=73),
        'FastH3_V2_Official_Comfy_Template_EXP': dict(profile='official_comfy_template_exp', frames=124),
        'FastH3_V2_Dense_First_Frame_T8_Memory_EXP': dict(profile='dense_compat_exp', frames=73,
            width=512, height=768, head_chunks=4, ffn_chunks=2, first_frame=reference),
    }
    return {name: build_graph(**config, min_tokens=12288, instrument=False)[0]
            for name, config in cases.items()}


def selected_frontend_schema(graph, info):
    """Materialize selected native DynamicCombo widgets without inventing options."""
    graph, info = deepcopy(graph), deepcopy(info)
    for node in graph.values():
        schema = info[node['class_type']]
        specs = schema.get('input', {})
        if not any(spec[0] == 'COMFY_DYNAMICCOMBO_V3' for group in specs.values()
                   for spec in group.values() if isinstance(spec, list) and spec):
            continue
        values = node['inputs']
        if node['class_type'] == 'SaveVideo' and 'codec' in values:
            values['format.codec'] = values.pop('codec')
        expanded = {'required': {}, 'optional': {}, 'hidden': deepcopy(specs.get('hidden', {}))}
        def add(name, spec, section):
            options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if options.get('hidden'):
                return
            if spec[0] != 'COMFY_DYNAMICCOMBO_V3':
                expanded[section][name] = deepcopy(spec)
                return
            choices = options['options']
            keys = [choice['key'] for choice in choices]
            selected = values.get(name, options.get('default', keys[0]))
            if selected not in keys:
                raise ValueError(f'Invalid native dynamic choice: {name}')
            values[name] = selected
            expanded[section][name] = [keys, {**options, 'default': keys[0]}]
            branch = next(choice for choice in choices if choice['key'] == selected)
            for child_section in ('required', 'optional'):
                for child, child_spec in branch['inputs'].get(child_section, {}).items():
                    add(f'{name}.{child}', child_spec, child_section)
        for section in ('required', 'optional'):
            for field, spec in specs.get(section, {}).items():
                add(field, spec, section)
        schema['input'] = expanded
        schema['input_order'] = {key: list(group) for key, group in expanded.items()}
    return graph, info


def build_workflow(graph, info, name):
    graph, info = selected_frontend_schema(graph, info)
    workflow = convert(graph, info, name)
    note_id = workflow['last_node_id'] + 1
    text = (
        'FastH3 V2 FULL STUDENT — EXP candidate, not a reviewed recommendation.\n\n'
        'Do not add the old EMA/Turbo acceleration LoRA. Reuse H3 Qwen + video/audio VAEs. '
        'Setup MODEL/SAMPLER/SIGMAS → BasicGuider CFG1 + SamplerCustomAdvanced. '
        'Decode final sigma0 output, not an intermediate x0.\n\n'
        'trained_vsa_exp: exact trained DMD8, AV shifts10/3, learned VSA keep20%. '
        'dense_compat_exp: same clocks but explicit Dense compatibility, not trained VSA. '
        'official_comfy_template_exp: simple8/res_multistep, keep10%, start20%; NOT the same recipe.\n\n'
        'Default min_tokens12288 may produce reported Dense for small layouts. '
        'Reference / LoRA / dual4+upscale+4 / loop are undistilled EXP. '
        'Choose a matching reference aspect:2:3 means512×768, never stretch. '
        'Place a selected local reference in ComfyUI/input.\n\n'
        f'Model: https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/tree/{REVISION}/diffusion_models\n'
        f'Filename: {MODEL}\n'
        'Docs: docs/FAST_H3_V2_EXP.md. No universal16GB, speed or quality promise.'
    )
    workflow['nodes'].append(dict(id=note_id, type='MarkdownNote', title='FastH3 V2 · Read first',
        pos=[0, -640], size=[900, 590], flags={}, order=len(graph), mode=0,
        inputs=[], outputs=[], properties={}, widgets_values=[text]))
    workflow['last_node_id'] = note_id
    return workflow, audit_candidate(graph, workflow, info), graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-info', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists() or not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Build into a new candidate artifact directory; never publish unreviewed graphs')
    info = json.loads(args.object_info.read_text(encoding='utf8'))
    root.mkdir(parents=True)
    receipts = {}
    for name, graph in recipes().items():
        workflow, receipts[name], graph = build_workflow(graph, info, name)
        for suffix, value in [('api.json', graph), ('json', workflow)]:
            with (root / f'{name}.{suffix}').open('x', encoding='utf8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
    with (root / 'audit.json').open('x', encoding='utf8') as stream:
        json.dump(dict(status='candidate_serialization_pass_not_live_or_human_review', cases=receipts), stream, indent=2)
    print(json.dumps(receipts), flush=True)


if __name__ == '__main__':
    main()
