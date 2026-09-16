"""Verify native UI exports from observed file-chooser/menu interactions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

from audit_progressive_workflows import audit_api, audit_candidate, widget
from build_fast_h3_v2_accepted_workflows import CASES, sha
from build_fast_h3_v2_workflows import selected_frontend_schema
from package_fast_h3_v2_candidate import ACCEPTED_V2_WORKFLOWS
from run_progressive_pilot import write_json

PROJECT = Path(__file__).resolve().parents[1]


def canonical_task_widgets(workflow, info):
    """Normalize documented non-inference UI fields, retaining raw exports."""
    source = PROJECT / 'web/task_type_labels.js'
    labels = dict(re.findall(r'\["([^"]+)", "([^"]+)"\]', source.read_text(encoding='utf8')))
    if set(labels) != {'auto', 'T2VA', 'I2VA', 'FL2VA', 'L2VA', 'Ref2VA', 'Hybrid'}:
        raise ValueError('Unexpected frontend localization contract')
    reverse = {v:k for k,v in labels.items()}
    normalized, rows = deepcopy(workflow), []
    for node in normalized['nodes']:
        if node['type'] == 'LoadImage':
            spec = info['LoadImage']['input']['required']['image']
            values = node.get('widgets_values', [])
            if len(values) == 2 and values[-1] == 'image' and spec[1].get('image_upload') is True:
                # Native upload button stores its type marker, not a backend
                # parameter. The image filename is still checked verbatim.
                node['widgets_values'] = values[:-1]
                rows.append(dict(node=node['id'], upload_button_marker='image'))
        if node['type'] != 'MarkdownNote' and 'widgets_values' not in node:
            groups = info[node['type']].get('input', {})
            if any(widget(spec) for section in ('required', 'optional') for spec in groups.get(section, {}).values()):
                raise ValueError('Native export omitted a nonempty widget array')
            node['widgets_values'] = []
        if node['type'] not in ('MiniMaxH3AudioConditioningT8', 'MiniMaxH3LongVideoConditioningT8'):
            continue
        pos = 0
        for section in ('required', 'optional'):
            for name, spec in info[node['type']].get('input', {}).get(section, {}).items():
                if not widget(spec):
                    continue
                if name == 'task_type':
                    value = node['widgets_values'][pos]
                    if value in reverse:
                        node['widgets_values'][pos] = reverse[value]
                        rows.append(dict(node=node['id'], display=value, canonical=reverse[value]))
                    elif value not in labels:
                        raise ValueError('Unknown task_type widget; do not guess auto')
                options = spec[1] if len(spec)>1 and isinstance(spec[1],dict) else {}
                pos += 1 + bool(options.get('control_after_generate', name in {'seed','noise_seed'}))
    return normalized, dict(source_sha256=sha(source), display_only_normalizations=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--object-info', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(PROJECT / 'artifacts') or not root.is_dir():
        raise ValueError('Existing task-owned evidence directory required')
    info = json.loads(args.object_info.read_text(encoding='utf8'))
    exports = root / 'exports'
    exports.mkdir()
    result = dict(status='incomplete', cases=[], generation=False,
        provenance='Observed cua_repl native Ctrl+O file chooser and export/API menu on isolated CPU8209; no app injection',
        browser='IAB tabs14/15; Chrome8214 client-blocked attempt retained separately')
    try:
        for i, case in enumerate(CASES, 1):
            name = case[0]
            input_path = PROJECT / 'examples/workflows/10-speed' / (name + '.json')
            assert sha(input_path) == ACCEPTED_V2_WORKFLOWS[input_path.relative_to(PROJECT).as_posix()]
            prompt_path = PROJECT / 'artifacts/fasth3-v2-six-accepted-workflows-v1' / (name + '.api.json')
            prompt = json.loads(prompt_path.read_text(encoding='utf8'))
            graph, selected = selected_frontend_schema(prompt, info)
            api_schema = deepcopy(selected)
            front_schema = deepcopy(selected)
            # Core exports some compatibility fields despite hiding their UI.
            # Accept only the native schema's exact default, never an arbitrary extra.
            for node in graph.values():
                cls = node['class_type']
                for section in ('required', 'optional'):
                    for key, spec in info[cls].get('input', {}).get(section, {}).items():
                        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                        if options.get('hidden'):
                            native = deepcopy(spec)
                            if 'default' not in options and spec[0] == 'COMFY_DYNAMICCOMBO_V3':
                                choices = options.get('options', [])
                                if choices and isinstance(choices[0], dict) and 'key' in choices[0]:
                                    # Native DynamicCombo defaults to its first option.
                                    # Core SaveVideo's hidden legacy codec is auto; its
                                    # format.codec still takes precedence in execute().
                                    native[1]['default'] = choices[0]['key']
                            if 'default' in native[1]:
                                api_schema[cls]['input'].setdefault(section, {})[key] = native
                                if cls == 'SaveVideo' and key == 'codec' and native[0] == 'COMFY_DYNAMICCOMBO_V3':
                                    # Actual Core serialization includes its hidden
                                    # legacy codec widget as well as format.codec.
                                    scalar = deepcopy(native)
                                    scalar[0] = [choice['key'] for choice in options['options']]
                                    front_schema[cls]['input'].setdefault(section, {})[key] = scalar
            paths = {}
            for kind in ('api', 'frontend'):
                source = args.downloads / f'T8_FastV2_UI_R2_{i:02d}_{kind}.json'
                target = exports / source.name
                shutil.copyfile(source, target)
                assert sha(target) == sha(source)
                paths[kind] = target
            api = json.loads(paths['api'].read_text(encoding='utf8'))
            workflow = json.loads(paths['frontend'].read_text(encoding='utf8'))
            values = audit_api(graph, api, api_schema)
            canonical, localization = canonical_task_widgets(workflow, front_schema)
            edges = audit_candidate(graph, canonical, front_schema)
            original = json.loads(input_path.read_text(encoding='utf8'))
            assert workflow['extra']['t8_bound_review'] == original['extra']['t8_bound_review']
            result['cases'].append(dict(name=name, input_sha256=sha(input_path),
                exported_files={kind:dict(path=str(p), sha256=sha(p)) for kind,p in paths.items()},
                actual_api_comparison=values, actual_serialization_comparison=edges,
                known_display_label_contract=localization,
                bound_review_metadata_unchanged=True))
        result.update(status='six_observed_native_ui_api_roundtrips_pass',
            quality_not_inferred_from_ui=True, no_inference_submitted=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(root / 'native-roundtrip-audit.json', result)
    print(json.dumps(dict(status=result['status'], cases=len(result['cases']))), flush=True)


if __name__ == '__main__':
    main()
