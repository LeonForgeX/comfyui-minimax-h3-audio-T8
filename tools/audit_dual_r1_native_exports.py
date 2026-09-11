"""Independently compare observed native UI edits with saves and API downloads."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_native_progressive_exports import identity, read  # noqa: E402
from audit_progressive_workflows import audit_candidate, equal  # noqa: E402


def audit_edits(original, saved, api, info, edits):
    saved = deepcopy(saved)
    presentation = []
    for node in saved['nodes']:
        node.setdefault('widgets_values', [])
        if node['type'] == 'MiniMaxH3AudioConditioningT8':
            labels = {'T2VA — 文生音视频': 'T2VA', 'I2VA — 图生音视频（首帧）': 'I2VA'}
            value = node['widgets_values'][4]
            if value in labels:
                node['widgets_values'][4] = labels[value]
                presentation.append('localized_task_label')
        if node['type'] == 'LoadImage' and len(node['widgets_values']) == 2:
            if node['widgets_values'][1] != 'image':
                raise ValueError('Unknown LoadImage button state')
            node['widgets_values'].pop()
            presentation.append('native_LoadImage_upload_button')
    expected = {n['id']: n for n in original['nodes'] if n['type'] != 'MarkdownNote'}
    actual = {n['id']: n for n in saved['nodes'] if n['type'] != 'MarkdownNote'}
    if expected.keys() != actual.keys():
        raise ValueError('Execution node identities changed')
    applied = []
    for node_id, node in expected.items():
        other = actual[node_id]
        if node['type'] != other['type'] or other.get('mode', 0) != node.get('mode', 0):
            raise ValueError('Execution type/mode changed')
        values = list(node.get('widgets_values', []))
        for edit in edits:
            if edit['node'] == node_id:
                index = edit['widget']
                if node['type'] != edit['type'] or not equal(values[index], edit['before']):
                    raise ValueError('Original edit precondition changed')
                values[index] = edit['after']
                applied.append(edit)
        if not equal(values, other['widgets_values']):
            raise ValueError(f'Unexpected widget difference: {node_id}')
    if len(applied) != len(edits):
        raise ValueError('Unapplied intended edit')

    def edges(wf):
        nodes = {n['id']: n for n in wf['nodes']}
        return sorted((e[1], e[2], e[3], nodes[e[3]]['inputs'][e[4]]['name'], e[5])
                      for e in wf['links'])
    if edges(original) != edges(saved):
        raise ValueError('Execution links changed')
    prompt = dict(sorted(api.items(), key=lambda item: int(item[0])))
    contract = audit_candidate(prompt, saved, info)
    return {'contract': contract, 'exact_edits': applied, 'presentation': presentation,
            'all_other_execution_widgets_and_links_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite evidence')
    with urlopen(args.url + '/object_info', timeout=20) as response:
        info = json.load(response)
    with urlopen(args.url + '/queue', timeout=20) as response:
        queue = json.load(response)
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise ValueError('Unexpected live queue')
    ready = read(args.root / 'ready.json')
    results = {'status': 'incomplete', 'cases': {}, 'generation_queued': False,
               'provenance': 'Observed native UI actions; original saves/downloads are read only'}
    args.output.mkdir(parents=True)
    for entry in ready['copies']:
        source, saved_path = Path(entry['source']), Path(entry['copy'])
        original, saved = read(source), read(saved_path)
        if 'Progressive' in source.name:
            task = 'I2VA' if '_I2VA_' in source.name else 'T2VA'
            download = args.downloads / f'T8_R1_{task}_clear_v7.json'
            prompt = next(n for n in original['nodes'] if n['type'] == 'PrimitiveStringMultiline')
            if not prompt['widgets_values'][0]:
                raise ValueError('Original prompt was already empty')
            edits = [{'node': prompt['id'], 'type': prompt['type'], 'widget': 0,
                      'before': prompt['widgets_values'][0], 'after': ''}]
        else:
            kind = 'Relay' if '_Relay_' in source.name else 'Plain'
            task = 'Dual' + kind
            download = args.downloads / f'T8_Dual_{kind}_native_v7.json'
            edits = [{'node': 3, 'type': 'MiniMaxH3LoRACompatibilityLoaderT8Advanced',
                      'widget': 1, 'before': 1., 'after': .85 if kind == 'Relay' else .9}]
            if kind == 'Relay':
                edits.append({'node': 7, 'type': 'MiniMaxH3PromptRelayPlanT8Advanced', 'widget': 4,
                              'before': '0-15\n15-40\n40-75\n75-100',
                              'after': '0-20\n20-40\n40-75\n75-100'})
        api = read(download)
        result = audit_edits(original, saved, api, info, edits)
        results['cases'][task] = {**result, 'files': list(map(identity, [source, saved_path, download]))}
        for label, data in [('saved', saved), ('api', api)]:
            (args.output / f'{task}.{label}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    results['status'] = 'four_observed_native_UI_save_reload_API_contracts_pass'
    (args.output / 'object-info.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
