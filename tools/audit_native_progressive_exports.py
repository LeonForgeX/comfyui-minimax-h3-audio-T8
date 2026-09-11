"""Audit actual UI saves/downloads; never queue inference or mutate those inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_progressive_workflows import audit_candidate, equal  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_change(original, saved, api, info, task):
    saved = deepcopy(saved)
    presentation = []
    for node in saved['nodes']:
        node.setdefault('widgets_values', [])
        if node['type'] == 'MiniMaxH3AudioConditioningT8':
            labels = {'T2VA': 'T2VA — 文生音视频', 'I2VA': 'I2VA — 图生音视频（首帧）'}
            # Exact display labels from web/task_type_labels.js. API must still
            # contain the canonical code; never normalize the exported API.
            if node['widgets_values'][4] == labels[task]:
                node['widgets_values'][4] = task
                presentation.append('localized_task_label')
        if node['type'] == 'LoadImage' and len(node['widgets_values']) == 2:
            if node['widgets_values'][1] != 'image':
                raise ValueError('Unexpected LoadImage upload button state')
            node['widgets_values'].pop()
            presentation.append('native_LoadImage_upload_button')
    expected = {n['id']: n for n in original['nodes'] if n['type'] != 'MarkdownNote'}
    actual = {n['id']: n for n in saved['nodes'] if n['type'] != 'MarkdownNote'}
    if expected.keys() != actual.keys():
        raise ValueError('Saved execution node identities changed')
    modified = []
    for node_id, node in expected.items():
        other = actual[node_id]
        if node['type'] != other['type']:
            raise ValueError('Saved node type changed')
        values = list(node['widgets_values'])
        if node['type'] == 'MiniMaxH3ProgressiveSamplerEXPT8':
            if values[4] != 6 or values[6] != task.lower():
                raise ValueError('Original fixture no longer is six-step task')
            values[4] = 7
            modified.append(node_id)
        if not equal(values, other['widgets_values']):
            raise ValueError(f'Unexpected saved widget difference in {node_id}')
    if len(modified) != 1:
        raise ValueError('Expected exactly one deliberately modified sampler')
    def canonical_edges(wf):
        by_id = {n['id']: n for n in wf['nodes']}
        return sorted((e[1], e[2], e[3], by_id[e[3]]['inputs'][e[4]]['name'], e[5])
                      for e in wf['links'])
    if canonical_edges(original) != canonical_edges(saved):
        raise ValueError('Original execution links changed')
    prompt = dict(sorted(api.items(), key=lambda item: int(item[0])))
    contract = audit_candidate(prompt, saved, info)
    return {'contract': contract, 'changed_sampler': modified[0],
            'only_parameter_change': 'low_evaluations:6->7',
            'all_other_execution_widgets_and_original_links_equal': True,
            'native_presentation_only_differences': presentation}


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
    project = Path(__file__).resolve().parents[1]
    result = {'status': 'incomplete', 'cases': {}, 'generation_queued': False,
              'copy_with_links': 'not_yet_qualified',
              'provenance': 'UI actions observed in task; downloaded APIs and UI saves independently reread'}
    args.output.mkdir(parents=True)
    for task in ('T2VA', 'I2VA'):
        name = f'2026-09-09_H3_Progressive_{task}_6plus2_EXP.json'
        paths = [project / 'examples/workflows/28-progressive-sampling' / name,
                 args.root / 'user/default/workflows' / name,
                 args.downloads / f'T8_R1_{task}_roundtrip_v5.json']
        original, saved, api = map(read, paths)
        result['cases'][task] = {**audit_change(original, saved, api, info, task),
                                 'inputs': list(map(identity, paths))}
        for label, data in [('saved', saved), ('api', api)]:
            (args.output / f'{task}.{label}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    result['status'] = 'two_actual_UI_save_reload_API_contracts_pass_copy_pending'
    (args.output / 'object-info.json').write_text(json.dumps(info, ensure_ascii=False), encoding='utf-8')
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
