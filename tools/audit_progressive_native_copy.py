"""Independently check saved copies made with the native Paste with Connect command."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_native_progressive_exports import equal, identity, read  # noqa: E402


def audit_copy(before, after):
    old = {n['id']: n for n in before['nodes']}
    new = {n['id']: n for n in after['nodes']}
    added = new.keys() - old.keys()
    if not old.keys() <= new.keys() or len(added) != 1:
        raise ValueError('Expected exactly one new node and no removed nodes')
    copied = new[next(iter(added))]
    originals = [n for n in old.values() if n['type'] == 'MiniMaxH3ProgressiveSamplerEXPT8']
    if len(originals) != 1 or copied['type'] != originals[0]['type']:
        raise ValueError('Expected a copied progressive sampler')
    source = originals[0]
    for node_id, node in old.items():
        actual = new[node_id]
        if any(not equal(node.get(key, default), actual.get(key, default))
               for key, default in [('type', ''), ('mode', 0), ('widgets_values', []), ('inputs', [])]):
            raise ValueError(f'Original execution node changed: {node_id}')
    if not equal(source['widgets_values'], copied['widgets_values']) or copied.get('mode', 0) != 0:
        raise ValueError('Copied widget values or mode changed')
    old_edges = {e[0]: e for e in before['links']}
    edges = {e[0]: e for e in after['links']}
    if len(edges) != len(after['links']) or any(edges.get(key) != value for key, value in old_edges.items()):
        raise ValueError('Original edges changed or duplicate edge id')
    for node_id, original in old.items():
        actual = new[node_id]
        outputs = original.get('outputs', [])
        if len(outputs) != len(actual.get('outputs', [])):
            raise ValueError('Original output count changed')
        for slot, (old_pin, new_pin) in enumerate(zip(outputs, actual.get('outputs', []))):
            if not equal({k: v for k, v in old_pin.items() if k != 'links'},
                         {k: v for k, v in new_pin.items() if k != 'links'}):
                raise ValueError('Original output metadata changed')
            expected = {key for key, edge in edges.items() if edge[1:3] == [node_id, slot]}
            owned = new_pin.get('links') or []
            if len(owned) != len(set(owned)) or set(owned) != expected:
                raise ValueError('Output ownership differs from complete edge table')
    checked = []
    pins = {p['name']: (i, p) for i, p in enumerate(copied['inputs'])}
    for pin in source['inputs']:
        if pin.get('link') is None:
            continue
        old_edge = old_edges[pin['link']]
        index, new_pin = pins[pin['name']]
        edge = edges.get(new_pin.get('link'))
        expected = [old_edge[1], old_edge[2], copied['id'], index, old_edge[5]]
        if edge is None or edge[1:] != expected or edge[0] in old_edges:
            raise ValueError('Copied input did not preserve its source')
        if edge[0] not in new[edge[1]]['outputs'][edge[2]].get('links', []):
            raise ValueError('Copied edge not owned by its source output')
        checked.append(edge[0])
    if len(checked) != 6 or set(edges) - set(old_edges) != set(checked):
        raise ValueError('Expected exactly six copied input edges')
    if any(p.get('links') for p in copied['outputs']):
        raise ValueError('Paste must not steal the original output consumers')
    return {'source_node': source['id'], 'copied_node': copied['id'], 'preserved_input_edges': 6,
            'original_nodes_and_edges_unchanged': True, 'copied_widgets_equal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Do not overwrite evidence')
    result = {'status': 'incomplete', 'cases': {}, 'generation_queued': False,
              'interaction': 'Native Paste with Connect; temporary Alt+Shift+P binding restored to default',
              'limitation': 'Browser automation Ctrl+Shift+V virtual clipboard failed; not a physical-keyboard test'}
    args.output.mkdir(parents=True)
    for task in ('T2VA', 'I2VA'):
        before = args.baseline / f'{task}.saved.json'
        after = args.root / 'user/default/workflows' / f'2026-09-09_H3_Progressive_{task}_6plus2_EXP.json'
        data = read(after)
        result['cases'][task] = {**audit_copy(read(before), data), 'inputs': [identity(before), identity(after)]}
        (args.output / f'{task}.copied.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    result['status'] = 'two_native_connected_node_copies_pass'
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
