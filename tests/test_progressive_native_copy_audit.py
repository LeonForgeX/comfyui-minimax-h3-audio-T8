from copy import deepcopy

import pytest

from tools.audit_progressive_native_copy import audit_copy


def pair():
    names = ['model', 'positive', 'negative', 'av_latent', 'sampler', 'sigmas']
    before = {'nodes': [
        {'id': 1, 'type': 'Source', 'widgets_values': [], 'inputs': [],
         'outputs': [{'links': list(range(1, 7))}]},
        {'id': 2, 'type': 'MiniMaxH3ProgressiveSamplerEXPT8',
         'widgets_values': ['weights', 123, 'fixed', 1, 7, .5, 't2va', 'fp16', 1024],
         'inputs': [{'name': name, 'link': i + 1} for i, name in enumerate(names)],
         'outputs': [{'links': []}]}],
        'links': [[i + 1, 1, 0, 2, i, 'TYPE'] for i in range(6)]}
    after = deepcopy(before)
    copied = deepcopy(before['nodes'][1])
    copied['id'] = 3
    for i, pin in enumerate(copied['inputs']):
        pin['link'] = i + 7
        after['links'].append([i + 7, 1, 0, 3, i, 'TYPE'])
        after['nodes'][0]['outputs'][0]['links'].append(i + 7)
    after['nodes'].append(copied)
    return before, after


def test_exact_native_connected_copy():
    before, after = pair()
    assert audit_copy(before, after)['preserved_input_edges'] == 6


@pytest.mark.parametrize('case', ['parameter', 'original', 'missing_link', 'wrong_source',
    'wrong_slot', 'output_owner', 'old_output_owner', 'output_metadata',
    'steal_consumer', 'mode', 'new_node', 'missing_node', 'duplicate_edge'])
def test_copy_audit_rejects_execution_drift(case):
    before, after = pair()
    if case == 'parameter':
        after['nodes'][2]['widgets_values'][4] = 6
    elif case == 'original':
        after['nodes'][1]['widgets_values'][1] = 456
    elif case == 'missing_link':
        after['links'].pop()
    elif case == 'wrong_source':
        after['links'][-1][1] = 2
    elif case == 'wrong_slot':
        after['links'][-1][2] = 1
    elif case == 'output_owner':
        after['nodes'][0]['outputs'][0]['links'].pop()
    elif case == 'old_output_owner':
        after['nodes'][0]['outputs'][0]['links'].remove(1)
    elif case == 'output_metadata':
        after['nodes'][0]['outputs'][0]['type'] = 'CHANGED'
    elif case == 'steal_consumer':
        after['nodes'][2]['outputs'][0]['links'] = [1]
    elif case == 'mode':
        after['nodes'][2]['mode'] = 4
    elif case == 'new_node':
        after['nodes'].append({'id': 4})
    elif case == 'missing_node':
        after['nodes'].pop(0)
    elif case == 'duplicate_edge':
        after['links'].append(after['links'][-1])
    with pytest.raises(ValueError):
        audit_copy(before, after)
