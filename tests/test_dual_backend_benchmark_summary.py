from copy import deepcopy

import pytest

from tools.summarize_dual_backend_benchmark import normalized_graph


def graph(backend):
    choices = {
        'pytorch': ('ModelAttentionBackend', {'attention': 'pytorch attention'}),
        'kj': ('PathchSageAttentionKJ', {'sage_attention': 'auto', 'allow_compile': False}),
        'sol': ('SolAttentionPatch', {'enabled': True, 'tau': 1.3, 'min_tokens': 4096,
            'strict': True, 'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}),
    }
    cls, options = choices[backend]
    result = {'8': {'inputs': {'chain_id': backend, 'total_duration_seconds': 3,
        'coarse_steps': 4, 'refine_steps': 4, 'base_seed': 123}},
        '2': {'inputs': {'strength_model': 1.0}},
        '3': {'inputs': {'strength_model': 0.9}}}
    for key, source in [('21', '2'), ('22', '3')]:
        result[key] = {'class_type': cls, 'inputs': {'model': [source, 0], **options}}
    return result


def test_only_known_backend_and_namespace_are_normalized():
    values = [graph(backend) for backend in ('pytorch', 'kj', 'sol')]
    before = deepcopy(values)
    actual = [normalized_graph(value, backend) for value, backend in
              zip(values, ('pytorch', 'kj', 'sol'))]
    assert actual[0] == actual[1] == actual[2]
    assert values == before


@pytest.mark.parametrize('change', ['seed', 'strength', 'steps'])
def test_non_backend_changes_remain_visible(change):
    baseline, altered = graph('pytorch'), graph('kj')
    if change == 'strength':
        altered['3']['inputs']['strength_model'] = 1.0
    else:
        altered['8']['inputs']['base_seed' if change == 'seed' else 'coarse_steps'] = 8
    assert normalized_graph(baseline, 'pytorch') != normalized_graph(altered, 'kj')


@pytest.mark.parametrize('change', ['branch', 'compile', 'extra', 'duration'])
def test_unknown_backend_or_recipe_rejected(change):
    value = graph('kj')
    if change == 'duration':
        value['8']['inputs']['total_duration_seconds'] = 24
    elif change == 'branch':
        value['22']['inputs']['model'] = ['2', 0]
    elif change == 'compile':
        value['21']['inputs']['allow_compile'] = True
    else:
        value['21']['inputs']['unknown_option'] = True
    with pytest.raises(ValueError):
        normalized_graph(value, 'kj')
