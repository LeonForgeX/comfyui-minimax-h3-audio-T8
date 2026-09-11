from copy import deepcopy

import pytest

from tools.run_dual_backend_benchmark import warm_graph


def graph():
    return {'8': {'inputs': {'total_duration_seconds': 3, 'coarse_steps': 4,
        'refine_steps': 4, 'prompt_relay_mode': 'disabled', 'eav_mode': 'disabled',
        'chain_id': 'sample', 'base_seed': 123}}}


def test_only_cache_namespace_changes():
    value = graph()
    before = deepcopy(value)
    warm = warm_graph(value)
    assert value == before
    assert warm['8']['inputs'].pop('chain_id') == 'sample_process_warm'
    before['8']['inputs'].pop('chain_id')
    assert warm == before


@pytest.mark.parametrize('key,value', [('total_duration_seconds', 24), ('coarse_steps', 20),
                                     ('prompt_relay_mode', 'apply_exp'), ('eav_mode', 'apply_exp')])
def test_reject_unmatched_recipe(key, value):
    data = graph()
    data['8']['inputs'][key] = value
    with pytest.raises(ValueError):
        warm_graph(data)
