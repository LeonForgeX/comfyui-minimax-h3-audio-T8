from copy import deepcopy

import pytest

from tools.run_dual_model_pilot import attach_memory_chunks
from tools.audit_dual_model_pilot import validate_memory_backend


@pytest.mark.parametrize('heads,ffn', [(1, 1), (4, 1), (1, 2), (4, 2)])
def test_both_model_branches_receive_declared_patches_only(heads, ffn):
    graph = {'8': {'inputs': {'model_pass1': ['21', 0], 'model_pass2': ['22', 0]}}}
    attach_memory_chunks(graph, 'kj-memory', heads, ffn)
    for branch, source, h, f in [('model_pass1', '21', '24', '26'), ('model_pass2', '22', '25', '27')]:
        if heads > 1:
            assert graph[h]['inputs'] == {'model': [source, 0], 'head_chunks': 4}
            source = h
        if ffn > 1:
            assert graph[f]['inputs'] == {'model': [source, 0], 'chunks': 2, 'seq_threshold': 4096}
            source = f
        assert graph['8']['inputs'][branch] == [source, 0]
    assert len(graph) == 1 + 2 * (heads > 1) + 2 * (ffn > 1)


@pytest.mark.parametrize('backend,heads,ffn', [('sol', 4, 2), ('kj', 4, 1), ('kj-memory', 3, 2)])
def test_invalid_composition_fails_without_changing_graph(backend, heads, ffn):
    graph = {'8': {'inputs': {}}}
    original = deepcopy(graph)
    with pytest.raises(ValueError):
        attach_memory_chunks(graph, backend, heads, ffn)
    assert graph == original


def test_existing_node_not_replaced():
    graph = {'8': {'inputs': {}}, '27': {'inputs': {'existing': True}}}
    original = deepcopy(graph)
    with pytest.raises(ValueError, match='overwrite'):
        attach_memory_chunks(graph, 'kj-memory', 4, 2)
    assert graph == original


@pytest.mark.parametrize('case', [None, 'heads', 'ffn', 'count', 'source', 'grouping'])
def test_actual_composition_audit_rejects_missing_or_different_patches(case):
    backend = {'memory_composition': {'kind': 'kj_memory_sage', 'head_chunks': 4,
               'ffn_settings': [2, 4096], 'source_sha256s': ['fixture']},
               'completed_calls': {'sage:unbiased': 800},
               'head_grouping': 'inside_delegate; full-head Relay/EAV once per block'}
    terminal = {'memory_head_chunks': 4, 'memory_ffn_chunks': 2}
    if case == 'heads':
        backend['memory_composition']['head_chunks'] = 1
    elif case == 'ffn':
        backend['memory_composition']['ffn_settings'] = None
    elif case == 'count':
        backend['completed_calls']['sage:unbiased'] = 200
    elif case == 'source':
        backend['memory_composition']['source_sha256s'] = []
    elif case == 'grouping':
        backend['head_grouping'] = 'outside'
    if case:
        with pytest.raises(ValueError):
            validate_memory_backend(backend, terminal)
    else:
        validate_memory_backend(backend, terminal)
