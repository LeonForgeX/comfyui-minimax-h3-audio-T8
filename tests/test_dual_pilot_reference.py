from copy import deepcopy

import pytest

from tools.run_dual_model_pilot import attach_first_frame


def test_real_input_bound_without_modifying_existing_graph_or_file(tmp_path):
    folder = tmp_path / 'input' / 'nested'
    folder.mkdir(parents=True)
    source = folder / 'reference.png'
    source.write_bytes(b'identity fixture; decoding is Core responsibility')
    graph = {'8': {'class_type': 'Dual', 'inputs': {'model_pass1': ['1', 0]}}}
    original = deepcopy(graph)
    receipt = attach_first_frame(graph, tmp_path, 'nested/reference.png')
    assert receipt['path'] == str(source.resolve())
    assert len(receipt['sha256']) == 64
    assert graph['23']['inputs'] == {'image': 'nested/reference.png'}
    assert graph['8']['inputs'].pop('first_frame') == ['23', 0]
    graph.pop('23')
    assert graph == original
    assert source.read_bytes() == b'identity fixture; decoding is Core responsibility'


@pytest.mark.parametrize('filename', ['../outside.png', '', '   '])
def test_unsafe_filename_rejected_before_graph_change(tmp_path, filename):
    (tmp_path / 'input').mkdir()
    graph = {'8': {'inputs': {}}}
    with pytest.raises(ValueError):
        attach_first_frame(graph, tmp_path, filename)
    assert graph == {'8': {'inputs': {}}}


def test_absolute_filename_rejected(tmp_path):
    (tmp_path / 'input').mkdir()
    with pytest.raises(ValueError):
        attach_first_frame({'8': {'inputs': {}}}, tmp_path, str(tmp_path / 'input/ref.png'))


@pytest.mark.parametrize('graph', [{'8': {'inputs': {'first_frame': ['9', 0]}}},
                                 {'8': {'inputs': {}}, '23': {'inputs': {}}}])
def test_existing_input_not_overwritten(tmp_path, graph):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input/ref.png').write_bytes(b'x')
    original = deepcopy(graph)
    with pytest.raises(ValueError, match='overwrite'):
        attach_first_frame(graph, tmp_path, 'ref.png')
    assert graph == original


def test_no_reference_keeps_existing_t2va_recipe(tmp_path):
    graph = {'8': {'inputs': {}}}
    assert attach_first_frame(graph, tmp_path, None) is None
    assert graph == {'8': {'inputs': {}}}
