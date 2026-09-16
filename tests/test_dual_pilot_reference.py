from copy import deepcopy

import pytest

from PIL import Image

from tools.run_dual_model_pilot import (
    apply_canvas_size,
    apply_context_frames,
    apply_low_context_source,
    attach_first_frame,
)


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


def test_exact_reference_aspect_is_recorded_without_resizing(tmp_path):
    source = tmp_path / 'input' / 'portrait.png'
    source.parent.mkdir()
    Image.new('RGB', (1024, 1536), 'white').save(source)
    graph = {'8': {'inputs': {}}}
    receipt = attach_first_frame(graph, tmp_path, 'portrait.png', canvas_size=(512, 768))
    assert (receipt['pixel_width'], receipt['pixel_height']) == (1024, 1536)
    assert graph['8']['inputs']['first_frame'] == ['23', 0]


def test_mismatched_reference_aspect_fails_before_graph_change(tmp_path):
    source = tmp_path / 'input' / 'portrait.png'
    source.parent.mkdir()
    Image.new('RGB', (1024, 1536), 'white').save(source)
    graph = {'8': {'inputs': {}}}
    with pytest.raises(ValueError, match='refusing implicit distortion'):
        attach_first_frame(graph, tmp_path, 'portrait.png', canvas_size=(896, 448))
    assert graph == {'8': {'inputs': {}}}


def test_canvas_size_requires_matched_low_high_ratio():
    graph = {'8': {'inputs': {}}}
    apply_canvas_size(graph, 512, 768, 256, 384)
    assert graph['8']['inputs'] == {
        'width': 512, 'height': 768, 'low_width': 256, 'low_height': 384
    }
    with pytest.raises(ValueError, match='same aspect ratio'):
        apply_canvas_size(graph, 512, 768, 448, 224)


def test_context_length_is_an_explicit_native_choice():
    graph = {'8': {'inputs': {'context_frames': 22, 'unchanged': 'value'}}}
    apply_context_frames(graph, 39)
    assert graph['8']['inputs'] == {'context_frames': 39, 'unchanged': 'value'}
    with pytest.raises(ValueError, match='22 or 39'):
        apply_context_frames(graph, 23)


def test_low_context_source_is_an_explicit_single_variable():
    graph = {'8': {'inputs': {'context_frames': 22, 'unchanged': 'value'}}}
    apply_low_context_source(graph, 'accepted_picture_low_context_v1')
    assert graph['8']['inputs'] == {
        'context_frames': 22,
        'low_context_source': 'accepted_picture_low_context_v1',
        'unchanged': 'value',
    }
    with pytest.raises(ValueError, match='Unknown LOW'):
        apply_low_context_source(graph, 'guess')
