import pytest

from tools.run_dual_model_pilot import attach_second_base


def graph():
    return {'1': {'inputs': {'unet_name': 'first.safetensors'}},
            '2': {'inputs': {'model': ['1', 0]}}, '3': {'inputs': {'model': ['1', 0]}}}


def test_second_base_only_rewires_second_lora(tmp_path):
    folder = tmp_path / 'models/diffusion_models'
    folder.mkdir(parents=True)
    source = folder / 'second.safetensors'
    source.write_bytes(b'path validation fixture, never loaded as a model')
    value = graph()
    assert attach_second_base(value, tmp_path, source.name) == source
    assert value['2']['inputs']['model'] == ['1', 0]
    assert value['3']['inputs']['model'] == ['28', 0]
    assert value['28']['inputs']['unet_name'] == source.name


def test_second_base_default_preserves_graph():
    value = graph()
    assert attach_second_base(value, None, None) is None
    assert value == graph()


@pytest.mark.parametrize('name', ['../escape.safetensors', 'file.txt', 'first.safetensors'])
def test_invalid_or_same_base_refused(tmp_path, name):
    folder = tmp_path / 'models/diffusion_models'
    folder.mkdir(parents=True)
    (folder / 'first.safetensors').write_bytes(b'test-only')
    with pytest.raises(ValueError):
        attach_second_base(graph(), tmp_path, name)
