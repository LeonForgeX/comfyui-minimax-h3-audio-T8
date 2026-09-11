"""Exercise both existing loaders' quantization selection without model allocation."""
import json
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg import h3_fun_control_advanced as fun


class Selected(Exception):
    pass


def fixture_state():
    # Pinned header has [21504,2688] packed QKV: output extent is unchanged,
    # input extent is halved. No tensor payload or full-size allocation here.
    specs = {'control_proj_in.weight': (5376, 196),
        'control_blocks.0.attn.qkv_proj.weight': (21504, 2688),
        'control_blocks.0.attn.q_norm.weight': (128,),
        'control_blocks.0.mlp.fc1.weight': (28672, 2688),
        'control_blocks.0.after_proj.weight': (5376, 5376),
        'control_blocks.0.adaln_proj.linear.weight': (32256, 8)}
    sd = {name: torch.empty(shape, device='meta', dtype=torch.float32)
        for name, shape in specs.items()}
    sd['control_blocks.0.attn.qkv_proj.weight'] = torch.empty((21504, 2688), device='meta', dtype=torch.int8)
    config = {'format': 'asym_w4a8_int8', 'group_size': 16,
        'convrot': True, 'convrot_groupsize': 256}
    return sd, config


@pytest.mark.parametrize('loader', ['official', 'compatibility'])
@pytest.mark.parametrize('encoding', ['legacy', 'native', 'plain'])
def test_existing_loaders_choose_correct_quantization(monkeypatch, loader, encoding):
    sd, config = fixture_state()
    metadata = {'minimax_h3_fun_controlnet': 'adaln_basis'}
    if encoding == 'legacy':
        metadata['_quantization_metadata'] = json.dumps({'layers': {'control_blocks.0.attn.qkv_proj': config}})
    elif encoding == 'native':
        sd['control_blocks.0.attn.qkv_proj.comfy_quant'] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    else:
        sd['control_blocks.0.attn.qkv_proj.weight'] = torch.empty((21504, 5376), device='meta')
    def load(_path, safe_load=True, return_metadata=False):
        assert safe_load
        return (dict(sd), dict(metadata)) if return_metadata else dict(sd)
    monkeypatch.setattr(fun.comfy.utils, 'load_torch_file', load)
    monkeypatch.setattr(fun.comfy.model_management, 'get_torch_device', lambda: torch.device('cpu'))
    def mixed(quantization, compute_dtype):
        assert encoding != 'plain'
        assert quantization == {'mixed_ops': True}
        assert compute_dtype == torch.bfloat16
        raise Selected('mixed')
    def dense(*args, **kwargs):
        assert encoding == 'plain', 'Quantized checkpoint silently selected dense operations'
        raise Selected('dense')
    monkeypatch.setattr(fun.comfy.ops, 'mixed_precision_ops', mixed)
    monkeypatch.setattr(fun.comfy.ops, 'pick_operations', dense)
    monkeypatch.setattr(fun.comfy.model_management, 'unet_dtype', lambda **kw: torch.float32)
    monkeypatch.setattr(fun.comfy.model_management, 'unet_manual_cast', lambda *a, **kw: None)
    monkeypatch.setattr(fun, '_official_model_patch_fun_control_modules', lambda: (
        SimpleNamespace(is_minimax_h3_fun_state_dict=lambda _sd: True), object, object))
    target = fun._load_official_model_patch_control if loader == 'official' else fun._load_compatibility_control
    with pytest.raises(Selected, match='dense' if encoding == 'plain' else 'mixed'):
        target('fixture.safetensors')


@pytest.mark.parametrize('loader', ['official', 'compatibility'])
def test_corrupt_legacy_metadata_never_selects_dense(monkeypatch, loader):
    sd, _ = fixture_state()
    def load(_path, safe_load=True, return_metadata=False):
        return (dict(sd), {'_quantization_metadata': '{broken'}) if return_metadata else dict(sd)
    monkeypatch.setattr(fun.comfy.utils, 'load_torch_file', load)
    monkeypatch.setattr(fun, '_official_model_patch_fun_control_modules', lambda: (
        SimpleNamespace(is_minimax_h3_fun_state_dict=lambda _sd: True), object, object))
    def unexpected(*a, **kw):
        raise AssertionError('Corrupt metadata reached device/operations selection')
    monkeypatch.setattr(fun.comfy.model_management, 'get_torch_device', unexpected)
    target = fun._load_official_model_patch_control if loader == 'official' else fun._load_compatibility_control
    with pytest.raises((ValueError, RuntimeError), match='(?i)quant|metadata|property'):
        target('fixture.safetensors')


def test_normalization_preserves_tensor_identity_and_does_not_mutate_caller():
    sd, config = fixture_state()
    metadata = {'_quantization_metadata': json.dumps({'layers': {'control_blocks.0.attn.qkv_proj': config}})}
    keys = set(sd)
    result, returned = fun._normalize_fun_quantization(sd, metadata)
    assert set(sd) == keys and all(result[k] is v for k, v in sd.items())
    assert returned == metadata
    assert json.loads(bytes(result['control_blocks.0.attn.qkv_proj.comfy_quant'].tolist())) == config


@pytest.mark.parametrize('fault', ['unknown_format', 'missing_weight', 'group_zero', 'bool_group', 'invalid_convrot', 'conflict'])
def test_bad_or_conflicting_metadata_refuses_before_core_conversion(monkeypatch, fault):
    sd, config = fixture_state()
    layer = 'control_blocks.0.attn.qkv_proj'
    if fault == 'unknown_format':
        config['format'] = 'unknown_quant_format'
    elif fault == 'missing_weight':
        del sd[layer + '.weight']
    elif fault == 'group_zero':
        config['group_size'] = 0
    elif fault == 'bool_group':
        config['group_size'] = True
    elif fault == 'invalid_convrot':
        config['convrot'] = 'false'
    elif fault == 'conflict':
        sd[layer + '.comfy_quant'] = torch.tensor(list(json.dumps(dict(config, group_size=32)).encode()), dtype=torch.uint8)
    def unexpected(*a, **kw):
        raise AssertionError('Invalid metadata must not reach conversion')
    monkeypatch.setattr(fun.comfy.utils, 'convert_old_quants', unexpected)
    with pytest.raises((ValueError, RuntimeError), match='(?i)quant'):
        fun._normalize_fun_quantization(sd, {'_quantization_metadata': json.dumps({'layers': {layer: config}})})


def test_old_core_without_converter_reports_required_capability(monkeypatch):
    sd, config = fixture_state()
    monkeypatch.delattr(fun.comfy.utils, 'convert_old_quants')
    with pytest.raises(RuntimeError, match='lacks convert_old_quants'):
        fun._normalize_fun_quantization(sd, {'_quantization_metadata': json.dumps({'layers': {'control_blocks.0.attn.qkv_proj': config}})})
    assert fun._normalize_fun_quantization(sd, None)[0] is sd


def test_converter_that_drops_metadata_is_not_accepted(monkeypatch):
    sd, config = fixture_state()
    monkeypatch.setattr(fun.comfy.utils, 'convert_old_quants', lambda state, metadata: (state, metadata))
    with pytest.raises(RuntimeError, match='did not return valid'):
        fun._normalize_fun_quantization(sd, {'_quantization_metadata': json.dumps({'layers': {'control_blocks.0.attn.qkv_proj': config}})})
