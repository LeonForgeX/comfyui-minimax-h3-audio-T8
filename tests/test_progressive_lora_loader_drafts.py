"""Use actual Core LoRA loaders and safetensors metadata before native LOW replay.

Tiny H3 CPU/Euler exercises execution and receipt publication. No trained-model
GPU performance or picture-quality claim is made by these tests.
"""
import json

import comfy.lora
import comfy.samplers
import comfy.sd
import comfy.utils
import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg.progressive_first_pass_draft import low_model_identity
from test_progressive_first_pass_draft import run, same
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_masking import memory_nodes, kj  # noqa: F401


def loaded_model(tmp_path, *, model=None, bypass=True, metadata=None, strength=.3):
    """Same native file/metadata/patch path used by LoraLoaderModelOnly."""
    model = model or tiny_model()
    name, layer = next((name, layer) for name, layer in model.model.named_modules()
                       if isinstance(layer, torch.nn.Linear))
    aliases = comfy.lora.model_lora_keys_unet(model.model, {})
    alias = next(key for key, target in aliases.items() if target == name + '.weight')
    weights = {alias + '.lora_up.weight': torch.full((layer.out_features, 1), .03),
               alias + '.lora_down.weight': torch.full((1, layer.in_features), .04),
               alias + '.alpha': torch.tensor(1.)}
    path = tmp_path / 'synthetic-native-lora.safetensors'
    save_file(weights, str(path), metadata=metadata or {'format': 'pt', 'ss_network_module': 'networks.lora'})
    weights, actual_metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    if bypass:
        model = comfy.sd.load_bypass_lora_for_models(model, None, weights, .7, 0)[0]
    model = comfy.sd.load_lora_for_models(model, None, weights, strength, 0,
                                        lora_metadata=actual_metadata)[0]
    assert model.patches and model.get_attachment('lora_metadata') == actual_metadata
    return model


@pytest.mark.parametrize('bypass', [False, True])
def test_actual_loader_metadata_is_portable_and_bound_to_low(tmp_path, bypass):
    sampler = comfy.samplers.ksampler('euler')
    model = loaded_model(tmp_path, bypass=bypass)
    before = low_model_identity(model, sampler)
    assert before['weights'].get('portable_cache_reuse') is not False
    same_model = loaded_model(tmp_path, bypass=bypass)
    assert low_model_identity(same_model, sampler) == before
    changed = loaded_model(tmp_path, bypass=bypass, metadata={'format': 'pt', 'training': 'changed'})
    assert low_model_identity(changed, sampler) != before
    assert model.get_attachment('lora_metadata') == {'format': 'pt', 'ss_network_module': 'networks.lora'}


@pytest.mark.parametrize('shared', [False, True])
def test_actual_lora_loaders_save_then_retry_high_with_zero_low_forwards(tmp_path, stub_lifter, shared):  # noqa: F811
    def models(strength):
        low = loaded_model(tmp_path)
        high = loaded_model(tmp_path, model=low.clone() if shared else None, strength=strength)
        return dict(model=low, high=high)
    root = tmp_path / 'drafts'
    first, saved = run(root, **models(.2))
    replay, report = run(root, mode='reuse_only', high_seed=97, **models(.9))
    baseline, _ = run(high_seed=97, **models(.9))
    same(replay, baseline)
    assert saved['first_pass_draft']['saved_low'] is True
    assert report['first_pass_draft']['reused_low'] is True
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    assert report['noise']['high_seed'] == 97
    assert first['samples'].is_nested
    receipt, = root.glob('low-boundary-*.json')
    assert json.loads(receipt.read_text())['contract']['scope']['material_pack_index'] == 2


def test_actual_lora_loader_with_kj_memory_saves_and_replays(tmp_path, stub_lifter, memory_nodes):  # noqa: F811
    _, sage, _ = memory_nodes
    def model():
        return loaded_model(tmp_path, model=sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(tiny_model()).result[0])
    root = tmp_path / 'drafts'
    _, saved = run(root, model=model())
    replay, report = run(root, mode='reuse_only', model=model())
    baseline, _ = run(model=model())
    same(replay, baseline)
    assert saved['first_pass_draft']['saved_low']
    assert report['counts']['actual_forwards']['low'] == 0


@pytest.mark.parametrize('bypass', [False, True])
def test_metadata_acceptance_does_not_authenticate_other_attachments(tmp_path, bypass):
    model = loaded_model(tmp_path, bypass=bypass)
    model.set_attachments('unreviewed_owner', {'active': object()})
    identity = low_model_identity(model, comfy.samplers.ksampler('euler'))
    assert identity['weights']['portable_cache_reuse'] is False
    assert identity['weights']['invalid_reason_code'] == 'model_attachments_unverified'


@pytest.mark.parametrize('value', [{'callback': lambda: None}, {'format': 3}, ['pt']])
def test_core_metadata_requires_inert_string_map(tmp_path, value):
    model = loaded_model(tmp_path)
    model.set_attachments('lora_metadata', value)
    with pytest.raises(ValueError, match='Core LoRA metadata'):
        low_model_identity(model, comfy.samplers.ksampler('euler'))


@pytest.mark.parametrize('bypass', [False, True])
@pytest.mark.parametrize('owner,reason', [('attention', 'model_attention_unverified'),
                                        ('wrapper', 'model_wrappers_unverified')])
def test_actual_refusal_after_lora_is_not_masked_by_native_injection(tmp_path, bypass, owner, reason):
    model = loaded_model(tmp_path, bypass=bypass)
    def unknown(*args, **kwargs):
        pytest.fail('identity inspection must not execute the unknown owner')
    if owner == 'attention':
        model.model_options['transformer_options']['optimized_attention_override'] = unknown
    else:
        model.add_wrapper_with_key('outer_sample', 'unknown_preview_owner', unknown)
    identity = low_model_identity(model, comfy.samplers.ksampler('euler'))
    assert identity['weights']['portable_cache_reuse'] is False
    assert identity['weights']['invalid_reason_code'] == reason
