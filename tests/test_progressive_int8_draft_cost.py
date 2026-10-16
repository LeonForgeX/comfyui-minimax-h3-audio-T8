"""Native INT8 draft cost and corruption guards; CPU mechanics, not GPU timing."""
import json
from copy import copy
from dataclasses import replace

import comfy.ops
import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

from h3_audio_t8_pkg import progressive_checkpoint as checkpoint
from h3_audio_t8_pkg import progressive_first_pass_draft as draft
from h3_audio_t8_pkg import progressive_int8_readers as readers
from test_fast_h3_v2_quantized_identity import _model
from test_progressive_draft_nonportable_cost import count_identities
from test_progressive_first_pass_draft import run, same, draft_arguments, SCOPE
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_kj_sol_drafts import stack, third_party  # noqa: F401
from test_progressive_masking import memory_nodes, kj  # noqa: F401


def int8_model():
    model = tiny_model()
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
    for name, original in list(model.model.named_modules()):
        if '.blocks.' not in name or not isinstance(original, torch.nn.Linear):
            continue
        parent_name, attr = name.rsplit('.', 1)
        layer = ops.Linear(original.in_features, original.out_features,
                           bias=original.bias is not None, device='cpu')
        layer.quant_format, layer.layout_type = 'int8_tensorwise', 'TensorWiseINT8Layout'
        layer._full_precision_mm_config = False
        layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            original.weight.detach(), 'TensorWiseINT8Layout', is_weight=True,
            per_channel=True, convrot=False, stochastic_rounding=0), requires_grad=False)
        if original.bias is not None:
            layer.bias = torch.nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        setattr(model.model.get_submodule(parent_name), attr, layer)
    assert any(type(p) is QuantizedTensor for p in model.model.parameters())
    return model


@pytest.mark.parametrize('full_precision', [False, True])
@pytest.mark.parametrize('extras', [False, True])
def test_actual_int8_serializer_is_read_only(full_precision, extras):
    model, _ = _model(full_precision=full_precision, extras=extras)
    assert draft._read_only_identity_model(model)


@pytest.mark.parametrize('change', ['helper', 'property', 'layout', 'defaults', 'tensor_attribute',
                                  'scale_subclass', 'state_hook', 'serializer'])
def test_modified_int8_readers_keep_independent_captures(monkeypatch, change):
    model, layer = _model()
    assert draft._read_only_identity_model(model)
    if change == 'helper':
        monkeypatch.setattr(comfy.ops, '_quantized_weight_state_dict', lambda *a, **kw: None)
    elif change == 'property':
        monkeypatch.setattr(QuantizedTensor, 'params', property(lambda self: self._params))
    elif change == 'layout':
        import comfy_kitchen.tensor.base as base
        monkeypatch.setitem(base.LAYOUTS, 'TensorWiseINT8Layout', object())
    elif change == 'defaults':
        monkeypatch.setattr(type(layer).state_dict, '__defaults__', ({}, 'changed', True))
    elif change == 'tensor_attribute':
        layer.weight.extra_reader = object()
    elif change == 'scale_subclass':
        class Foreign(torch.Tensor):
            pass
        layer.weight._params = replace(layer.weight._params,
                                      scale=layer.weight._params.scale.as_subclass(Foreign))
    elif change == 'state_hook':
        layer.register_state_dict_post_hook(lambda *a: None)
    else:
        monkeypatch.setattr(type(layer), 'state_dict', lambda *a, **kw: {})
    assert not draft._read_only_identity_model(model)


def test_reader_source_parsed_once_per_inspection_not_once_per_layer(monkeypatch):
    model, layer = _model()
    model.model.add_module('repeated', torch.nn.ModuleList([copy(layer) for _ in range(64)]))
    calls, original = [], readers._ast_hash
    def observed(node):
        calls.append(node.name)
        return original(node)
    monkeypatch.setattr(readers, '_ast_hash', observed)
    assert draft._read_only_identity_model(model)
    assert len(calls) == len(readers._READERS)
    calls.clear()
    assert draft._read_only_identity_model(model)
    assert len(calls) == len(readers._READERS)  # no proof reused across boundaries


@pytest.mark.parametrize('mode', ['reuse_only', 'save_or_reuse'])
def test_selected_int8_replay_uses_four_captures_and_zero_low(
        tmp_path, stub_lifter, monkeypatch, mode):  # noqa: F811
    captures = count_identities(monkeypatch)
    _, saved = run(tmp_path, model=int8_model(), high=int8_model())
    assert len(captures) == 6
    captures.clear()
    replay, report = run(tmp_path, model=int8_model(), high=int8_model(), mode=mode,
                         draft_id=saved['first_pass_draft']['cache_id'], high_seed=97)
    assert len(captures) == 4
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    baseline, _ = run(model=int8_model(), high=int8_model(), high_seed=97)
    same(replay, baseline)


@pytest.mark.parametrize('change', ['raw', 'scale', 'lora', 'conditioning', 'tensor_sha',
                                  'tensor_size', 'receipt_digest', 'report_digest'])
def test_selected_int8_replay_rejects_changed_inputs_before_lift(
        tmp_path, stub_lifter, monkeypatch, change):  # noqa: F811
    _, saved = run(tmp_path, model=int8_model(), high=int8_model())
    low, high = int8_model(), int8_model()
    weight = next(p for p in low.model.parameters() if type(p) is QuantizedTensor)
    values = {}
    if change in ('raw', 'scale', 'lora', 'conditioning'):
        original = checkpoint.load_file
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            if change == 'raw':
                weight._qdata.data.flatten()[0] += 1
            elif change == 'scale':
                weight._params.scale.data.mul_(2)
            elif change == 'lora':
                name, tensor = next(iter(low.model.named_parameters()))
                low.add_patches({name: ('diff', (torch.full_like(tensor, .03),))}, .8)
            else:
                values['positive'][0][0].data.add_(.1)
            return result
        monkeypatch.setattr(checkpoint, 'load_file', mutate)
        if change == 'conditioning':
            from test_progressive_sampling_runtime import conditioning
            values['positive'] = conditioning()
    else:
        path, = tmp_path.glob('low-boundary-*.json')
        record = json.loads(path.read_text())
        if change == 'receipt_digest':
            record['contract']['scope']['material_pack_index'] += 1
        else:
            key = {'tensor_sha': 'tensor_sha256', 'tensor_size': 'tensor_size_bytes',
                   'report_digest': 'report_sha256'}[change]
            record[key] = 1 if change == 'tensor_size' else 'f' * 64
        path.write_text(json.dumps(record))
    starts = len(stub_lifter)
    with pytest.raises(ValueError, match='incompatible|integrity|corrupt|evidence'):
        run(tmp_path, model=low, high=high, mode='reuse_only',
            draft_id=saved['first_pass_draft']['cache_id'], **values)
    assert len(stub_lifter) == starts


def test_selected_preload_requires_exclusive_session(tmp_path):
    session = draft.ProgressiveFirstPassDraftSession(tmp_path, SCOPE, 'reuse_only', 'a' * 64)
    model = int8_model()
    with pytest.raises(RuntimeError, match='locked'):
        session._bind_and_load_low(**draft_arguments(model, model))


def test_original_unaudited_int8_fallback_cost_remains_for_unknown_readers(
        tmp_path, stub_lifter, monkeypatch):  # noqa: F811
    # Reproduce the previous INT8 gate result and retain its conservative path.
    monkeypatch.setattr(draft, '_read_only_identity_model', lambda model: False)
    captures = count_identities(monkeypatch)
    _, saved = run(tmp_path, model=int8_model(), high=int8_model())
    assert len(captures) == 12
    captures.clear()
    _, report = run(tmp_path, model=int8_model(), high=int8_model(), mode='reuse_only',
                    draft_id=saved['first_pass_draft']['cache_id'])
    assert len(captures) == 10
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}


@pytest.mark.parametrize('third_party', [True], indirect=True)
def test_real_int8_kj_sol_lora_preview_chain_keeps_same_pack_retry(
        tmp_path, memory_nodes, third_party, stub_lifter, monkeypatch):  # noqa: F811
    import test_progressive_kj_sol_drafts as cases
    import test_progressive_lora_loader_drafts as loaders
    monkeypatch.setattr(cases, 'tiny_model', int8_model)
    monkeypatch.setattr(loaders, 'tiny_model', int8_model)
    def models():
        return dict(model=stack(tmp_path, memory_nodes, third_party),
                    high=stack(tmp_path, memory_nodes, third_party, high=True))
    captures = count_identities(monkeypatch)
    root = tmp_path / 'drafts'
    _, saved = run(root, **models())
    assert len(captures) == 6
    captures.clear()
    replay, report = run(root, mode='reuse_only', high_seed=97,
                         draft_id=saved['first_pass_draft']['cache_id'], **models())
    assert len(captures) == 4
    baseline, _ = run(high_seed=97, **models())
    same(replay, baseline)
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 2}
    record = json.loads(next(root.glob('low-boundary-*.json')).read_text())
    assert record['contract']['scope']['material_pack_index'] == 2
    assert not record['contract']['scope']['parent_candidate_id']
    assert record['contract']['model']['weights']['backend']['kernels']['runtime']
