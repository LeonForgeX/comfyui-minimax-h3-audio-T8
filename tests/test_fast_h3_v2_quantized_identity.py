"""Original INT8 storage identities through actual Core serialization, CPU only."""
from collections import namedtuple
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from comfy import ops
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

from h3_audio_t8_pkg import long_video_dual_identity as identity
from test_relay_kj_memory import small_model


KEY = "diffusion_model.blocks.0.attn.qkv_proj.weight"
PREFIX = KEY[:-len("weight")]
Backup = namedtuple("Backup", "weight inplace_update")


def _tensor(raw=None, *, scale=None, convrot=True, layout="TensorWiseINT8Layout"):
    raw = (torch.arange(72 * 24).reshape(72, 24) % 251 - 125).to(torch.int8) if raw is None else raw
    scale = torch.full((72, 1), 0.03125) if scale is None else scale
    params = TensorWiseINT8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16, orig_shape=(72, 24),
        is_weight=True, convrot=convrot, convrot_groupsize=8)
    return QuantizedTensor(raw, layout, params)


def _model(*, full_precision=False, extras=True):
    model = small_model()
    layer = ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(
        24, 72, bias=True, device="cpu")
    layer.quant_format = "int8_tensorwise"
    layer.layout_type = "TensorWiseINT8Layout"
    layer._full_precision_mm_config = full_precision
    layer.weight = torch.nn.Parameter(_tensor(), requires_grad=False)
    layer.bias = torch.nn.Parameter(torch.arange(72).to(torch.bfloat16) / 100, requires_grad=False)
    if extras:
        layer.register_buffer("input_scale", torch.tensor([0.125], dtype=torch.float32))
        layer.register_buffer("pre_quant_scale", torch.full((24,), 0.25, dtype=torch.float32))
    model.model.diffusion_model.blocks[0].attn.qkv_proj = layer
    model.model.register_buffer("identity_probe", torch.tensor([1.125, 2.25], dtype=torch.float32))
    model.patches = {KEY: [(0.5, LoRAAdapter({"up", "down"}, (
        torch.ones(72, 1), torch.ones(1, 24), 1.0, None, None, None)), 1.0, None, None)]}
    return model, layer


def _hot(model, layer, *, changed_convrot=True):
    original = layer.weight
    model.backup[KEY] = Backup(original, False)
    model.backup[PREFIX + "bias"] = Backup(layer.bias, False)
    model.backup_buffers = {"identity_probe": model.model.identity_probe}
    for name in ("input_scale", "pre_quant_scale"):
        if getattr(layer, name, None) is not None:
            model.backup_buffers[PREFIX + name] = getattr(layer, name)
            setattr(layer, name, getattr(layer, name).to(torch.float16) + 7)
    model.model.identity_probe = model.model.identity_probe.to(torch.float16) + 9
    layer.weight = torch.nn.Parameter(_tensor(
        -original._qdata, scale=original.params.scale * 3,
        convrot=changed_convrot), requires_grad=False)
    layer.bias = torch.nn.Parameter(layer.bias + 5, requires_grad=False)
    return original


def _assert_state_bytes(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].shape == expected[key].shape, key
        assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8)), key


@pytest.mark.parametrize("full_precision", [False, True])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("changed_convrot", [False, True])
def test_actual_core_serializer_restores_cold_int8_identity_without_dequantizing(
        monkeypatch, full_precision, extras, changed_convrot):
    model, layer = _model(full_precision=full_precision, extras=extras)
    cold = model.model_state_dict()
    cold_identity = identity.stage_model_identity(model)
    original = _hot(model, layer, changed_convrot=changed_convrot)
    current = model.model_state_dict()
    current_weight, current_input = layer.weight, getattr(layer, "input_scale", None)
    original_raw, original_scale = original._qdata, original.params.scale
    backups, buffers = dict(model.backup), dict(model.backup_buffers)
    serializer, calls = ops._quantized_weight_state_dict, []

    def forbidden(*args, **kwargs):
        pytest.fail("Identity must not dequantize Qtensor storage")

    def observed(proxy, state, prefix, **kwargs):
        calls.append((proxy, prefix, kwargs))
        return serializer(proxy, state, prefix, **kwargs)

    monkeypatch.setattr(QuantizedTensor, "dequantize", forbidden)
    monkeypatch.setattr(TensorWiseINT8Layout, "dequantize", forbidden)
    monkeypatch.setattr(ops, "_quantized_weight_state_dict", observed)
    restored = identity._original_state(model, current)
    _assert_state_bytes(restored, cold)
    assert identity.stage_model_identity(model)["sha256"] == cold_identity["sha256"]
    assert restored is not current
    assert current[KEY] is current_weight._qdata
    assert current[KEY + "_scale"] is current_weight.params.scale
    assert layer.weight is current_weight
    assert getattr(layer, "input_scale", None) is current_input
    assert original._qdata is original_raw and original.params.scale is original_scale
    assert all(model.backup[key] is value for key, value in backups.items())
    assert all(model.backup_buffers[key] is value for key, value in buffers.items())
    restored_calls = [call for call in calls if call[0] is not layer]
    assert restored_calls and all(call[0].weight is original for call in restored_calls)
    assert all(call[1] == PREFIX for call in restored_calls)
    assert all(call[2] == {"extra_quant_params": ("input_scale", "pre_quant_scale")}
               for call in restored_calls)
    assert not torch.cuda.is_initialized()


def test_live_requantization_scale_does_not_change_original_identity():
    model, layer = _model()
    before = identity.stage_model_identity(model)["sha256"]
    original = _hot(model, layer)
    assert identity.stage_model_identity(model)["sha256"] == before
    layer.weight = torch.nn.Parameter(_tensor(scale=original.params.scale * 11), requires_grad=False)
    assert identity.stage_model_identity(model)["sha256"] == before
    original.params.scale[0, 0] *= 2
    assert identity.stage_model_identity(model)["sha256"] != before


@pytest.mark.parametrize("mutation", ["layout", "subclass", "params", "not_weight", "meta", "scale_shape"])
def test_unknown_or_invalid_quantized_backups_fail_closed_before_dequantizing(monkeypatch, mutation):
    model, layer = _model()
    original = _hot(model, layer)
    bad_key, bad = KEY, original
    if mutation == "layout":
        bad = _tensor(layout="TensorCoreFP8Layout")
    elif mutation == "subclass":
        class ForeignQuantizedTensor(QuantizedTensor):
            pass
        bad = ForeignQuantizedTensor(original._qdata, original._layout_cls, original.params)
    elif mutation == "params":
        bad = QuantizedTensor(original._qdata, original._layout_cls, replace(original.params, is_weight=False))
    elif mutation == "not_weight":
        bad_key = PREFIX + "bias"
    elif mutation == "meta":
        bad = _tensor(torch.empty((72, 24), dtype=torch.int8, device="meta"))
    else:
        bad = _tensor(scale=torch.ones(3, 3))
    model.backup[bad_key] = Backup(bad, False)
    monkeypatch.setattr(QuantizedTensor, "dequantize", lambda *a: pytest.fail("must fail before dequantize"))
    with pytest.raises(ValueError, match="Quantized backup|QuantizedTensor"):
        identity.stage_model_identity(model)
    assert layer.weight is not original


@pytest.mark.parametrize("mutation", ["missing_scale", "missing_extra", "unknown_buffer", "quantized_buffer", "nonfinite_scale"])
def test_incomplete_or_bad_original_storage_cannot_get_a_resume_identity(mutation):
    model, layer = _model()
    original = _hot(model, layer)
    current = model.model_state_dict()
    if mutation == "missing_scale":
        current.pop(KEY + "_scale")
    elif mutation == "missing_extra":
        current.pop(PREFIX + "input_scale")
        model.backup_buffers.pop(PREFIX + "input_scale")
    elif mutation == "unknown_buffer":
        model.backup_buffers["unknown_runtime_buffer"] = torch.ones(1)
    elif mutation == "quantized_buffer":
        model.backup_buffers["identity_probe"] = original
    else:
        original.params.scale[0, 0] = float("nan")
    with pytest.raises(ValueError, match="backup|nonfinite"):
        identity.content_identity(identity._original_state(model, current))


def test_raw_qtensor_cannot_fall_through_to_materializing_content_identity(monkeypatch):
    monkeypatch.setattr(QuantizedTensor, "dequantize", lambda *a: pytest.fail("must not dequantize"))
    with pytest.raises(ValueError, match="raw-storage adapter"):
        identity.content_identity({"unexpected": _tensor()})


def test_no_backup_and_ordinary_tensor_backup_paths_keep_existing_state_identity():
    model = small_model()
    state = model.model_state_dict()
    assert identity._original_state(model, state) is state
    before = identity.stage_model_identity(model)["sha256"]
    weight = state[KEY]
    model.backup[KEY] = Backup(weight.clone(), False)
    with torch.no_grad():
        weight.add_(1)
    assert identity.stage_model_identity(model)["sha256"] == before


def test_plain_original_buffer_backup_is_restored_without_mutating_the_live_model():
    model = small_model()
    model.model.register_buffer("identity_probe", torch.tensor([1.0], dtype=torch.float32))
    before = identity.stage_model_identity(model)["sha256"]
    original = model.model.identity_probe
    model.backup_buffers = {"identity_probe": original}
    model.model.identity_probe = original.to(torch.float16) + 3
    live = model.model.identity_probe
    assert identity.stage_model_identity(model)["sha256"] == before
    assert model.model.identity_probe is live
    assert model.backup_buffers["identity_probe"] is original


def test_plain_legacy_identity_import_does_not_require_core_quantized_tensor_class(monkeypatch):
    name = 'h3_audio_t8_pkg.legacy_quant_identity_probe'
    spec = importlib.util.spec_from_file_location(name, Path(identity.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'comfy.quant_ops', ModuleType('comfy.quant_ops'))
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    model = small_model()
    state = model.model_state_dict()
    assert module._original_state(model, state) is state
    assert module.stage_model_identity(model)['sha256'] == identity.stage_model_identity(model)['sha256']
    assert not torch.cuda.is_initialized()
