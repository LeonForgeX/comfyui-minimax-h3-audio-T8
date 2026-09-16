"""Real Core nonzero LoRA -> INT8 requantization -> read-only identity, CPU only.

No simulated hot assignment, quantizer/backend replacement, checkpoint download
or inference-quality qualification. A missing native CPU kernel is a real
capability failure, not silently replaced with a fake quantized result.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

from h3_audio_t8_pkg import long_video_dual_identity as identity
from test_fast_h3_v2_quantized_identity import KEY, _assert_state_bytes, _model


def _branch(strength, *, extras=True):
    assert strength in (0.25, 0.75) and strength != 0
    model, layer = _model(extras=extras)
    # The serializer-only helper uses a synthetic group8 storage descriptor.
    # Actual Kitchen Hadamard requires a power-of-four group. Construct genuine
    # initial CPU quantization; no simulated hot assignment or backend bypass.
    initial = (torch.arange(72 * 24).reshape(72, 24) % 251 - 125).float() / 32
    initial = initial.to(torch.bfloat16)
    layer.weight = torch.nn.Parameter(QuantizedTensor.from_float(
        initial, "TensorWiseINT8Layout", is_weight=True, per_channel=True,
        convrot=True, convrot_groupsize=4, stochastic_rounding=0,
    ), requires_grad=False)
    model.patches = {}  # Remove the earlier simulated fixture's descriptor.
    adapter = LoRAAdapter({"up", "down"}, (
        torch.ones(72, 1, dtype=torch.float32),
        torch.ones(1, 24, dtype=torch.float32),
        1.0, None, None, None,
    ))
    added = model.add_patches({KEY: adapter}, strength_patch=strength)
    assert added == [KEY]
    assert model.patches[KEY][0][0] == strength
    assert model.patches[KEY][0][1] is adapter
    assert model.patches[KEY][0][2:] == (1.0, None, None)
    return model, layer


def _state_bytes(model):
    return {key: value.detach().clone() for key, value in model.model_state_dict().items()}


def _storage(qtensor):
    return {
        "raw": qtensor._qdata.detach().clone(),
        "scale": qtensor.params.scale.detach().clone(),
        "params": {key: deepcopy(value) for key, value in vars(qtensor.params).items()
                   if key != "scale"},
    }


def _assert_storage(qtensor, before):
    assert torch.equal(qtensor._qdata, before["raw"])
    assert torch.equal(qtensor.params.scale, before["scale"])
    assert {key: value for key, value in vars(qtensor.params).items()
            if key != "scale"} == before["params"]


def _real_patch(model, layer, strength, monkeypatch):
    """Observe the actual requantizer, never alter its input/result or backend."""
    original = layer.weight
    original_storage = _storage(original)
    # Match Core's genuine temporary float32 conversion rather than rounding
    # through original bf16 first. This invokes the installed CPU dequantizer.
    base_float = layer.convert_weight(original.to(dtype=torch.float32, copy=True))
    expected_float = base_float + strength
    calls = []
    requantize = QuantizedTensor.requantize_from_float

    def observed(weight, tensor, **kwargs):
        calls.append({"owner": weight, "tensor": tensor.detach().clone(),
                      "kwargs": dict(kwargs)})
        return requantize(weight, tensor, **kwargs)

    with monkeypatch.context() as observed_patch:
        observed_patch.setattr(QuantizedTensor, "requantize_from_float", observed)
        model.patch_weight_to_device(KEY, device_to=torch.device("cpu"))
    assert len(calls) == 1
    assert calls[0]["owner"] is original
    assert calls[0]["tensor"].device.type == "cpu"
    assert calls[0]["tensor"].dtype == torch.float32
    assert torch.allclose(calls[0]["tensor"], expected_float, atol=1e-6, rtol=0)
    assert calls[0]["kwargs"]["scale"] == "recalculate"
    assert calls[0]["kwargs"]["inplace_ops"] is True
    assert type(calls[0]["kwargs"]["stochastic_rounding"]) is int
    assert calls[0]["kwargs"]["stochastic_rounding"] > 0
    assert type(layer.weight) is QuantizedTensor
    assert layer.weight is not original
    assert layer.weight._qdata.dtype == torch.int8
    assert layer.weight._qdata.device.type == "cpu"
    assert layer.weight.layout_cls is TensorWiseINT8Layout
    assert layer.weight.params.convrot is True
    assert layer.weight.params.convrot_groupsize == 4
    assert not torch.equal(layer.weight._qdata, original_storage["raw"])
    assert set(model.backup) == {KEY}
    assert type(model.backup[KEY].weight) is QuantizedTensor
    assert model.backup[KEY].inplace_update is False
    _assert_storage(model.backup[KEY].weight, original_storage)
    _assert_storage(original, original_storage)
    return original


def _assert_read_only_hot_identity(model, layer, cold_identity, cold_state, monkeypatch):
    hot_state = _state_bytes(model)
    live = layer.weight
    raw, scale, params = live._qdata, live.params.scale, live.params
    live_storage = _storage(live)
    backup_objects = dict(model.backup)
    backup_buffers = dict(model.backup_buffers)
    buffers = {key: value for key, value in model.model.named_buffers()}
    buffer_bytes = {key: value.detach().clone() for key, value in buffers.items()}

    def forbidden(*args, **kwargs):
        pytest.fail("Identity may not dequantize or requantize the real hot model")

    with monkeypatch.context() as read_only:
        read_only.setattr(QuantizedTensor, "dequantize", forbidden)
        read_only.setattr(TensorWiseINT8Layout, "dequantize", forbidden)
        read_only.setattr(QuantizedTensor, "requantize_from_float", forbidden)
        original_state = identity._original_state(model, model.model_state_dict())
        _assert_state_bytes(original_state, cold_state)
        assert identity.stage_model_identity(model) == cold_identity
        assert identity.stage_model_identity(model) == cold_identity
    assert layer.weight is live
    assert live._qdata is raw and live.params is params and live.params.scale is scale
    _assert_storage(live, live_storage)
    _assert_state_bytes(_state_bytes(model), hot_state)
    assert set(model.backup) == set(backup_objects)
    assert all(model.backup[key] is value for key, value in backup_objects.items())
    assert set(model.backup_buffers) == set(backup_buffers)
    assert all(model.backup_buffers[key] is value for key, value in backup_buffers.items())
    assert dict(model.model.named_buffers()).keys() == buffers.keys()
    for key, value in model.model.named_buffers():
        assert value is buffers[key]
        assert torch.equal(value.view(torch.uint8), buffer_bytes[key].view(torch.uint8))


@pytest.mark.parametrize("strength", [0.25, 0.75])
@pytest.mark.parametrize("extras", [False, True])
def test_actual_nonzero_core_lora_requant_cold_hot_and_unpatch_identity(
        strength, extras, monkeypatch):
    model, layer = _branch(strength, extras=extras)
    cold_state = _state_bytes(model)
    cold_identity = identity.stage_model_identity(model)
    original = _real_patch(model, layer, strength, monkeypatch)
    try:
        _assert_read_only_hot_identity(model, layer, cold_identity, cold_state, monkeypatch)
    finally:
        model.unpatch_model()
    assert model.backup == {}
    assert model.backup_buffers == {}
    assert type(layer.weight) is QuantizedTensor
    _assert_storage(layer.weight, _storage(original))
    _assert_state_bytes(_state_bytes(model), cold_state)
    # Descriptor stays installed: restored bytes + the same nonzero LoRA
    # configuration represent the same future execution, not an unpatched task.
    assert identity.stage_model_identity(model) == cold_identity
    assert not torch.cuda.is_initialized()


def test_different_nonzero_strengths_have_distinct_cold_hot_and_restored_identities(monkeypatch):
    observations = []
    for strength in (0.25, 0.75):
        model, layer = _branch(strength)
        cold_state = _state_bytes(model)
        cold_identity = identity.stage_model_identity(model)
        _real_patch(model, layer, strength, monkeypatch)
        try:
            _assert_read_only_hot_identity(model, layer, cold_identity, cold_state, monkeypatch)
            hot = _storage(layer.weight)
            hot_identity = identity.stage_model_identity(model)
        finally:
            model.unpatch_model()
        restored_identity = identity.stage_model_identity(model)
        observations.append({"cold_state": cold_state, "cold": cold_identity,
                             "hot": hot, "hot_identity": hot_identity,
                             "restored": restored_identity})
    left, right = observations
    _assert_state_bytes(left["cold_state"], right["cold_state"])
    assert left["cold"]["sha256"] != right["cold"]["sha256"]
    assert left["hot_identity"]["sha256"] != right["hot_identity"]["sha256"]
    assert left["restored"]["sha256"] != right["restored"]["sha256"]
    assert not torch.equal(left["hot"]["raw"], right["hot"]["raw"])
    assert not torch.equal(left["hot"]["scale"], right["hot"]["scale"])
    assert not torch.cuda.is_initialized()
