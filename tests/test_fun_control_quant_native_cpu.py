"""Real small W4A8 tensor payload through Core loading/forward, not full Fun model."""
import json

import pytest
import torch
from safetensors.torch import save_file

from h3_audio_t8_pkg import h3_fun_control_advanced as fun


@pytest.mark.parametrize('encoding', ['legacy', 'native'])
def test_real_native_w4a8_weight_load_and_forward_cpu(tmp_path, encoding):
    from comfy_kitchen.tensor.w4a8_int8 import quantize_w4a8_int8_weight, w4a8_int8_linear
    from comfy_kitchen.tensor import QuantizedTensor
    generator = torch.Generator(device='cpu').manual_seed(911)
    weight = torch.randn((32, 256), generator=generator)
    packed, relative, channel, codebook, correction = quantize_w4a8_int8_weight(weight,
        group_size=16, convrot_groupsize=256, codebook=False)
    assert correction is None
    state = {'control.weight': packed, 'control.weight_s_rel': relative,
        'control.weight_s_channel': channel, 'control.bias': torch.zeros(32)}
    if codebook is not None:
        state['control.weight_codebook'] = codebook
    config = {'format': 'asym_w4a8_int8', 'group_size': 16, 'convrot': True, 'convrot_groupsize': 256}
    metadata = {}
    if encoding == 'legacy':
        metadata['_quantization_metadata'] = json.dumps({'layers': {'control': config}})
    else:
        state['control.comfy_quant'] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    path = tmp_path / 'small_payload.safetensors'
    save_file(state, str(path), metadata=metadata)
    loaded, meta = fun.comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    normalized, _ = fun._normalize_fun_quantization(loaded, meta)
    quantization = fun.comfy.utils.detect_layer_quantization(normalized, '')
    operations = fun.comfy.ops.mixed_precision_ops(quantization, torch.bfloat16)
    module = torch.nn.Module()
    module.control = operations.Linear(256, 32, bias=True, device='cpu', dtype=torch.bfloat16)
    result = module.load_state_dict(normalized, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert isinstance(module.control.weight, QuantizedTensor)
    assert module.control.quant_format == 'asym_w4a8_int8'
    inputs = torch.randn((4, 256), generator=generator).bfloat16()
    actual = module.control(inputs)
    expected = w4a8_int8_linear(inputs, packed, relative, channel, codebook=codebook,
        bias=torch.zeros(32, dtype=torch.bfloat16), group_size=16, convrot_groupsize=256)
    assert torch.isfinite(actual).all() and actual.shape == (4, 32)
    assert torch.equal(actual, expected)
    assert not torch.cuda.is_initialized()
