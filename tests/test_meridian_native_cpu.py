"""Native Core loading and real small ConvRot operators, never whole-model GPU."""

import pytest
import torch

import test_meridian_convert_job as job_tests
from h3_audio_t8_pkg.meridian_convert_job import run_conversion
from h3_audio_t8_pkg.meridian_conversion import TensorRule, encode_native_tensor


@pytest.fixture
def conversion(tmp_path):
    return job_tests.conversion.__wrapped__(tmp_path)


def test_complete_tiny_checkpoint_native_core_strict_load(conversion):
    import comfy.model_detection
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy_kitchen.tensor import QuantizedTensor

    run_conversion(**conversion)
    state, metadata = comfy.utils.load_torch_file(
        str(conversion["destination"]), return_metadata=True
    )
    detected = comfy.model_detection.detect_unet_config(state, "", metadata=metadata)
    assert detected["image_model"] == "minimax_h3" and detected["num_layers"] == 1
    ops = comfy.ops.mixed_precision_ops(
        comfy.utils.detect_layer_quantization(state, ""), torch.bfloat16
    )
    native = MiniMaxH3Model(
        **detected, device="cpu", dtype=torch.bfloat16, operations=ops
    )
    outcome = native.load_state_dict(state, strict=True)
    assert not outcome.missing_keys and not outcome.unexpected_keys
    quant = [
        m
        for m in native.modules()
        if isinstance(getattr(m, "weight", None), QuantizedTensor)
    ]
    assert len(quant) == 5
    assert all(m.weight._params.convrot for m in quant)
    assert native.video_patch_proj.quant_format is None
    assert native.token_refiner.blocks[0].attn.qkv_proj.quant_format is None
    assert not torch.cuda.is_initialized()


def test_complete_tiny_checkpoint_standard_diffusion_loader(conversion):
    import comfy.sd
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy_kitchen.tensor import QuantizedTensor

    run_conversion(**conversion)
    patcher = comfy.sd.load_diffusion_model(
        str(conversion["destination"]), model_options={"dtype": torch.bfloat16}
    )
    assert patcher is not None
    native = patcher.model.diffusion_model
    assert isinstance(native, MiniMaxH3Model)
    assert isinstance(native.blocks[0].mlp.fc1.weight, QuantizedTensor)
    assert native.blocks[0].mlp.fc1.weight._params.convrot_groupsize == 16
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("columns,group", [(64, 64), (256, 256), (2688, 64)])
def test_actual_native_convrot_cpu_linear_dispatch(columns, group):
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor

    rule = TensorRule(
        "layer.weight", ("unused.weight",), ((32, columns),), quantize=True
    )
    gen = torch.Generator().manual_seed(120)
    value = torch.randn(32, columns, generator=gen)
    state, _ = encode_native_tensor(rule, value, torch.bfloat16, row_chunk=7)
    operations = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.bfloat16)
    module = torch.nn.Module()
    module.layer = operations.Linear(columns, 32, bias=False, device="cpu")
    outcome = module.load_state_dict(state, strict=True)
    assert not outcome.missing_keys and not outcome.unexpected_keys
    assert isinstance(module.layer.weight, QuantizedTensor)
    assert module.layer.weight._params.convrot_groupsize == group
    inputs = torch.randn(8, columns, generator=gen).bfloat16()
    # A Python dispatch mode is popped inside QuantizedTensor's own dispatch;
    # profile actual dispatcher events instead of inferring a missing kernel.
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile,
    ):
        actual = module.layer(inputs)
    calls = [
        event.key
        for event in profile.key_averages()
        if "comfy_kitchen::int8_linear" in event.key
    ]
    reference = torch.nn.functional.linear(inputs.float(), value)
    error = (actual.float() - reference).norm() / reference.norm()
    assert calls, [event.key for event in profile.key_averages()]
    assert actual.shape == (8, 32) and torch.isfinite(actual).all()
    assert float(error) < 0.035
    assert not torch.cuda.is_initialized()
