from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.trt_vae_build import INPUT_SHAPE, OUTPUT_SHAPE
from h3_audio_t8_pkg.trt_vae_engine import enqueue_checked, resolve_io


@pytest.mark.parametrize("bad", [None, "wrong_name", "wrong_shape", "old_decoder", "bad_kind"])
def test_encoder_runtime_io_separate_and_resolved_before_allocation(bad):
    engine, context, trt = fake_io()
    engine.get_tensor_name = lambda i: ("pixel_tile", "moments_tile")[i]
    engine.get_tensor_mode = lambda name: "in" if name == "pixel_tile" else "out"
    context.get_tensor_shape = lambda _: (1, 48, 5, 16, 16)
    if bad == "wrong_name":
        engine.get_tensor_name = lambda i: ("pixel_tile", "wrong")[i]
    if bad == "wrong_shape":
        context.get_tensor_shape = lambda _: (1, 48, 1, 16, 16)
    if bad == "old_decoder":
        engine, context, trt = fake_io()
    if bad:
        with pytest.raises((ValueError, RuntimeError)):
            resolve_io(engine, context, trt, (1, 3, 17, 256, 256), kind="invalid" if bad == "bad_kind" else "encoder")
    else:
        assert resolve_io(engine, context, trt, (1, 3, 17, 256, 256), kind="encoder") == (1, 48, 5, 16, 16)


def test_encoder_enqueue_uses_pixel_input_and_moments_output():
    calls = []
    context = SimpleNamespace(set_tensor_address=lambda name, pointer: calls.append(name) or True,
                              execute_async_v3=lambda **kwargs: True)
    enqueue_checked(context, 1, 2, 3, kind="encoder")
    assert calls == ["pixel_tile", "moments_tile"]


def test_t1_context_must_resolve_real_output_not_trust_model_annotation():
    engine, context, trt = fake_io()
    engine.get_tensor_name = lambda i: ("pixel_tile", "moments_tile")[i]
    engine.get_tensor_mode = lambda name: "in" if name == "pixel_tile" else "out"
    context.get_tensor_shape = lambda _: (1, 48, 1, 16, 16)
    shape = (1, 3, 1, 256, 256)
    assert resolve_io(engine, context, trt, shape, kind="encoder", profile="t1") == (1, 48, 1, 16, 16)
    with pytest.raises(ValueError):
        resolve_io(engine, context, trt, shape, kind="encoder")
    context.get_tensor_shape = lambda _: (1, 48, 5, 16, 16)
    with pytest.raises(RuntimeError, match="output shape"):
        resolve_io(engine, context, trt, shape, kind="encoder", profile="t1")
    with pytest.raises(ValueError):
        resolve_io(engine, context, trt, shape, profile="t1")


def fake_io():
    trt = SimpleNamespace(float16="half", TensorIOMode=SimpleNamespace(INPUT="in", OUTPUT="out"),
                          TensorLocation=SimpleNamespace(DEVICE="gpu"))
    engine = SimpleNamespace(num_io_tensors=2, get_tensor_name=lambda i: ("latent_tile", "pixel_tile")[i],
        get_tensor_mode=lambda name: "in" if name == "latent_tile" else "out",
        get_tensor_dtype=lambda _: "half", get_tensor_location=lambda _: "gpu")
    context = SimpleNamespace(set_input_shape=lambda name, shape: True,
                              get_tensor_shape=lambda _: OUTPUT_SHAPE)
    return engine, context, trt


def test_runtime_shape_resolved_before_allocation():
    assert resolve_io(*fake_io(), INPUT_SHAPE) == OUTPUT_SHAPE


@pytest.mark.parametrize("failure", ["count", "name", "mode", "dtype", "host", "set_shape", "output", "input"])
def test_invalid_runtime_io_rejected(failure):
    engine, context, trt = fake_io()
    shape = INPUT_SHAPE
    if failure == "count":
        engine.num_io_tensors = 3
    elif failure == "name":
        engine.get_tensor_name = lambda _: "wrong"
    elif failure == "mode":
        engine.get_tensor_mode = lambda _: "out"
    elif failure == "dtype":
        engine.get_tensor_dtype = lambda _: "float32"
    elif failure == "host":
        engine.get_tensor_location = lambda _: "host"
    elif failure == "set_shape":
        context.set_input_shape = lambda *args: False
    elif failure == "output":
        context.get_tensor_shape = lambda _: (1, 3, -1, 256, 256)
    else:
        shape = (1, 24, 1, 16, 16)
    with pytest.raises((ValueError, RuntimeError)):
        resolve_io(engine, context, trt, shape)


@pytest.mark.parametrize("failure", [None, "latent_tile", "pixel_tile", "execute"])
def test_enqueue_never_ignores_failed_binding_or_execution(failure):
    calls = []

    def bind(name, pointer):
        calls.append(name)
        return name != failure

    def execute(**kwargs):
        calls.append("execute")
        return failure != "execute"

    context = SimpleNamespace(set_tensor_address=bind, execute_async_v3=execute)
    if failure:
        with pytest.raises(RuntimeError):
            enqueue_checked(context, 1, 2, 3)
        assert calls[-1] == failure
    else:
        enqueue_checked(context, 1, 2, 3)
        assert calls == ["latent_tile", "pixel_tile", "execute"]


@pytest.mark.parametrize("pointer", [0, -1, True, None, 1.5])
def test_bad_address_rejected_before_enqueue(pointer):
    with pytest.raises(ValueError):
        enqueue_checked(None, pointer, 2, 3)


def test_native_reference_materializes_nonpersistent_rope_without_weights_or_gpu():
    import torch
    from comfy.ldm.minimax.vae import RotaryEmbeddingND
    from tools.trt_vae_tile_probe_worker import native_decoder_shell

    decoder, conv = native_decoder_shell()
    assert all(parameter.is_meta for parameter in decoder.parameters())
    assert all(parameter.is_meta for parameter in conv.parameters())
    assert not decoder.pos_embed.inv_freq.is_meta
    torch.testing.assert_close(decoder.pos_embed.inv_freq, RotaryEmbeddingND(48, 100.0, 3).inv_freq)
