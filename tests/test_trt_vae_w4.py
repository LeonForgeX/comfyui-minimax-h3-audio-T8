from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import trt_vae_build as build
from h3_audio_t8_pkg.trt_vae_backend import supports_shapes
from test_trt_vae_build import request


def w4_request(tmp_path):
    value = request(tmp_path)
    value.update(schema="t8-trt-decoder-w4a16-build-v1", kind="decoder", precision="w4a16",
                 model_sha256=build.DECODER_W4_MODEL_SHA, weights_path=None, weights_sha256=None)
    return value


def test_embedded_w4_identity_keeps_fp16_io_and_fixed_t7(tmp_path):
    value = w4_request(tmp_path)
    assert build.validate_request(value) == build.canonical_hash(value)
    assert build.graph_spec(value) == ("latent_tile", "pixel_tile", build.INPUT_SHAPE, build.OUTPUT_SHAPE)
    assert build.parsed_input_shape(value) == (-1, 24, 7, 16, 16)
    assert supports_shapes(value, [build.INPUT_SHAPE])
    assert not supports_shapes(value, [(1, 24, 1, 16, 16)])


@pytest.mark.parametrize("key,bad", [("kind", "encoder"), ("schema", "t8-trt-build-v1"),
    ("precision", "fp16"), ("model_sha256", build.MODEL_SHA),
    ("weights_sha256", build.WEIGHTS_SHA), ("weights_path", "not-embedded.data"),
    ("encoder_norm_precision", "fp32_norm_affine"), ("input_shape", [1,24,1,16,16])])
def test_w4_cannot_be_relabelled_or_silently_substituted(tmp_path, key, bad):
    with pytest.raises(ValueError):
        build.validate_request(dict(w4_request(tmp_path), **{key: bad}))


@pytest.mark.parametrize("quantized", [False, True])
def test_strong_typing_not_fp16_or_int8_builder_selection(tmp_path, quantized):
    value = w4_request(tmp_path) if quantized else request(tmp_path)
    trt = SimpleNamespace(NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
                          BuilderFlag=SimpleNamespace(FP16="fp16", INT8="int8", TF32="tf32"),
                          ProfilingVerbosity=SimpleNamespace(DETAILED="detailed"))
    flags = {"tf32"}
    config = SimpleNamespace(set_flag=flags.add, clear_flag=flags.discard, get_flag=lambda flag: flag in flags)
    report = build.configure_precision(value, config, trt)
    assert build.network_flags(value, trt) == (3 if quantized else 1)
    assert report == {"strongly_typed": quantized, "fp16_enabled": not quantized, "tf32_enabled": False}
    assert flags == (set() if quantized else {"fp16"})
    if quantized:
        assert config.profiling_verbosity == "detailed"


@pytest.mark.parametrize("key,bad", [("strongly_typed", False), ("strongly_typed", 1),
    ("fp16_enabled", True), ("tf32_enabled", True), ("precision", "fp16")])
def test_loader_and_publication_require_explicit_compiler_precision_receipt(tmp_path, key, bad):
    value = w4_request(tmp_path)
    manifest = {"strongly_typed": True, "fp16_enabled": False, "tf32_enabled": False, "precision": "w4a16"}
    build.validate_quantized_manifest(manifest, value)
    with pytest.raises(ValueError):
        build.validate_quantized_manifest(dict(manifest, **{key: bad}), value)
    with pytest.raises(ValueError):
        build.validate_quantized_manifest({}, value)


def test_no_new_fields_required_for_existing_fp16_engine_manifests(tmp_path):
    build.validate_quantized_manifest({}, request(tmp_path))


def test_saved_reference_probe_is_inference_only_resource_policy():
    from tools.run_trt_vae_video_probe import probe_policy
    policy = probe_policy(saved_reference=True)
    assert policy.startup_free_gpu_bytes == 10*1024**3
    assert policy.minimum_free_gpu_bytes == 1024**3
    assert probe_policy().startup_free_gpu_bytes == 12000*1024**2


def test_trt_rgb_cannot_be_used_as_native_reference(tmp_path):
    from tools.trt_vae_saved_reference import bind_reference
    path = tmp_path / "trt-rgb.safetensors"
    path.touch()
    with pytest.raises(ValueError, match="actual native"):
        bind_reference(path, "unused", "unused")
