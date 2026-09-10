import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import trt_vae_build as build
from tools import run_trt_vae_build as control


def request(tmp_path):
    root = tmp_path.resolve()
    stage = root / ".building-test"
    stage.mkdir()
    return {"schema": "t8-trt-build-v1", "precision": "fp16", "runtime_version": build.TRT_VERSION,
            "input_shape": list(build.INPUT_SHAPE), "model_sha256": build.MODEL_SHA,
            "weights_sha256": build.WEIGHTS_SHA, "model_path": str(root / "decoder.onnx"),
            "weights_path": str(root / "decoder.onnx.data"), "runtime_site": str(root),
            "staging_path": str(stage), "gpu_uuid": "GPU-test", "device_index": 0,
            "driver_version": "test-only", "workspace_bytes": 4 * 1024**3,
            "runtime_sources": {str(root / "runtime.dll"): "a" * 64},
            "worker_sources": {str(root / "worker.py"): "b" * 64}}


def synthetic_bundle(value):
    stage = Path(value["staging_path"])
    (stage / "model.engine").write_bytes(b"CPU fake engine")
    manifest = {"request_sha256": build.validate_request(value), "status": "compiled_not_execution_qualified",
                "engine_sha256": build.digest_file(stage / "model.engine"),
                "input_shape": list(build.INPUT_SHAPE), "output_shape": list(build.OUTPUT_SHAPE),
                "gpu_uuid": value["gpu_uuid"], "runtime_version": build.TRT_VERSION}
    build.write_new_json(stage / "manifest.json", manifest)
    return stage


def encoder_request(tmp_path):
    value = request(tmp_path)
    value.update(schema="t8-trt-encoder-build-v1", kind="encoder",
                 input_shape=list(build.ENCODER_INPUT_SHAPE),
                 model_sha256=build.ENCODER_MODEL_SHA,
                 model_path=str(tmp_path.resolve() / "encoder.onnx"),
                 weights_path=None, weights_sha256=None)
    return value


def test_t1_encoder_uses_its_static_graph_without_reinterpreting_t17(tmp_path):
    value = encoder_request(tmp_path)
    before = build.validate_request(value)
    assert build.parsed_input_shape(value) == (-1, 3, 17, 256, 256)
    assert build.validate_request(value) == before
    value.update(schema="t8-trt-encoder-t1-build-v1", model_sha256=build.ENCODER_T1_MODEL_SHA,
                 input_shape=list(build.ENCODER_T1_INPUT_SHAPE))
    assert build.validate_request(value) != before
    assert build.parsed_input_shape(value) == (1, 3, 1, 256, 256)
    assert build.graph_spec(value)[3] == (1, 48, 1, 16, 16)
    for key, bad in (("model_sha256", build.ENCODER_MODEL_SHA),
                     ("input_shape", list(build.ENCODER_INPUT_SHAPE)),
                     ("encoder_norm_precision", "fp32_norm_affine")):
        with pytest.raises(ValueError):
            build.validate_request(dict(value, **{key: bad}))


def test_encoder_has_separate_identity_shape_and_atomic_bundle(tmp_path):
    value = encoder_request(tmp_path)
    assert build.graph_spec(value) == ("pixel_tile", "moments_tile", (1, 3, 17, 256, 256), (1, 48, 5, 16, 16))
    stage = Path(value["staging_path"])
    (stage / "model.engine").write_bytes(b"CPU synthetic encoder")
    manifest = {"request_sha256": build.validate_request(value), "status": "compiled_not_execution_qualified",
                "engine_sha256": build.digest_file(stage / "model.engine"),
                "input_shape": list(build.ENCODER_INPUT_SHAPE), "output_shape": list(build.ENCODER_OUTPUT_SHAPE),
                "input_name": "pixel_tile", "output_name": "moments_tile",
                "gpu_uuid": value["gpu_uuid"], "runtime_version": build.TRT_VERSION}
    build.write_new_json(stage / "manifest.json", manifest)
    build.publish_bundle(stage, tmp_path / "encoder-final", value)
    assert (tmp_path / "encoder-final/model.engine").is_file()


@pytest.mark.parametrize("key,bad", [("kind", "decoder"), ("schema", "t8-trt-build-v1"),
    ("input_shape", [1, 3, 1, 256, 256]), ("input_shape", [True, 3, 17, 256, 256]),
    ("model_sha256", build.MODEL_SHA), ("weights_sha256", build.WEIGHTS_SHA),
    ("weights_path", "embedded.onnx.data")])
def test_encoder_never_admits_decoder_assets_single_images_or_external_weights(tmp_path, key, bad):
    value = encoder_request(tmp_path)
    value[key] = bad
    with pytest.raises(ValueError):
        build.validate_request(value)


def test_old_decoder_manifest_remains_valid(tmp_path):
    value = request(tmp_path)
    before = build.canonical_hash(value)
    assert build.validate_request(value) == before
    assert build.graph_spec(value) == ("latent_tile", "pixel_tile", build.INPUT_SHAPE, build.OUTPUT_SHAPE)


@pytest.mark.parametrize("missing", [False, True])
def test_encoder_norm_policy_requires_all_25_norm_and_affine_triplets(missing):
    layers = []
    class Layer:
        num_outputs = 1
        def __init__(self, name):
            self.name, self.precision, self.output_type = name, "half", "half"
        def get_output(self, index):
            return SimpleNamespace(is_shape_tensor=False)
        def set_output_type(self, index, dtype):
            self.output_type = dtype
        def get_output_type(self, index):
            return self.output_type
    for index in range(25):
        for operation in ("InstanceNormalization", "Mul_1", "Add"):
            layers.append(Layer(f"/encoder/block.0/norm1_{index}/{operation}"))
    if missing:
        layers.pop()
    network = SimpleNamespace(num_layers=len(layers), get_layer=lambda i: layers[i])
    flags = []
    config = SimpleNamespace(set_flag=flags.append)
    trt = SimpleNamespace(float32="float", float16="half", BuilderFlag=SimpleNamespace(OBEY_PRECISION_CONSTRAINTS="obey"))
    if missing:
        with pytest.raises(ValueError, match="coverage"):
            build.encoder_norm_precision(network, config, trt, "fp32_norm_affine")
    else:
        rows = build.encoder_norm_precision(network, config, trt, "fp32_norm_affine")
        assert len(rows) == 75 and flags == ["obey"]
        assert all(layer.precision == "float" for layer in layers)
        assert layers[0].output_type == layers[1].output_type == "float"
        assert layers[2].output_type == "half"


@pytest.mark.parametrize("mode", ["none_return", "bad_readback", "raises", "add_failed"])
def test_python_profile_void_setter_is_verified_by_readback(mode):
    class Profile:
        def set_shape(self, name, low, optimal, high):
            if mode == "raises":
                raise ValueError("invalid dimensions")
            self.value = [low, optimal, high]
            return None

        def get_shape(self, name):
            return [] if mode == "bad_readback" else self.value

    profile = Profile()
    config = SimpleNamespace(add_optimization_profile=lambda _: -1 if mode == "add_failed" else 0)
    if mode == "none_return":
        build.set_fixed_profile(profile, config)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            build.set_fixed_profile(profile, config)


@pytest.mark.parametrize("key,bad", [("workspace_bytes", True), ("workspace_bytes", 0),
    ("workspace_bytes", 5 * 1024**3), ("device_index", 1), ("device_index", False),
    ("runtime_version", "latest"), ("input_shape", [1, 24, 1, 16, 16]),
    ("model_sha256", "bad"), ("runtime_sources", {}), ("gpu_uuid", ""),
    ("precision", "int8"), ("driver_version", None), ("model_path", "relative.onnx")])
def test_bad_request_before_gpu(tmp_path, key, bad):
    value = request(tmp_path)
    value[key] = bad
    with pytest.raises(ValueError):
        build.validate_request(value)


def test_atomic_complete_bundle(tmp_path):
    value = request(tmp_path)
    stage = synthetic_bundle(value)
    destination = tmp_path / "final"
    report = build.publish_bundle(stage, destination, value)
    assert report["status"] == "compiled_not_execution_qualified"
    assert not stage.exists() and (destination / "model.engine").is_file()


@pytest.mark.parametrize("failure", ["partial", "corrupt", "manifest", "extra", "exists"])
def test_incomplete_or_changed_bundle_never_publishes(tmp_path, failure):
    value = request(tmp_path)
    stage = synthetic_bundle(value)
    destination = tmp_path / "final"
    if failure == "partial":
        (stage / "model.engine").rename(stage / "model.engine.partial")
    elif failure == "corrupt":
        (stage / "model.engine").write_bytes(b"corrupt")
    elif failure == "manifest":
        (stage / "manifest.json").write_text("{}")
    elif failure == "extra":
        (stage / "other").touch()
    else:
        destination.mkdir()
        (destination / "user-file").write_text("preserve")
    with pytest.raises(ValueError):
        build.publish_bundle(stage, destination, value)
    assert stage.is_dir()
    assert not (destination / "model.engine").exists()
    if failure == "exists":
        assert (destination / "user-file").read_text() == "preserve"


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object")
@pytest.mark.parametrize("mode", ["complete", "error", "hang", "cancel"])
@pytest.mark.parametrize("public", [False, True])
def test_controller_real_cpu_process_tree(tmp_path, monkeypatch, mode, public):
    if public:
        from h3_audio_t8_pkg import trt_vae_compile as selected_control
    else:
        selected_control = control
    value = request(tmp_path)
    value["test_mode"] = "hang" if mode == "cancel" else mode
    build.write_new_json(tmp_path / "request.json", value)
    # Synthetic assets: do not label this a real-weights/compiler qualification.
    monkeypatch.setattr(selected_control, "verify_sources", lambda _: None)
    cancel = threading.Event()
    timer = threading.Timer(.8, cancel.set) if mode == "cancel" else None
    if timer:
        timer.start()
    try:
        arguments = dict(check=lambda: None, cancel=cancel, timeout=1.5 if mode == "hang" else 10,
                         worker=Path(__file__).parent / "fixtures/trt_build_worker.py")
        if mode == "complete":
            selected_control.execute_owned(value, tmp_path, tmp_path / "final", **arguments)
        else:
            with pytest.raises(selected_control.IsolatedTaskError):
                selected_control.execute_owned(value, tmp_path, tmp_path / "final", **arguments)
    finally:
        if timer:
            timer.cancel()
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["isolated"]["active_after_cleanup"] == 0
    assert terminal["isolated"]["job_assigned_before_task"]
    if mode != "complete":
        assert not (tmp_path / "final").exists()
        assert Path(value["staging_path"]).is_dir()
