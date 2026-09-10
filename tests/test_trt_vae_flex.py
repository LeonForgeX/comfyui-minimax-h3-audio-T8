from types import SimpleNamespace
import pytest
from h3_audio_t8_pkg import trt_vae_build as build
from h3_audio_t8_pkg.trt_vae_engine import resolve_io
from h3_audio_t8_pkg.trt_vae_backend import supports_shapes
from test_trt_vae_build import request
from test_trt_vae_engine import fake_io
from tools.prepare_trt_vae_flex_decoder import derive


def test_shape_derivation_changes_only_external_annotations():
    import onnx
    from onnx import helper, TensorProto
    model = helper.make_model(helper.make_graph([helper.make_node("Identity", ["latent_tile"], ["pixel_tile"])], "test",
             [helper.make_tensor_value_info("latent_tile", TensorProto.FLOAT16, ["b",24,7,16,16])],
             [helper.make_tensor_value_info("pixel_tile", TensorProto.FLOAT16, ["b",3,28,256,256])]))
    before = model.SerializeToString()
    candidate = derive(model)
    assert model.SerializeToString() == before
    assert candidate.graph.node[0].SerializeToString() == model.graph.node[0].SerializeToString()
    assert candidate.graph.input[0].type.tensor_type.shape.dim[2].dim_param == "tile_t"
    candidate.graph.input[0].CopyFrom(model.graph.input[0])
    candidate.graph.output[0].CopyFrom(model.graph.output[0])
    assert candidate.SerializeToString() == before
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 1
    with pytest.raises(ValueError):
        derive(model)
    assert onnx.TensorProto.FLOAT16 == 10


def test_flex_separate_source_identity_and_external_weights(tmp_path):
    value = request(tmp_path)
    old_hash = build.validate_request(value)
    assert build.validate_request(value) == old_hash
    value.update(schema="t8-trt-decoder-flex-build-v1", kind="decoder", model_sha256=build.DECODER_FLEX_MODEL_SHA,
                 weights_path=str(tmp_path.resolve() / "minimax_h3_vae_decoder.onnx.data"))
    assert build.validate_request(value) != old_hash
    assert build.parsed_input_shape(value) == (1,24,-1,-1,-1)
    assert supports_shapes(value, [(1,24,1,1,16),(1,24,7,8,13)])
    assert not supports_shapes(value, [(1,24,2,16,16)])
    with pytest.raises(ValueError):
        build.validate_request(dict(value, model_sha256=build.MODEL_SHA))


@pytest.mark.parametrize("shape", [(1,24,1,1,1),(1,24,1,16,16),(1,24,7,8,13),(1,24,7,16,16)])
def test_flex_resolves_actual_output_before_allocation(shape):
    engine, context, trt = fake_io()
    engine.get_tensor_profile_shape = lambda *args: (build.DECODER_FLEX_MIN_SHAPE,build.INPUT_SHAPE,build.INPUT_SHAPE)
    expected = build.flex_output_shape(shape)
    context.get_tensor_shape = lambda _: expected
    assert resolve_io(engine, context, trt, shape, profile="flex") == expected
    context.get_tensor_shape = lambda _: (1,3,-1,256,256)
    with pytest.raises(RuntimeError):
        resolve_io(engine, context, trt, shape, profile="flex")
    engine.get_tensor_profile_shape = lambda *args: (build.INPUT_SHAPE,)*3
    with pytest.raises(ValueError):
        resolve_io(engine, context, trt, shape, profile="flex")


@pytest.mark.parametrize("shape", [(2,24,1,16,16),(1,24,2,16,16),(1,24,8,16,16),
                                    (1,24,1,0,16),(1,24,7,17,16),(1,24,7,16,17),(True,24,1,16,16)])
def test_flex_does_not_admit_unsupported_batch_time_or_spatial_values(shape):
    with pytest.raises(ValueError):
        build.flex_output_shape(shape)


def test_flex_builder_checks_profile_readback():
    class Profile:
        def set_shape(self, name, *shapes):
            self.shapes = shapes
        def get_shape(self, name):
            return self.shapes
    profile = Profile()
    build.set_flex_profile(profile, SimpleNamespace(add_optimization_profile=lambda _: 0))
    assert profile.shapes == (build.DECODER_FLEX_MIN_SHAPE,build.INPUT_SHAPE,build.INPUT_SHAPE)
    with pytest.raises(RuntimeError):
        build.set_flex_profile(profile, SimpleNamespace(add_optimization_profile=lambda _: -1))


def test_boundary_inputs_are_exact_real_regions_not_repeated_or_rescaled():
    import torch
    from tools.trt_vae_boundary_probe_worker import make_cases
    image, video = torch.randn(1,24,1,32,64), torch.randn(1,24,22,32,64)
    cases = make_cases(image, video)
    assert len(cases) == 14
    assert torch.equal(cases["image_full"], image.half())
    assert torch.equal(cases["video_t12"], video[:,:,:12,:16,:16].half())
    assert torch.equal(cases["video_seam"], video[:,:,:7,:17,:19].half())
    with pytest.raises(ValueError):
        make_cases(image, video[:,:,:7])


def test_boundary_inference_admission_does_not_relax_runtime_or_old_route_guards():
    from tools.run_trt_vae_video_probe import probe_policy
    ordinary, boundary = probe_policy(), probe_policy(boundary=True)
    assert ordinary.startup_free_gpu_bytes == 12000*1024**2
    assert boundary.startup_free_gpu_bytes == 10*1024**3
    assert ordinary.minimum_free_gpu_bytes == boundary.minimum_free_gpu_bytes == 1024**3
    assert ordinary.minimum_free_ram_bytes == boundary.minimum_free_ram_bytes == 8*1024**3
