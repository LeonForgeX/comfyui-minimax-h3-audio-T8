import pytest
import torch
from comfy.ldm.minimax.vae import CausalConv3d
from tools.trt_vae_t1_export_worker import static_t1_conv, bind_traced_output_shape
from h3_audio_t8_pkg.trt_vae_prepare_t1_worker import static_t1_conv as public_static_t1_conv


@pytest.mark.parametrize("kernel,padding,stride", [(1, 0, 1), (3, 1, 1), (3, (1, 0, 0), (2, 2, 2))])
@pytest.mark.parametrize('factory',[static_t1_conv,public_static_t1_conv])
def test_static_single_frame_kernel_matches_actual_core_exact_active_tap(kernel, padding, stride, factory):
    layer = CausalConv3d(4, 8, kernel_size=kernel, padding=padding, stride=stride)
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    static = factory(layer)
    x = torch.rand(1, 4, 1, 16, 16)
    torch.testing.assert_close(layer(x), static(x), atol=0, rtol=0)
    assert torch.equal(static.weight, layer.weight[:, :, -1:])
    with pytest.raises(ValueError, match="exactly one"):
        static(x.expand(-1, -1, 2, -1, -1))


def test_static_output_annotation_never_changes_graph_or_contradicts_fixed_dimensions():
    import onnx
    from onnx import helper, TensorProto
    model = helper.make_model(helper.make_graph([], "test", [helper.make_tensor_value_info("pixel_tile", TensorProto.FLOAT16, [1,3,1,256,256])],
                                    [helper.make_tensor_value_info("moments_tile", TensorProto.FLOAT16, ["b",48,"t","h","w"])]))
    assert bind_traced_output_shape(model, (1,48,1,16,16)) == ["b",48,"t","h","w"]
    assert [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim] == [1,48,1,16,16]
    assert not model.graph.node and not model.graph.initializer
    model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 15
    with pytest.raises(ValueError, match="contradicts"):
        bind_traced_output_shape(model, (1,48,1,16,16))
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
