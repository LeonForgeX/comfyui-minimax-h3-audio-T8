from types import SimpleNamespace

import numpy as np
import pytest

from tools.audit_trt_vae_encoder_weights import map_affine, same_half_bits


def node(op, inputs, outputs):
    return SimpleNamespace(op_type=op, input=inputs, output=outputs)


@pytest.mark.parametrize("op,field", [("Mul", "weight"), ("Add", "bias")])
@pytest.mark.parametrize(
    "conv,norm",
    [
        ("encoder.down.3.block.1.conv2.weight", "encoder.down.3.block.1.norm2"),
        ("encoder.conv_out.weight", "encoder.norm_out"),
    ],
)
def test_mapping_uses_real_following_conv_not_nonunique_export_names(
    op, field, conv, norm
):
    graph = [
        node(op, ["activation", "folded"], ["affine"]),
        node("Reshape", ["affine", "shape"], ["reshaped"]),
        node("Transpose", ["reshaped"], ["transposed"]),
        node("Sigmoid", ["transposed"], ["sig"]),
        node("Mul", ["transposed", "sig"], ["silu"]),
        node("Pad", ["silu", "padding"], ["padded"]),
        node("Conv", ["padded", conv], ["result"]),
    ]
    assert map_affine("folded", graph, {conv, norm + "." + field}) == (
        norm + "." + field,
        conv,
    )


@pytest.mark.parametrize(
    "bad", ["two_direct", "two_convs", "missing_key", "unexpected_op", "no_conv"]
)
def test_rejects_ambiguous_or_unmapped_parameters(bad):
    conv = "encoder.conv_out.weight"
    graph = [node("Mul", ["x", "folded"], ["a"]), node("Conv", ["a", conv], ["z"])]
    keys = {conv, "encoder.norm_out.weight"}
    if bad == "two_direct":
        graph.append(node("Add", ["folded", "y"], ["extra"]))
    if bad == "two_convs":
        graph.append(node("Conv", ["a", "different.weight"], ["extra"]))
    if bad == "missing_key":
        keys.remove("encoder.norm_out.weight")
    if bad == "unexpected_op":
        graph[0].op_type = "Sub"
    if bad == "no_conv":
        graph.pop()
    with pytest.raises(ValueError):
        map_affine("folded", graph, keys)


def test_half_bits_and_exact_broadcast_shape_not_tolerant_numeric_match():
    native = np.array([0.0, 1.0], dtype=np.float16)
    exported = native.reshape(2, 1, 1).copy()
    assert same_half_bits(native, exported, True)
    assert not same_half_bits(native, exported.reshape(1, 2, 1), True)
    assert not same_half_bits(native, exported.astype(np.float32), True)
    exported[0] = -0.0
    assert not same_half_bits(native, exported, True)
