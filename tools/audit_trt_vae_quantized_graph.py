"""Verify genuine pinned INT4 weight/DQ structure, not filename-based precision."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import MODEL_SHA, digest_file, write_new_json  # noqa: E402

QUANT_SHA = "0bfcfdc31ea767cb8b0fb371ea1629f0803ee2e81058a2ca7f60ab52a8bc7a2e"


def audit(quant_path, original_path):
    if digest_file(quant_path) != QUANT_SHA or digest_file(original_path) != MODEL_SHA:
        raise ValueError("Unexpected pinned graph identity")
    import onnx
    from onnx import numpy_helper
    import numpy as np
    quant = onnx.load(quant_path, load_external_data=False)
    original = onnx.load(original_path, load_external_data=False)
    onnx.checker.check_model(str(quant_path))
    initial = {v.name:v for v in quant.graph.initializer}
    nodes = list(quant.graph.node)
    dqs = [n for n in nodes if n.op_type == "DequantizeLinear"]
    if len(initial) != 585 or len(dqs) != 144 or any(v.data_location == onnx.TensorProto.EXTERNAL for v in initial.values()):
        raise ValueError("Quantized initializer/DQ count/storage differs")
    rows, quant_names, scale_names = [], set(), set()
    weight_mapping = {}
    for dq in dqs:
        if len(dq.input) != 2 or len(dq.output) != 1:
            raise ValueError("Expected symmetric blockwise weight-only INT4 DQ")
        weight, scale = (initial[name] for name in dq.input)
        attrs = {a.name:onnx.helper.get_attribute_value(a) for a in dq.attribute}
        if weight.data_type != onnx.TensorProto.INT4 or scale.data_type != onnx.TensorProto.FLOAT16 or attrs != {"axis":0,"block_size":64}:
            raise ValueError("Unexpected INT4 dequantization dtype/block contract")
        if len(weight.dims) != 2 or list(scale.dims) != [(weight.dims[0]+63)//64,weight.dims[1]]:
            raise ValueError("Quantized matrix/scale dimensions differ")
        scales = numpy_helper.to_array(scale)
        if not np.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("Invalid quantization scales")
        consumers = [node for node in nodes if dq.output[0] in node.input]
        if len(consumers) != 1 or consumers[0].op_type != "MatMul" or consumers[0].input[1] != dq.output[0]:
            raise ValueError("INT4 DQ is not used as one MatMul weight")
        consumer = consumers[0]
        original_node = next((n for n in original.graph.node if n.name == consumer.name), None)
        if original_node is None or original_node.op_type != "MatMul":
            raise ValueError("Quantized layer has no original counterpart")
        weight_mapping[dq.output[0]] = original_node.input[1]
        quant_names.add(weight.name)
        scale_names.add(scale.name)
        rows.append({"node":consumer.name,"weight":weight.name,"shape":list(weight.dims),"scale":scale.name,"scale_shape":list(scale.dims),"block_size":64,"axis":0})
    if len(quant_names) != 144 or len(scale_names) != 144:
        raise ValueError("Unexpected repeated quantized payloads")
    # Remove only DQ additions and restore the original weight-edge names.
    # Byte-identical remaining nodes prove position/time/output logic unchanged.
    import copy
    restored_nodes = []
    for node in nodes:
        if node.op_type == "DequantizeLinear":
            continue
        node = copy.deepcopy(node)
        for i, value in enumerate(node.input):
            node.input[i] = weight_mapping.get(value, value)
        restored_nodes.append(node)
    # Quantization topologically reorders independent nodes. Compare unique
    # named operator records, while ONNX checker separately proves topology.
    before = {n.name:n.SerializeToString() for n in original.graph.node}
    after = {n.name:n.SerializeToString() for n in restored_nodes}
    if len(before) != len(original.graph.node) or len(after) != len(restored_nodes) or before != after:
        raise ValueError("Quantization changed non-weight graph operators")
    return {"status":"pinned_w4a16_actual_int4_graph_bound_not_engine_or_quality_qualified",
            "repo":"lihaoyun6/MiniMax-H3-VAE-ONNX","revision":"deacbbd48dd3bba13461fa29c2ea8ffb1042c5da",
            "sha256":QUANT_SHA,"bytes":Path(quant_path).stat().st_size,"opset":21,
            "dtype_counts":dict(Counter(onnx.TensorProto.DataType.Name(v.data_type) for v in initial.values())),
            "quantized_matmuls":len(rows),"non_weight_operators_unchanged":True,"rows":rows,
            "node_order":"Independent nodes topologically reordered; unique named operator records unchanged after restoring144 weight edges",
            "limits":"Official named AWQ asset structurally checked. No training/calibration reproduction, inference, visual-quality, memory-benefit or engine compatibility claim. Use strongly-typed TRT without FP16 builder flag."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quant", "original", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.quant, args.original)
    write_new_json(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
