"""CPU-only pinned encoder identity; graph tracing, never value-based key matching."""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402

ENCODER_SHA = "f1b8137f1f60e5a9829af0f50ad93932dbb12ebbdce853011093538886c8a069"
NATIVE_SHA = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
REVISION = "deacbbd48dd3bba13461fa29c2ea8ffb1042c5da"


def map_affine(name, nodes, native_keys):
    """Follow a folded norm parameter to its unique first Conv, not export labels.

    ONNX labels omit down-module indices, so '/block.0/norm1_1' alone cannot
    identify the native block. Conv initializers retain exact checkpoint keys.
    """
    consumers = defaultdict(list)
    for index, node in enumerate(nodes):
        for value in node.input:
            consumers[value].append(index)
    first = consumers[name]
    if len(first) != 1:
        raise ValueError("Affine initializer must have one direct consumer")
    operation = nodes[first[0]].op_type
    if operation not in ("Mul", "Add") or name not in nodes[first[0]].input:
        raise ValueError("Expected exported norm Mul/Add")
    queue, visited, convolutions = deque(first), set(), set()
    while queue:
        index = queue.popleft()
        if index in visited:
            continue
        visited.add(index)
        node = nodes[index]
        if node.op_type == "Conv":
            if len(node.input) < 2:
                raise ValueError("Conv has no weight")
            convolutions.add(node.input[1])
            continue
        if node.op_type not in (
            "Mul",
            "Add",
            "Sigmoid",
            "Reshape",
            "Transpose",
            "Pad",
            "Identity",
            "Cast",
        ):
            raise ValueError("Unexpected operation on norm-to-convolution path")
        for value in node.output:
            queue.extend(consumers[value])
    if len(convolutions) != 1:
        raise ValueError("Norm-to-convolution mapping is absent or ambiguous")
    conv = convolutions.pop()
    match = re.fullmatch(r"(encoder\.down\.\d+\.block\.\d+)\.conv([12])\.weight", conv)
    if match:
        prefix = match[1] + ".norm" + match[2]
    elif conv == "encoder.conv_out.weight":
        prefix = "encoder.norm_out"
    else:
        raise ValueError("Unsupported norm target convolution")
    key = prefix + (".weight" if operation == "Mul" else ".bias")
    if key not in native_keys or conv not in native_keys:
        raise ValueError("Mapped checkpoint key is missing")
    return key, conv


def same_half_bits(native, exported, affine):
    import numpy as np

    if native.dtype != np.float16 or exported.dtype != np.float16:
        return False
    if affine:
        if native.ndim != 1 or exported.shape != (native.size, 1, 1):
            return False
        native = native.reshape(exported.shape)
    return native.shape == exported.shape and np.array_equal(
        native.view(np.uint16), exported.view(np.uint16)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        digest_file(args.onnx) != ENCODER_SHA
        or digest_file(args.checkpoint) != NATIVE_SHA
    ):
        raise ValueError("Unexpected pinned encoder or native checkpoint")
    import onnx
    from onnx import numpy_helper
    from safetensors import safe_open
    import torch

    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    rows, mapped = [], set()
    with safe_open(args.checkpoint, framework="pt", device="cpu") as state:
        keys = set(state.keys())
        for initializer in model.graph.initializer:
            key, conv = initializer.name, None
            affine = key not in keys
            if affine:
                key, conv = map_affine(key, model.graph.node, keys)
            if not key.startswith(("encoder.", "quant_conv.")) or key in mapped:
                raise ValueError("Unexpected or duplicate native key")
            mapped.add(key)
            native = state.get_tensor(key).numpy()
            exported = numpy_helper.to_array(initializer)
            rows.append(
                {
                    "onnx": initializer.name,
                    "native": key,
                    "following_conv": conv,
                    "affine_broadcast": affine,
                    "shape": list(exported.shape),
                    "bit_equal": bool(same_half_bits(native, exported, affine)),
                }
            )
    unused = sorted(
        k for k in keys if k.startswith(("encoder.", "quant_conv.")) and k not in mapped
    )

    def io(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "dtype": tensor.elem_type,
            "shape_annotation": [d.dim_param or d.dim_value for d in tensor.shape.dim],
        }

    report = {
        "status": "encoder_weight_identity_only_not_execution_qualification",
        "onnx_sha256": ENCODER_SHA,
        "native_sha256": NATIVE_SHA,
        "hf_revision": REVISION,
        "initializer_count": len(rows),
        "matched_count": sum(row["bit_equal"] for row in rows),
        "affine_count": sum(row["affine_broadcast"] for row in rows),
        "unused_native_encoder_keys": unused,
        "inputs": list(map(io, model.graph.input)),
        "outputs": list(map(io, model.graph.output)),
        "rows": rows,
        "cuda_initialized": torch.cuda.is_initialized(),
        "auditor_sha256": digest_file(__file__),
        "limits": "Weight identity is not ONNX graph equivalence. Output symbolic shape is not runtime proof. Single-image padding, pixel/latent normalization, TRT compilation and real reference encoding remain unqualified.",
    }
    report["pass"] = (
        len(rows) == 118
        and report["matched_count"] == 118
        and not unused
        and not report["cuda_initialized"]
    )
    write_new_json(args.output.resolve(), report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
