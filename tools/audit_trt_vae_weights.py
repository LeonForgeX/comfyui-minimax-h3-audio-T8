"""Compare every pinned decoder ONNX initializer to a native FP16 checkpoint.

Named constants compare directly. Export-folded MatMul weights must map through
their actual sole consuming node and match the transposed native weight exactly.
This proves weight identity only, not numerical equivalence of the ONNX graph.
"""

import argparse
import ast
import copy
import json
from pathlib import Path

from audit_trt_vae_environment import DATA_SHA, ONNX_SHA, contained_file, hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--core-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import onnx
    from onnx import numpy_helper
    from safetensors import safe_open
    import torch

    onnx_path = args.onnx.resolve(strict=True)
    if hash_file(onnx_path) != ONNX_SHA:
        raise ValueError("Unexpected ONNX source")
    model = onnx.load(onnx_path, load_external_data=False)
    data = contained_file(onnx_path.parent, "minimax_h3_vae_decoder.onnx.data")
    if hash_file(data) != DATA_SHA:
        raise ValueError("Unexpected ONNX data")
    rows = []
    with safe_open(args.checkpoint, framework="pt", device="cpu") as state:
        keys = set(state.keys())
        metadata = json.loads(state.metadata()["minimax_h3_video_vae"])
        for initializer in model.graph.initializer:
            key = initializer.name
            transpose = key not in keys
            if transpose:
                consumers = [node for node in model.graph.node if key in node.input]
                if len(consumers) != 1 or consumers[0].op_type != "MatMul" or consumers[0].input[1] != key:
                    raise ValueError(f"Cannot map initializer: {key}")
                key = consumers[0].name.rsplit("/", 1)[0].strip("/").replace("/", ".") + ".weight"
            if key not in keys:
                raise ValueError(f"Native weight missing: {key}")
            native = state.get_tensor(key).numpy()
            if transpose:
                if native.ndim != 2:
                    raise ValueError("Folded MatMul native weight is not 2D")
                native = native.T
            # numpy_helper may fill raw_data in place; don't retain every weight.
            exported = numpy_helper.to_array(copy.deepcopy(initializer), base_dir=str(onnx_path.parent))
            equal = (native.shape == exported.shape and native.dtype == exported.dtype
                     and np.array_equal(native.view(np.uint16), exported.view(np.uint16)))
            rows.append({"onnx": initializer.name, "native": key, "transpose": transpose,
                         "shape": list(exported.shape), "bit_equal": bool(equal)})
            del native, exported
    constants = {}
    for node in ast.parse(args.core_vae.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("LATENTS_MEAN", "LATENTS_STD"):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    normalization = all(np.array_equal(np.asarray(metadata[field], dtype=np.float32),
                                       np.asarray(constants[field.upper()], dtype=np.float32))
                        for field in ("latents_mean", "latents_std"))
    report = {"scope": "CPU bitwise weight identity only; no graph/engine/GPU quality qualification",
              "onnx_sha256": ONNX_SHA, "data_sha256": DATA_SHA,
              "native_checkpoint": str(args.checkpoint.resolve()), "native_sha256": hash_file(args.checkpoint),
              "core_vae_sha256": hash_file(args.core_vae), "normalization_fp32_equal": normalization,
              "initializer_count": len(rows), "matched_count": sum(row["bit_equal"] for row in rows),
              "unused_native_decoder_keys": sorted(key for key in keys
                                                    if key.startswith(("decoder.", "post_quant_conv."))
                                                    and key not in {row["native"] for row in rows}),
              "rows": rows, "cuda_initialized": torch.cuda.is_initialized()}
    report["pass"] = normalization and report["matched_count"] == len(rows) and not report["cuda_initialized"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
