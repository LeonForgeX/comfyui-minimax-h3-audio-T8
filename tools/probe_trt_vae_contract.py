"""CPU-only upstream temporal reproduction and ONNX geometry receipt.

Executes only selected method ASTs from an explicitly supplied trusted local
source snapshot. Uses synthetic RGB, no weights/TRT engine, no GPU evidence.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import onnx

    if torch.cuda.is_initialized():
        raise RuntimeError("This diagnostic must not initialize CUDA")
    torch.set_num_threads(2)
    wanted = {"_decode_temporal_chunks", "_decode_temporal_pad_frames",
              "_decode_temporal_frame_plan", "decode_output_shape", "decode_temporal", "blend"}
    tree = ast.parse(args.upstream.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3TRTVAE")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in methods} != wanted:
        raise ValueError("Upstream methods changed")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(args.upstream), "exec"), namespace)
    obj = SimpleNamespace(token_drop=3, tokens_chunk_size=5, clip_length=17,
                          vae_ratio_t=4, vae_ratio=16, token_overlap=2,
                          frame_pre_padding=3, frame_overlap=5)
    for name in wanted:
        setattr(obj, name, MethodType(namespace[name], obj))
    obj.tiled_decode = lambda z: torch.zeros(z.shape[0], 3, z.shape[2] * 4, z.shape[3] * 16, z.shape[4] * 16)
    obj._finalize_pixels = lambda x: x
    cases = []
    for tokens in (2, 3, 4, 5, 6, 7, 8, 12, 22, 227):
        z = torch.zeros(1, 24, tokens, 1, 1)
        declared = obj.decode_output_shape(z.shape)[2]
        actual = obj.decode_temporal(z).shape[2]
        cases.append({"tokens": tokens, "declared_frames": declared,
                      "actual_synthetic_frames": actual, "excess_frames": actual - declared})
    model = onnx.load(args.onnx, load_external_data=False)

    def spec(value):
        return {"name": value.name, "element_type": value.type.tensor_type.elem_type,
                "shape": [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]}

    external = sorted({entry.value for tensor in model.graph.initializer
                       for entry in tensor.external_data if entry.key == "location"})
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "scope": "synthetic CPU orchestration only; not engine or image quality",
              "upstream_source": str(args.upstream.resolve()), "upstream_sha256": sha(args.upstream),
              "temporal_cases": cases, "onnx_path": str(args.onnx.resolve()), "onnx_sha256": sha(args.onnx),
              "onnx_inputs": [spec(x) for x in model.graph.input],
              "onnx_outputs": [spec(x) for x in model.graph.output],
              "external_data": [{"name": name, "present": (args.onnx.parent / name).is_file()} for name in external],
              "cuda_initialized": torch.cuda.is_initialized()}
    if report["cuda_initialized"]:
        raise RuntimeError("Unexpected CUDA initialization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
