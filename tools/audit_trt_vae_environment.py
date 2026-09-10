"""Verify pinned TRT research dependencies and ONNX files without building an engine."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ONNX_SHA = "d2420053272010f701c17cd295d6f89642f9728461c2b74124268de90670f1c1"
DATA_SHA = "8f3e8869794e896628a0b8e6f48c03522172d8715c4d06fe497769a78daadee4"
MODEL_REVISION = "deacbbd48dd3bba13461fa29c2ea8ffb1042c5da"
RUNTIME_VERSION = "10.13.3.9.post1"


def hash_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contained_file(root, name):
    name = Path(name)
    if name.is_absolute():
        raise ValueError("Absolute external-data path is not allowed")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("External data is missing or outside the ONNX directory")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = args.runtime_site.resolve(strict=True)
    if any(name in sys.modules for name in ("tensorrt", "tensorrt_bindings", "tensorrt_libs")):
        raise RuntimeError("Use a fresh process to verify runtime identity")
    sys.path.insert(0, str(runtime))
    import torch
    import onnx
    import tensorrt
    import tensorrt_bindings
    import tensorrt_libs

    modules = {}
    for module in (tensorrt, tensorrt_bindings, tensorrt_libs):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(runtime):
            raise RuntimeError("TensorRT module resolved outside the isolated directory")
        modules[module.__name__] = {"path": str(path), "sha256": hash_file(path)}
    if tensorrt.__version__ != RUNTIME_VERSION:
        raise RuntimeError("Unexpected TensorRT version")
    onnx_path = args.onnx.resolve(strict=True)
    onnx_hash = hash_file(onnx_path)
    if onnx_hash != ONNX_SHA:
        raise ValueError("ONNX does not match the pinned upstream asset")
    model = onnx.load(onnx_path, load_external_data=False)
    locations = {entry.value for tensor in model.graph.initializer
                 for entry in tensor.external_data if entry.key == "location"}
    if locations != {"minimax_h3_vae_decoder.onnx.data"}:
        raise ValueError("Unexpected external data references")
    data = contained_file(onnx_path.parent, next(iter(locations)))
    data_hash = hash_file(data)
    if data_hash != DATA_SHA or data.stat().st_size != 4847062144:
        raise ValueError("External weights do not match the pinned upstream asset")
    mapped = {}
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        kernel32.GetModuleFileNameW.restype = ctypes.c_uint32
        for name in ("nvinfer_10.dll", "nvonnxparser_10.dll", "cudart64_13.dll"):
            handle = kernel32.GetModuleHandleW(name)
            if not handle:
                raise RuntimeError(f"Required DLL is not loaded: {name}")
            buffer = ctypes.create_unicode_buffer(32768)
            if not kernel32.GetModuleFileNameW(handle, buffer, len(buffer)):
                raise ctypes.WinError(ctypes.get_last_error())
            path = Path(buffer.value).resolve()
            if name.startswith(("nvinfer", "nvonnxparser")) and not path.is_relative_to(runtime):
                raise RuntimeError("TRT DLL loaded from outside the isolated runtime")
            mapped[name] = str(path)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "status": "asset_and_import_verified_not_engine_qualified", "python": sys.version,
              "runtime_version": tensorrt.__version__, "runtime_modules": modules, "loaded_dlls": mapped,
              "torch_version": torch.__version__, "torch_path": torch.__file__,
              "torch_cuda": torch.version.cuda, "onnx_version": onnx.__version__,
              "cuda_runtime_package": importlib.metadata.version("nvidia-cuda-runtime"),
              "gpu_readonly_snapshot": gpu.stdout.strip(), "cuda_initialized": torch.cuda.is_initialized(),
              "model_repository": "lihaoyun6/MiniMax-H3-VAE-ONNX", "model_revision": MODEL_REVISION,
              "onnx_sha256": onnx_hash, "data_sha256": data_hash,
              "model_inputs": [{"name": value.name, "element_type": value.type.tensor_type.elem_type,
                                "shape": [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]}
                               for value in model.graph.input],
              "limitation": "Static T7/H16/W16 graph does not qualify T1 or smaller spatial tiles; no engine created"}
    if report["cuda_initialized"]:
        raise RuntimeError("Unexpected Torch CUDA initialization")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
