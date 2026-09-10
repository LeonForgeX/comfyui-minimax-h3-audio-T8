"""Independent CPU T1 export evidence and exact native active-weight audit."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from tools.audit_trt_vae_encoder_weights import map_affine, NATIVE_SHA  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    request = json.loads((root / "request.json").read_text(encoding="utf8"))
    report = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf8"))["isolated"]
    guard = json.loads((root / "resource-summary.json").read_text(encoding="utf8"))
    if terminal["status"] != "complete" or terminal["exit_code"] or terminal["active_after_cleanup"] or not terminal["job_assigned_before_task"]:
        raise ValueError("Export Job did not finish cleanly")
    if guard["status"] != "observations_within_policy":
        raise ValueError("Export resource guard failed")
    for path, sha in request["sources"].items():
        source = Path(path)
        if source.suffix == ".py":
            source = root / "source-snapshot" / (sha + "-" + source.name)
        if digest_file(source) != sha:
            raise ValueError("Frozen export source changed")
    if digest_file(request["native_vae"]) != NATIVE_SHA or digest_file(root / "encoder-t1.onnx") != report["onnx_sha256"] or digest_file(root / "reference.safetensors") != report["reference_sha256"]:
        raise ValueError("Native/export/reference identity mismatch")
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    model = onnx.load(root / "encoder-t1.onnx")
    onnx.checker.check_model(model)
    # Normalize only tensor NAMES for following the same downstream graph path;
    # do not modify any on-disk graph operator, initializer or shape.
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = value.removeprefix("model.")
        for index, value in enumerate(node.output):
            node.output[index] = value.removeprefix("model.")
    rows, mapped = [], set()
    with safe_open(request["native_vae"], framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        for initial in model.graph.initializer:
            key, conv = initial.name.removeprefix("model."), None
            affine = key not in keys
            if affine:
                key, conv = map_affine(key, model.graph.node, keys)
            if key in mapped or not key.startswith(("encoder.", "quant_conv.")):
                raise ValueError("Unexpected/duplicate source parameter")
            mapped.add(key)
            native = saved.get_tensor(key).numpy()
            exported = numpy_helper.to_array(initial)
            tap = False
            if affine:
                if native.ndim != 1 or exported.shape != (native.size, 1, 1, 1):
                    raise ValueError("Unexpected native norm affine broadcast")
                native = native.reshape(exported.shape)
            elif native.ndim == 5 and native.shape[2] == 3:
                if not key.startswith("encoder.") or exported.shape != native[:, :, -1:].shape:
                    raise ValueError("Only encoder T1 active causal tap may be specialized")
                native, tap = native[:, :, -1:], True
            if native.dtype != np.float16 or exported.dtype != np.float16 or native.shape != exported.shape or not np.array_equal(native.view(np.uint16), exported.view(np.uint16)):
                raise ValueError("Export changed native active parameter bits: " + key)
            rows.append({"onnx": initial.name, "native": key, "affine": affine, "last_temporal_tap": tap,
                         "following_conv": conv, "bit_equal": True})
        if mapped != {key for key in keys if key.startswith(("encoder.", "quant_conv."))} or len(rows) != 118:
            raise ValueError("Not every native encoder parameter is represented")
    reference = load_file(str(root / "reference.safetensors"))
    if not torch.equal(reference["native_moments"], reference["static_moments"]) or tuple(reference["native_moments"].shape) != (1,48,1,16,16):
        raise ValueError("Saved native/static real moments do not match")
    if not all(bool(torch.isfinite(value).all()) for value in reference.values()):
        raise ValueError("Nonfinite reference evidence")
    image = load_file(request["source_rgb8"])["image"]
    raw = (image[...,128:384,384:640].float()/255*2-1).half()
    mean = raw.new_tensor((.485,.456,.406)).view(1,3,1,1,1)
    std = raw.new_tensor((.229,.224,.225)).view(1,3,1,1,1)
    if not torch.equal(((raw+1)*.5-mean)/std, reference["normalized_pixels"]):
        raise ValueError("Export trace did not use the declared actual reference pixels")
    result = {"status": "t1_export_weights_and_actual_reference_bound_not_trt_qualified", "onnx_sha256": report["onnx_sha256"],
              "native_sha256": NATIVE_SHA, "parameters": len(rows), "exact_active_taps": sum(r["last_temporal_tap"] for r in rows),
              "norm_affines": sum(r["affine"] for r in rows), "native_static_moments_bit_identical": True,
              "cuda_initialized": torch.cuda.is_initialized(), "rows": rows,
              "limits": "Export graph output annotations follow real trace, not proof of TRT output shape. Must compile a separate T1 engine, resolve actual context dimensions and compare native/TRT outputs. No training/new weight values or full image speed acceptance."}
    if result["cuda_initialized"]:
        raise ValueError("Independent export audit must remain CPU-only")
    write_new_json(root / "independent-export-audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
