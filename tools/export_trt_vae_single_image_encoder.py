"""CPU-only native H3 T1 encoder export; no temporal padding or GPU inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(CORE))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_relative_to(PROJECT / "artifacts/acceleration-research-20260909") or root.exists():
        raise ValueError("Use a new isolated research export directory")
    import psutil
    if psutil.virtual_memory().available < 12 * 1024**3 or shutil.disk_usage(PROJECT).free < 3 * 1024**3:
        raise ValueError("Insufficient CPU RAM or disk for a bounded encoder export")
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import onnx
    from safetensors import safe_open
    from tools.trt_vae_encoder_probe_worker import native_encoder_shell
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise ValueError("Export must not initialize CUDA")
    native_sha = digest_file(args.native_vae)
    if native_sha != "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522":
        raise ValueError("Unexpected native video VAE identity")
    root.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), PROJECT / "tools/trt_vae_encoder_probe_worker.py", CORE / "comfy/ldm/minimax/vae.py"]
    identity = {str(path): digest_file(path) for path in sources}
    write_new_json(root / "request.json", {"native_vae": str(args.native_vae.resolve()), "native_sha256": native_sha,
                   "sources": identity, "input": [1, 3, 1, 256, 256], "scope": "Native single-frame encoder export, not inference qualification"})
    try:
        core = native_encoder_shell()
        with safe_open(args.native_vae, framework="pt", device="cpu") as saved:
            state = {key: saved.get_tensor(key) for key in saved.keys() if key.startswith(("encoder.", "quant_conv."))}
        state.update(latents_mean=core.latents_mean, latents_std=core.latents_std)
        core.load_state_dict(state, strict=True, assign=True)
        del state
        core.encoder = core.encoder.half().eval()
        core.quant_conv = core.quant_conv.half().eval()
        if any(value.is_meta or value.device.type != "cpu" for value in (*core.parameters(), *core.buffers())):
            raise ValueError("Encoder weights were not fully materialized on CPU")
        class Encoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, pixels):
                return self.model._encode_moments(pixels)
        encoder = Encoder(core).eval()
        pixels = torch.zeros(1, 3, 1, 256, 256, dtype=torch.float16)
        started = time.perf_counter()
        with torch.inference_mode():
            reference = encoder(pixels)
            if tuple(reference.shape) != (1, 48, 1, 16, 16) or not bool(torch.isfinite(reference).all()):
                raise ValueError("Native single-image encoder shape or values invalid")
            torch.onnx.export(encoder, (pixels,), str(root / "encoder-t1.onnx"), input_names=["pixel_tile"],
                              output_names=["moments_tile"], opset_version=18, dynamo=False,
                              export_params=True, keep_initializers_as_inputs=False, do_constant_folding=True)
        model = onnx.load(root / "encoder-t1.onnx", load_external_data=False)
        onnx.checker.check_model(model)
        def shape(value):
            return [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if shape(model.graph.input[0]) != [1, 3, 1, 256, 256] or shape(model.graph.output[0]) != [1, 48, 1, 16, 16]:
            raise ValueError("Export did not preserve fixed T1 input/output geometry")
        if any(digest_file(path) != sha for path, sha in identity.items()) or torch.cuda.is_initialized():
            raise ValueError("Source identity changed or export initialized CUDA")
        report = {"status": "native_t1_onnx_exported_not_trt_validated", "onnx_sha256": digest_file(root / "encoder-t1.onnx"),
                  "bytes": (root / "encoder-t1.onnx").stat().st_size, "native_sha256": native_sha,
                  "input": shape(model.graph.input[0]), "output": shape(model.graph.output[0]),
                  "opset": 18, "nodes": len(model.graph.node), "initializers": len(model.graph.initializer),
                  "export_seconds": time.perf_counter() - started, "cuda_initialized": False,
                  "scope": "Normalized RGB[-ImageNet] to raw moments; exact native T1 causal path, no17-frame pad. Not a new trained model, compiled engine, quality or speed acceptance."}
        write_new_json(root / "manifest.json", report)
        print(json.dumps(report, indent=2))
    except BaseException as error:
        write_new_json(root / "failure.json", {"status": "export_failed_no_usable_engine", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
