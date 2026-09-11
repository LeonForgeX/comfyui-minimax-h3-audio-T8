"""Small, serial CUDA kernel check. No diffusion weights or frontend startup.

Run explicitly; outputs are numerical/call evidence, never video qualification.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import FunctionType, ModuleType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--kj-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core_dir.resolve()))
    # Core CPU startup avoids automatic model allocator/VRAM reservation. Only
    # the explicit tiny tensors below use CUDA, on one GPU in this one process.
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import torch
    torch.set_num_threads(2)
    from comfy.ldm.modules import attention
    root = Path(__file__).resolve().parents[1]
    package = ModuleType("probe_h3_package")
    package.__path__ = [str(root / "h3_t8")]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("probe_h3_package.relay_kj_backend", root / "h3_t8/relay_kj_backend.py")
    backend_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend_module
    spec.loader.exec_module(backend_module)

    payload = args.kj_source.read_bytes()
    filename = str(args.kj_source.resolve())
    tree = ast.parse(payload, filename=filename)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in {"get_sage_func", "PathchSageAttentionKJ"}]
    kj = ModuleType("probe_actual_kj_selector_definitions")
    kj.__file__ = filename
    kj.__dict__.update(torch=torch, logging=logging, wrap_attn=attention.wrap_attn,
                       attention_pytorch=attention.attention_pytorch)
    sys.modules[kj.__name__] = kj
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, "exec", dont_inherit=True), vars(kj))
    codes = tuple(backend_module._codes(compile(payload, filename, "exec", dont_inherit=True)))
    kj.get_sage_func = FunctionType(next(code for code in codes if code.co_name == "get_sage_func"),
                                   vars(kj), argdefs=(False,))

    class ModelShell:
        def __init__(self):
            self.model_options = {"transformer_options": {}}

        def clone(self):
            return copy.deepcopy(self)

    free, total = torch.cuda.mem_get_info(0)
    if free < 2 * 1024**3:
        raise RuntimeError("Tiny kernel probe requires 2 GiB free headroom; no model/application will be closed")
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(20260911)
    q = torch.randn(1, 2, 96, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    k, v = [torch.randn(1, 2, 320, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    bias = torch.zeros(96, 320, device="cuda", dtype=q.dtype)
    # Moderate bias catches natural-log vs exp2 scaling errors, which a nearly
    # saturated -8 mask would make difficult to distinguish.
    bias[:, :160] = -2
    reference = {}
    import sageattention.core as sage_core
    import sageattention.triton.attn_qk_int8_per_block as sage_launch
    original_forward = sage_core.attn_false
    original_jit = sage_launch._attn_fwd
    for masked in (False, True):
        reference[masked] = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias if masked else None)
        reference[masked] = reference[masked].transpose(1, 2).reshape(1, 96, 256)
    rows = []
    for mode in ("auto", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn_qk_int8_pv_fp16_triton"):
        model, = kj.PathchSageAttentionKJ().patch(ModelShell(), mode, False)
        backend = backend_module.capture_kj_relay_backend(model.model_options["transformer_options"]["optimized_attention_override"])
        if backend is None:
            raise RuntimeError("Installed KJ selector failed source/live-code authentication")
        for masked in (False, True):
            output = backend.attention(q, k, v, 2, mask=bias if masked else None, skip_reshape=True)
            torch.cuda.synchronize()
            relative_rms = float((output.float() - reference[masked].float()).square().mean().sqrt()
                                 / reference[masked].float().square().mean().sqrt())
            finite = bool(torch.isfinite(output).all())
            rows.append({"mode": mode, "biased": masked, "finite": finite,
                         "relative_rms_vs_sdpa": relative_rms, "policy": backend.report()})
            print(json.dumps(rows[-1]), flush=True)
            if not finite or relative_rms > .06:
                raise RuntimeError("Kernel numerical guard failed; not qualified")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sage_core.attn_false is not original_forward or sage_launch._attn_fwd is not original_jit:
        raise RuntimeError("Probe changed upstream globals")
    report = {"scope": "tiny serial GPU kernels only; no H3 sampling, speed or human quality claim",
              "gpu": torch.cuda.get_device_name(0), "free_bytes_before": free, "total_bytes": total,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "kj_source_sha256": hashlib.sha256(payload).hexdigest(),
              "adapter_sha256": hashlib.sha256((root / "h3_t8/relay_kj_backend.py").read_bytes()).hexdigest(),
              "scoped_launch_sha256": hashlib.sha256((root / "h3_t8/scoped_sage_triton.py").read_bytes()).hexdigest(),
              "upstream_globals_unchanged": True,
              "rows": rows}
    target = args.output_dir / "kernel-report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(target.resolve()), flush=True)


if __name__ == "__main__":
    main()
