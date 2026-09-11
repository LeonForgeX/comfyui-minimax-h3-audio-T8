"""Serial tiny H3 projection/RoPE/attention composition, no model checkpoints.

Uses actual installed KJ definitions and native Core blocks. This is not an H3
generation benchmark, nor a full 50-block EAV schedule qualification.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core_dir.resolve()))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import torch
    torch.set_num_threads(2)
    import comfy.model_management as mm
    import comfy.quant_ops
    from comfy import ops
    from comfy.ldm.minimax.model import Attention, DiTBlock, MiniMaxH3Model, rope_rotation_table
    from comfy.model_base import MiniMaxH3
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor
    from comfy_api.latest import io
    free, _ = torch.cuda.mem_get_info(0)
    if free < 2 * 1024**3:
        raise RuntimeError("Tiny projection probe needs 2 GiB free headroom")
    import sageattention  # noqa: F401 -- actual installed kernel, no doubles
    import sageattention.core as sage_core
    root = Path(__file__).resolve().parents[1]
    package = ModuleType("probe_h3_package")
    package.__path__ = [str(root / "h3_t8")]
    sys.modules[package.__name__] = package
    from probe_h3_package import prompt_relay_advanced as relay
    from probe_h3_package import enhance_a_video_advanced as eav
    from probe_h3_package.relay_kj_backend import _codes
    from probe_h3_package.relay_kj_memory import inspect_memory_composition

    kj_dir = args.core_dir / "custom_nodes/ComfyUI-KJNodes/nodes"
    spec = importlib.util.spec_from_file_location("probe_actual_kj_lowmem", kj_dir / "minimax_nodes.py")
    lowmem = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lowmem
    spec.loader.exec_module(lowmem)
    path = kj_dir / "ltxv_nodes.py"
    payload = path.read_bytes()
    compiled = tuple(_codes(compile(payload, str(path), "exec", dont_inherit=True)))
    kj_sage = ModuleType("probe_actual_kj_memory_sage")
    kj_sage.__file__ = str(path)
    kj_sage.__dict__.update(torch=torch, mm=mm, _ck=comfy.quant_ops.ck, io=io, logging=logging,
                            _cuda_archs=sage_core.get_cuda_arch_versions(), _MiniMaxH3Model=MiniMaxH3Model)
    sys.modules[kj_sage.__name__] = kj_sage
    kj_sage.minimax_sageattn_forward = FunctionType(
        next(code for code in compiled if code.co_name == "minimax_sageattn_forward"), vars(kj_sage), argdefs=(None, {}))
    declaration = next(node for node in ast.parse(payload).body if isinstance(node, ast.ClassDef)
                       and node.name == "MiniMaxH3MemoryEfficientSageAttentionPatch")
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(path), "exec", dont_inherit=True), vars(kj_sage))

    base = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(base)
    diffusion = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    diffusion.blocks = torch.nn.ModuleList([DiTBlock(512, 4, 128, 768, 512, 1e-6, 1e-6,
        dtype=torch.bfloat16, device="cuda", operations=ops.disable_weight_init) for _ in range(2)])
    base.diffusion_model = diffusion
    base.requires_grad_(False)
    base.model_sampling = SimpleNamespace(percent_to_sigma=lambda p: 1 - p)
    model = ModelPatcher(base, torch.device("cuda"), torch.device("cpu"))
    generator = torch.Generator(device="cuda").manual_seed(90611)
    with torch.no_grad():
        for value in base.parameters():
            value.copy_(torch.randn(value.shape, device="cuda", dtype=torch.bfloat16, generator=generator) * .025)
        for block in diffusion.blocks:
            block.attn.q_norm.weight.fill_(1)
            block.attn.k_norm.weight.fill_(1)
    layout = relay.build_packed_layout(32, 3, 16, 16, 16, frame_count=15)
    binding = {"schema": relay.PROMPT_RELAY_PATCH_VERSION, "plan_hash": "tiny_memory_projection",
               "text_len": 32, "query_route": "joint_av_exp", "events": [
                   {"text_key_start": 0, "text_key_end": 16, "midpoint": 1., "window": .5, "sigma": 1.},
                   {"text_key_start": 16, "text_key_end": 32, "midpoint": 4., "window": .5, "sigma": 1.}]}
    binding = relay._bind_layout_contract(binding, layout, resolved_task="t2va", keyframes=None, refs=None)
    latent = [torch.zeros(1, 24, 3, 16, 16, device="cuda"), torch.zeros(1, 32, 16, device="cuda")]
    context = torch.zeros(1, 32, 512, device="cuda")
    x = torch.randn(layout.seq_len, 512, device="cuda", dtype=torch.bfloat16, generator=generator)
    angles = torch.randn(layout.seq_len, 48, device="cuda", generator=generator)
    rope = rope_rotation_table(torch.cat([angles, angles], -1), x.dtype)
    sigmas = torch.cat([torch.linspace(1, .05, 20), torch.zeros(1)])
    eav_config = dict(mode="apply_exp", tau=4., start_video_progress=0., end_video_progress=1.,
                      max_workspace_mib=32, g_hard_limit=3.)
    route = relay._runtime_route(layout, binding, x.device)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for case in ("sage_relay", "sage_relay_eav", "sage_eav_only"):
            source = kj_sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
            source = lowmem.MiniMaxLowVRAMAttention.execute(source, 3).result[0]
            source = lowmem.MiniMaxChunkFeedForward.execute(source, 3, 256).result[0]
            runtime = None
            if case == "sage_eav_only":
                patched, runtime, _ = eav.build_eav_model(source, sigmas, **eav_config)
                wrapper_key = eav.EAV_WRAPPER_KEY
            else:
                patched, _ = relay.patch_prompt_relay_model(source, binding, 32)
                wrapper_key = relay.PROMPT_RELAY_WRAPPER_KEY
                if case.endswith("_eav"):
                    patched, runtime, _ = eav.build_eav_prompt_relay_model(patched, sigmas, **eav_config)
                    wrapper_key = eav.EAV_PROMPT_RELAY_WRAPPER_KEY
            backend = inspect_memory_composition(patched)["backend"]
            saved = []
            for method in patched.object_patches.values():
                owner = method.__self__
                saved.append((owner, owner.forward))
                owner.forward = method
            attn = diffusion.blocks[0].attn

            def dense_reference(_, q, k, v, heads, **kwargs):
                bias = torch.zeros(layout.seq_len, layout.seq_len, device=q.device, dtype=q.dtype)
                if case != "sage_eav_only":
                    for segment in route["query_segments"]:
                        for event in binding["events"]:
                            outside = ((segment["query_times"] - event["midpoint"]).abs() - event["window"]).clamp_min(0)
                            bias[segment["start"]:segment["end"], event["text_key_start"]:event["text_key_end"]] = (
                                -.5 * (outside / event["sigma"]).square())[:, None].to(q.dtype)
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                output = output.transpose(1, 2).reshape(1, layout.seq_len, 512)
                if runtime is not None:
                    qs = q[0, :, route["video_start"]:].reshape(4, 3, 64, 128).permute(2, 0, 1, 3)
                    ks = k[0, :, route["video_start"]:].reshape(4, 3, 64, 128).permute(2, 0, 1, 3)
                    probabilities = torch.softmax(((qs / (128 ** .5)) @ ks.transpose(-2, -1)).float(), -1)
                    trace = probabilities.diagonal(dim1=-2, dim2=-1).sum().double()
                    cfi = (64 * 4 * 3 - trace) / (64 * 4 * 3 * 2)
                    gain = torch.clamp_min((3 + 4.) * cfi, 1.).float().to(output.dtype)
                    output[:, route["video_start"]:].mul_(gain)
                return output

            try:
                expected = Attention.forward(attn, x.clone(), rope, {"optimized_attention_override": dense_reference})

                def body(packed, timestep, text, options, **kwargs):
                    handed = [x.clone()]
                    result = attn(handed, rope_freqs=rope, transformer_options=options)
                    if handed:
                        raise RuntimeError("KJ list input was not released")
                    return result

                options = patched.model_options["transformer_options"]
                call_kwargs = {"minimax_payload": {"layout": layout}}
                if case != "sage_eav_only":
                    call_kwargs[relay.PROMPT_RELAY_PAYLOAD_KEY] = binding["binding_hash"]
                actual = WrapperExecutor.new_executor(body, patched.get_wrappers("diffusion_model", wrapper_key)).execute(
                    latent, torch.tensor([500.], device="cuda"), context, options, **call_kwargs)
                torch.cuda.synchronize()
                relative = float((actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt())
                measurements = len(runtime._forwards[0]["g_values"]) if runtime is not None else 0
                if not bool(torch.isfinite(actual).all()) or relative > .08 or (runtime is not None and measurements != 1):
                    raise RuntimeError(f"Projection/RoPE/attention guard failed: {case}, {relative}, {measurements}")
                if relay.PROMPT_RELAY_RUNTIME_KEY in options or eav.EAV_RUNTIME_KEY in options:
                    raise RuntimeError("Runtime route leaked")
                rows.append({"case": case, "relative_rms_vs_dense": relative,
                             "eav_measurements_for_one_block": measurements, "backend": backend.report()})
                print(json.dumps(rows[-1]), flush=True)
            finally:
                for owner, method in saved:
                    owner.forward = method
    report = {"scope": "tiny real H3 projection/RoPE/attention; no diffusion checkpoint, full schedule, or video qualification",
              "gpu": torch.cuda.get_device_name(0), "free_bytes_before": free,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "rows": rows,
              "source_sha256": {name: hashlib.sha256((root / "h3_t8" / name).read_bytes()).hexdigest()
                                  for name in ("relay_kj_memory.py", "relay_kj_backend.py", "scoped_sage_triton.py",
                                               "prompt_relay_advanced.py", "enhance_a_video_advanced.py")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "memory-report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(target.resolve()))
    # Delete patchers while Core callbacks still exist, not at interpreter teardown.
    del patched, source, model


if __name__ == "__main__":
    main()
