"""Tiny serial Sol selector capability receipt; no diffusion weights/video."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.core_dir.resolve()))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import torch
    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info(0)
    if free < 2 * 1024**3:
        raise RuntimeError("Tiny Sol test needs 2 GiB free headroom")
    root = Path(__file__).resolve().parents[1]
    for name, path in (("probe_h3_package", root / "h3_t8"),
                       ("probe_actual_sol", args.core_dir / "custom_nodes/ComfyUI-sol-attn")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    nodes = importlib.import_module("probe_actual_sol.nodes")
    adapter = importlib.import_module("probe_h3_package.relay_sol_backend")
    generator = torch.Generator(device="cuda").manual_seed(91107)
    inputs = [torch.randn(1, 4, 256, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)]
    snapshots = [value.clone() for value in inputs]
    rows = []
    # The production adapter now requires Core's authenticated packed H3
    # layout before it may dispatch Sol. Keep the probe on the same contract;
    # an unbound tensor must correctly fall back instead of being mislabeled as
    # an executed Sol call.
    layout = SimpleNamespace(
        seq_len=256,
        segments=[(0, 64, "text"), (64, 128, "audio"), (128, 256, "video")],
    )
    exact_prefix = [0, 2]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for label, tau, biased, short in (("sol_tau0_not_dense", 0., False, False),
                                         ("sol_tau1p3", 1.3, False, False),
                                         ("relay_bias_fallback", 1.3, True, False),
                                         ("relay_query_fallback", 1.3, False, True)):
            backend = adapter.capture_sol_relay_backend(nodes._make_override(tau, 256, True))
            if backend is None:
                raise RuntimeError("Installed Sol source not authenticated")
            q, k, v = inputs
            if short:
                q = q[:, :, :64]
            mask = None
            if biased:
                mask = torch.zeros(256, 256, device="cuda", dtype=q.dtype)
                mask[:, :64] = -4
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            actual = backend.attention(
                q,
                k,
                v,
                4,
                mask=mask,
                skip_reshape=True,
                skip_output_reshape=True,
                transformer_options={"minimax_h3_layout": layout},
            )
            torch.cuda.synchronize()
            relative = float((actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt())
            if not torch.isfinite(actual).all():
                raise RuntimeError(f"Sol numerical guard failed: {label}, {relative}")
            if not biased and not short:
                # tau=0 is mean-threshold sparse routing, NOT all-exact attention.
                # Adapter parity is against the original public Sol entry; SDPA
                # error remains diagnostic, not a fabricated quality threshold.
                upstream = nodes.sol_attn(
                    *(value.transpose(1, 2).contiguous() for value in (q, k, v)),
                    tau=tau,
                    sink_blocks=exact_prefix,
                    sink_q=exact_prefix,
                )
                torch.testing.assert_close(actual, upstream.transpose(1, 2), rtol=0, atol=0)
            else:
                # Core may upcast SDPA to FP32, while this independent control
                # uses BF16 fused SDPA; allow one small BF16 rounding envelope.
                torch.testing.assert_close(actual, expected, rtol=.02, atol=.001)
            expected_calls = 0 if biased or short else 1
            if backend.counters["sol:completed"] != expected_calls:
                raise RuntimeError("Reported actual kernel count is inconsistent")
            for saved, after in zip(snapshots, inputs):
                if not torch.equal(saved, after):
                    raise RuntimeError("Sol mutated Q/K/V")
            rows.append({"case": label, "relative_rms_vs_sdpa": relative, "backend": backend.report()})
            print(json.dumps(rows[-1]), flush=True)
    report = {"scope": "tiny tensor capability test only; no H3 video, speed, or all-shape qualification",
              "gpu": torch.cuda.get_device_name(), "free_bytes_before": free,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "rows": rows,
              "adapter_sha256": hashlib.sha256((root / "h3_t8/relay_sol_backend.py").read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "sol-report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(target.resolve()))


if __name__ == "__main__":
    main()
