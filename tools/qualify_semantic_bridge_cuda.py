"""Small, serial real CUDA Bridge math checks; no H3 diffusion or user UI."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a new task-owned evidence directory")
    root.mkdir(parents=True)
    result = {"status": "incomplete", "diffusion_forwards": 0, "quality_accepted": False}
    lease = args.core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader:
            guard = ResourceGuard()
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError("Startup resource guard: " + reason)
            import torch
            torch.set_num_threads(2)
            spec = importlib.util.spec_from_file_location("standalone_semantic_bridge", project / "h3_t8/semantic_bridge.py")
            sb = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sb
            spec.loader.exec_module(sb)
            inputs = torch.randn(2, 17, 5120, generator=torch.Generator().manual_seed(91717))
            inputs[:, 0] = 0
            tags = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
            models = [
                ("original/MiniMaxH3_SemanticBridge_v1.safetensors", "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors"),
                ("bunny/BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors"),
            ]
            result.update(torch=torch.__version__, device=torch.cuda.get_device_name(0),
                          allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                          matmul_precision=torch.get_float32_matmul_precision(),
                          sources=[file_identity(Path(__file__)), file_identity(project / "h3_t8/semantic_bridge.py")])
            rows, identities = [], []
            for original, wrapper in models:
                path = args.core / "models/semantic_bridge" / original
                compat = args.core / "models/semantic_bridge" / wrapper
                identities.extend([file_identity(path), file_identity(compat)])
                cfg = sb.BridgeConfig(str(path), sb.file_sha(path), device="cuda", chunk_tokens=9)
                wrapped = replace(cfg, path=str(compat), sha256=sb.file_sha(compat))
                for mode in ("per_token", "global", "none"):
                    for dtype in (torch.float32, torch.float16, torch.bfloat16):
                        source = [[inputs.to(dtype), {"minimax_token_tags": tags, "start_percent": .25}]]
                        chosen = replace(cfg, magnitude_match=mode)
                        cpu, _ = sb.apply_bridge(source, replace(chosen, device="cpu"), cancel=lambda: None)
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        actual, receipt = sb.apply_bridge(source, chosen, cancel=lambda: None)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        same, _ = sb.apply_bridge(source, replace(wrapped, magnitude_match=mode), cancel=lambda: None)
                        if not torch.equal(actual[0][0], same[0][0]):
                            raise RuntimeError("Lossless wrapper changed same-profile CUDA output")
                        # F16/BF16 rounding can place close FP32 values on opposite quantization bins.
                        atol = {torch.float32: 2e-5, torch.float16: .004, torch.bfloat16: .04}[dtype]
                        torch.testing.assert_close(actual[0][0], cpu[0][0], atol=atol, rtol=atol)
                        if actual[0][0].device != source[0][0].device or actual[0][0].dtype != dtype:
                            raise RuntimeError("Bridge did not restore conditioning dtype/device")
                        if actual[0][1]["minimax_token_tags"] is not tags:
                            raise RuntimeError("Bridge changed token metadata")
                        rows.append({"model": original, "mode": mode, "dtype": str(dtype),
                            "wrapper_bit_equal": True, "cpu_cuda_max_abs": float((actual[0][0].float()-cpu[0][0].float()).abs().max()),
                            "seconds": elapsed, "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                            "input_sha256": receipt["items"][0]["input_sha256"],
                            "output_sha256": receipt["items"][0]["output_sha256"]})
                calls = []
                class CancelledProbe(Exception):
                    pass
                def cancel():
                    calls.append(1)
                    if len(calls) == 3:
                        raise CancelledProbe("cancel inside token chunks")
                try:
                    sb.apply_bridge([[inputs, {"minimax_token_tags": tags}]], cfg, cancel=cancel)
                except CancelledProbe:
                    pass
                else:
                    raise RuntimeError("Cancellation was ignored")
                expected, _ = sb.apply_bridge([[inputs, {"minimax_token_tags": tags}]], cfg, cancel=lambda: None)
                preserved, _ = sb.apply_bridge([[inputs, {"minimax_token_tags": tags}]],
                    replace(cfg, token_scope="text_only_preserve_reference"), cancel=lambda: None)
                if (not torch.equal(preserved[0][0][:, tags == 0], inputs[:, tags == 0])
                        or not torch.equal(preserved[0][0][:, tags == 1], expected[0][0][:, tags == 1])):
                    raise RuntimeError("CUDA text-only token scope altered reference rows or text math")
                if any(value.device.type != "cpu" for state in sb._CACHE.values() for value in state.values()):
                    raise RuntimeError("Bridge retained cached weights on GPU")
            for identity in identities + result["sources"]:
                if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
                    raise RuntimeError("A qualification input/source changed")
            result.update(status="math_pass_not_video_quality", cases=rows, assets=identities,
                          cancellation_and_next_call_passed=True, persistent_gpu_cache=False)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with (root / "terminal.json").open("x", encoding="utf8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"status": result["status"], "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
