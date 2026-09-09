"""Real learned H3 encoder CPU parity probe; no GPU, diffusion or audio output.

Loads only encoder/posterior tensors from the unmodified supplied checkpoint.
Decoder weights are not needed to compare the original native encode method with
bounded encoding. Reports this limited scope explicitly; not a VAE GPU-I/O test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def run(args):
    # Set Comfy's CPU option before importing any native model-management module.
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    from safetensors import safe_open
    from comfy.ldm.minimax import audio_vae as native
    from comfy_api.input_impl import VideoFromFile

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("outpaint_cpu_probe_pkg")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    audio = importlib.import_module(package.__name__ + ".video_outpaint_audio")
    audio_file = importlib.import_module(package.__name__ + ".video_outpaint_audio_file")
    media = importlib.import_module(package.__name__ + ".video_outpaint_media")
    planner = importlib.import_module(package.__name__ + ".video_outpaint_plan")
    atomic = importlib.import_module(package.__name__ + ".long_video_delivery")._atomic_write_bytes

    class LearnedEncoder(torch.nn.Module):
        encode = native.MiniMaxH3AudioVAE.encode

        def __init__(self):
            super().__init__()
            self.sample_rate, self.hop_length = 32000, 800
            self.encoder = native.Encoder(64, (2, 4, 4, 5, 5), 2048)
            self.pre_block = native.AttnProjection(2048, 32, 8)
            self.mean_proj = native.ops.Conv1d(32, 32, 1)
            self.register_buffer("latents_mean", torch.empty(32))
            self.register_buffer("latents_std", torch.empty(32))

    model = args.model.resolve(strict=True)
    model_sha = _sha(model)
    stage = LearnedEncoder().eval()
    with safe_open(str(model), framework="pt", device="cpu") as weights:
        state = {name: weights.get_tensor(name) for name in stage.state_dict()}
        loaded_bytes = sum(tensor.numel()*tensor.element_size() for tensor in state.values())
        stage.load_state_dict(state, strict=True, assign=True)
    del state
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in stage.state_dict().values()):
        raise ValueError("CPU parity probe requires actual float32 CPU encoder weights")
    inspection = media.inspect_outpaint_source(VideoFromFile(str(args.source.resolve(strict=True))))
    plan = planner.build_outpaint_plan(source_sha256=inspection["sha256"], width=inspection["width"],
                                     height=inspection["height"], frame_count=inspection["frames"],
                                     aspect="custom", left=32, right=32)
    args.run_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": "t8.h3.outpaint.real_audio_cpu_probe/v1", "model_sha256": model_sha,
              "model": str(model), "encoder_tensor_bytes": loaded_bytes, "source_sha256": inspection["sha256"],
              "source": inspection["path"], "device": "cpu", "threads": 2, "status": "running",
              "native_audio_source_sha256": _sha(native.__file__),
              "bounded_audio_source_sha256": _sha(root / "video_outpaint_audio.py"),
              "audio_file_source_sha256": _sha(root / "video_outpaint_audio_file.py"),
              "probe_source_sha256": _sha(__file__), "torch_version": torch.__version__,
              "rtol": 0.0001, "atol": 0.0001, "cases": [],
              "diffusion_generated": False, "gpu_vae_wrapper_tested": False, "perceptual_acceptance": False}

    def save():
        atomic(args.run_root / "report.json", json.dumps(report, indent=2).encode())

    save()
    try:
        with audio_file.prepare_outpaint_audio_file(inspection, plan, scratch_parent=args.run_root) as source:
            if source is None or source.sample_count < 129*800+17:
                raise ValueError("probe requires at least 3.226 seconds of source audio")
            report["canonical_pcm_sha256"] = source.pcm_sha256
            for samples, block in ((22*800+13, 16), (69*800-19, 32), (129*800+17, 64)):
                wave = source.read(0, samples)
                begin = time.monotonic()
                with torch.inference_mode():
                    expected = stage.encode(wave)
                whole_seconds = time.monotonic()-begin
                reads = []

                def read(start, stop):
                    reads.append(stop-start)
                    return source.read(start, stop)

                begin = time.monotonic()
                actual = torch.cat([tensor for _, tensor in audio.iter_encode_outpaint_audio(
                    stage, read, samples, block_tokens=block, scratch_parent=args.run_root)], dim=-1)
                difference = (actual-expected).abs()
                passed = torch.allclose(actual, expected, rtol=report["rtol"], atol=report["atol"])
                report["cases"].append({"sample_count": samples, "block_tokens": block,
                    "latent_shape": list(actual.shape), "max_abs_error": difference.max().item(),
                    "mean_abs_error": difference.mean().item(), "whole_seconds": whole_seconds,
                    "bounded_seconds": time.monotonic()-begin, "max_waveform_read_samples": max(reads),
                    "actual_latent_std": actual.std().item(), "passes_predeclared_tolerance": passed})
                save()
                print(json.dumps(report["cases"][-1]), flush=True)
                if not passed:
                    raise RuntimeError("learned audio posterior differs beyond the predeclared tolerance")
        if _sha(model) != model_sha:
            raise ValueError("checkpoint changed during parity probe")
        report["status"] = "passed_cpu_learned_encoder_parity_only"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    run(parser.parse_args())
