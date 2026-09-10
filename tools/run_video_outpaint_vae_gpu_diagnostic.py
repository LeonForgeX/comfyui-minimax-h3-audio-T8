"""Serial learned-VAE diagnostic on existing latents; never reruns H3 diffusion.

The 90-frame full native oracle intentionally materializes a short RGB batch.
It is diagnostic evidence, not the bounded production route or a long-video test.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types


def run(args):
    import comfy.cli_args
    comfy.cli_args.args.reserve_vram = 3.0
    import comfy.sd
    import comfy.utils
    import torch
    import psutil
    import pynvml
    from PIL import Image
    from safetensors.torch import load_file
    from run_video_outpaint_t8_probe import LiveMemorySampler, upstream

    root = Path(__file__).resolve().parents[1]
    sha, atomic = upstream._sha256_file, upstream._atomic_json
    if any(upstream.probe.port_is_listening("127.0.0.1", port) for port in (8188, 8190, 8191)):
        raise RuntimeError("another Comfy server is live; do not overlap this diagnostic")
    pynvml.nvmlInit()
    if pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free < 10000*1024**2:
        raise RuntimeError("VAE diagnostic needs at least 10000MiB free VRAM")
    if psutil.virtual_memory().available < 16*1024**3:
        raise RuntimeError("short full native oracle requires at least 16GiB free system RAM")
    if not args.confirm_run:
        print("VAE-only diagnostic preflight ready; --confirm-run is required")
        return
    args.run_root.mkdir(parents=True, exist_ok=False)
    package = types.ModuleType("outpaint_vae_diagnostic")
    package.__path__ = [str(root / "h3_t8"), str(root)]
    sys.modules[package.__name__] = package
    def module(name):
        return importlib.import_module(package.__name__+".video_outpaint_"+name)
    planner, prepare, decoder = module("plan"), module("prepare"), module("decode")
    source_runtime, source_store = module("source_runtime"), module("source_store")
    prior = args.sampled_run.resolve(strict=True)
    prior_report = json.loads((prior / "report.json").read_text())
    plan = planner.build_outpaint_plan(source_sha256=prior_report["source_sha256"], width=512, height=288,
        frame_count=90, aspect="custom", top=192, bottom=192, window_frames=90, generation_megapixels=0.5)
    cache = prior / "output/T8_H3_Outpaint_Cache" / ("t8_clean90-"+plan["plan_sha256"][:16])
    record = json.loads((cache / "sampling/outpaint_windows.json").read_text())["committed"][0]
    asset = cache / "sampling" / f"window-{record['sha256']}.safetensors"
    if not asset.is_file() or sha(asset).lower() != record["sha256"]:
        raise ValueError("sampled latent asset does not match its committed hash")
    sampled = load_file(str(asset))["video"]
    source = prior / "input/source_512x288_90f.mp4"
    if sha(source).lower() != plan["source"]["sha256"]:
        raise ValueError("source file hash mismatch")
    checkpoint = root.parents[1] / "models/vae/minimax_h3_video_vae_fp16.safetensors"
    report = {"status": "running", "scope": "learned VAE only; no diffusion, no model writes, no human acceptance",
        "sampled_asset_sha256": record["sha256"], "source_sha256": plan["source"]["sha256"],
        "vae_file_sha256": sha(checkpoint), "decoder_sha256": sha(root / 'h3_t8/video_outpaint_decode.py')}
    atomic(args.run_root / "report.json", report)
    telemetry = LiveMemorySampler(os.getpid(), args.run_root / "telemetry.live.jsonl")
    telemetry.start()
    try:
        torch.set_num_threads(2)
        with source_runtime.outpaint_gpu_lease(), torch.no_grad():
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(checkpoint)))
            identity = source_runtime.loaded_video_vae_identity(vae)
            stored = source_store.OutpaintSourceStore(cache / "source", plan, video_vae_sha256=identity["sha256"])
            with prepare.SequentialOutpaintFrameReader(source, plan) as reader:
                pixels = torch.cat([reader(a, min(a+17, 90)) for a in range(0, 90, 17)]).float()/255
            print("Encoding native whole source for comparison with the saved bounded source", flush=True)
            native_source = vae.encode(pixels)
            source_error = (native_source-stored.read_range(0, 0, 27)[:, :, :, 12:30]).abs()
            report["source_encode"] = {"max_abs": float(source_error.max()), "mean_abs": float(source_error.mean()),
                                       "within_1e_4": bool(torch.all(source_error <= 1e-4))}
            del pixels, native_source, source_error
            print("Decoding identical sampled latents through native and bounded decoders", flush=True)
            native = vae.decode(sampled)[0]
            if tuple(native.shape) != (90, 672, 512, 3):
                raise ValueError(f"unexpected native decode shape: {tuple(native.shape)}")
            def image_at(value, path):
                Image.fromarray((value.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy()).save(path)
            image_at(native[45], args.run_root / "native-frame45.png")
            maximum, total, elements = 0.0, 0.0, 0
            for start, pixels, _ in decoder.iter_decode_outpaint_shot(vae, lambda a, b: sampled[:, :, a:b], plan):
                error = (pixels-native[start:start+len(pixels)]).abs()
                maximum = max(maximum, float(error.max()))
                total += float(error.double().sum())
                elements += error.numel()
                if start <= 45 < start+len(pixels):
                    image_at(pixels[45-start], args.run_root / "bounded-frame45.png")
            report["decode"] = {"max_abs": maximum, "mean_abs": total/elements, "within_1e_4": maximum <= 1e-4}
            report["status"] = "diagnostic_complete_not_perceptual_acceptance"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        telemetry.stop()
        report["memory"] = telemetry.summary()
        report["telemetry_error"] = telemetry.journal_error
        atomic(args.run_root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--sampled-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    run(parser.parse_args())
