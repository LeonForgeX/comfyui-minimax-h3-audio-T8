"""Real Comfy H3 audio VAE/cache/cancel-resume probe on CPU only.

Does not run diffusion, write a soundtrack, modify models or touch existing
workflows. The short whole-shot oracle intentionally materializes <=10s audio;
production preparation uses only bounded reads and posterior chunks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def run(args):
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    from safetensors.torch import load_file
    from comfy_api.input_impl import VideoFromFile

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("outpaint_audio_cache_cpu")
    pkg.__path__ = [str(root)]
    sys.modules[pkg.__name__] = pkg
    runtime = importlib.import_module(pkg.__name__+".video_outpaint_audio_runtime")
    media = importlib.import_module(pkg.__name__+".video_outpaint_media")
    planner = importlib.import_module(pkg.__name__+".video_outpaint_plan")
    audio_file = importlib.import_module(pkg.__name__+".video_outpaint_audio_file")
    atomic = importlib.import_module(pkg.__name__+".long_video_delivery")._atomic_write_bytes
    model = args.model.resolve(strict=True)
    inspection = media.inspect_outpaint_source(VideoFromFile(str(args.source.resolve(strict=True))))
    if not inspection["audio_pcm"] or not 40 <= inspection["frames"] <= 240:
        raise ValueError("probe needs a 40..240-frame source with audio")
    plan = planner.build_outpaint_plan(source_sha256=inspection["sha256"], width=inspection["width"],
        height=inspection["height"], frame_count=inspection["frames"], aspect="custom", left=32, right=32,
        cut_frames=(39,))
    args.run_root.mkdir(parents=True, exist_ok=False)
    report = {"schema": "t8.h3.outpaint.real_audio_cache_cpu_probe/v1", "status": "running", "device": "cpu",
              "model_sha256": sha(model), "source_sha256": inspection["sha256"], "shots": len(plan["shots"]),
              "source": inspection["path"], "torch_version": torch.__version__, "threads": 2,
              "implementation_sha256": {name: sha(root / name) for name in (
                  "video_outpaint_audio.py", "video_outpaint_audio_file.py", "video_outpaint_audio_store.py",
                  "video_outpaint_audio_runtime.py", "video_outpaint_audio_decode.py")}, "atol": 0.0001, "rtol": 0.0001,
              "diffusion_generated": False, "cuda_tested": False, "perceptual_acceptance": False}

    def save():
        atomic(args.run_root / "report.json", json.dumps(report, indent=2).encode())

    save()
    try:
        state = load_file(str(model), device="cpu")
        vae = comfy.sd.VAE(sd=state, device=torch.device("cpu"), dtype=torch.float32)
        expected_keys = set(vae.first_stage_model.state_dict())
        if expected_keys != set(state):
            raise ValueError("actual VAE/checkpoint keys differ; no missing/unused-key parity claim")
        del state
        cache = args.run_root / "audio-cache"

        def cancel(progress):
            if progress["new_chunks"] == 1:
                raise RuntimeError("intentional real CPU audio cancellation after first commit")

        try:
            runtime.prepare_outpaint_audio_cache(vae, inspection, plan, cache, block_tokens=32, progress=cancel)
        except RuntimeError as error:
            if str(error) != "intentional real CPU audio cancellation after first commit":
                raise
            report["intentional_cancel_observed"] = True
        else:
            raise RuntimeError("cancellation hook was not reached")
        before = json.loads((cache / "outpaint_source_audio.json").read_text())["chunks"]
        if len(before) != 1:
            raise RuntimeError("cancellation did not retain exactly one audio chunk")
        provider, resume_report = runtime.prepare_outpaint_audio_cache(
            vae, inspection, plan, cache, block_tokens=32, resume=True)
        report["resume"] = resume_report
        report["first_commit_unchanged"] = provider.store.snapshot()["chunks"][0] == before[0]
        report["oracle"] = []
        with audio_file.prepare_outpaint_audio_file(inspection, plan, scratch_parent=args.run_root) as source:
            for shot, total in enumerate(provider.store.totals):
                count, read = source.shot(shot)
                waveform = torch.cat([read(a, min(a+64000, count)) for a in range(0, count, 64000)], dim=-1)
                with torch.inference_mode():
                    expected = vae.first_stage_model.encode(waveform)
                actual = provider.store.read_range(shot, 0, total)
                error = (actual-expected).abs()
                passed = torch.allclose(actual, expected, rtol=report["rtol"], atol=report["atol"])
                report["oracle"].append({"shot": shot, "sample_count": count, "tokens": total,
                                        "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
                                        "passes_predeclared_tolerance": passed})
                print(json.dumps(report["oracle"][-1]), flush=True)
                if not passed:
                    raise ValueError("cached learned posterior differs from whole native CPU encode")
        _, reuse = runtime.prepare_outpaint_audio_cache(vae, inspection, plan, cache, block_tokens=32)
        report["completed_cache_reuse"] = reuse
        if (reuse["chunks_encoded_this_call"] != 0 or reuse["replayed_prefix_chunks"] != 0
                or not report["first_commit_unchanged"] or sha(model) != report["model_sha256"]):
            raise ValueError("reuse or unchanged-model/commit checks failed")
        report["status"] = "passed_real_comfy_audio_cache_cpu_only"
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
