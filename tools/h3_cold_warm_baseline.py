"""Isolated, opt-in full H3 timing: three fresh workers, each cold then warm.

Calls public nodes directly; there is no HTTP service or graph-output cache.
Only loader objects are reused on the warm call. OS file caches are untouched.
The controller must have exclusive GPU scheduling authorization before --run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT.parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_content_identity(sources, runtime_source, runtime_fallback, core, controller):
    """Use content + logical source names, not the random evidence directory."""
    normalized = {}
    roots = ((Path(runtime_source).resolve(), "project/h3_t8/"),
             (Path(runtime_fallback).resolve(), "project/root/"),
             (Path(core).resolve(), "core/"))
    for name, sha256 in sources.items():
        path = Path(name).resolve()
        if path == Path(controller).resolve():
            key = "tools/h3_cold_warm_baseline.py"
        else:
            key = next((prefix + path.relative_to(root).as_posix()
                        for root, prefix in roots if path.is_relative_to(root)), None)
            if key is None:
                raise ValueError("Source is outside the frozen source roots: " + name)
        normalized[key] = sha256
    return json_sha(normalized)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def package(runtime_source=None, runtime_fallback=None):
    name = "t8_h3_baseline_probe"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(runtime_source or (ROOT / "h3_t8")), str(runtime_fallback or ROOT)]
        sys.modules[name] = module
    return name


def recipe():
    return {
        "width": 832, "height": 480, "length": 107, "seconds": 4,
        "seed": 123456789, "steps": 4, "shift_video": 12., "shift_audio": 3.,
        "prompt": "One stable wide shot of a candle flame flickering on a wooden desk. Soft room tone; no speech or music.",
        "unet": "minimax_h3_fl2va_int8_convrot.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "lora": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors",
        "strength": 1., "reserve_vram_gib": 5., "disable_pinned_memory": True,
        "attention": "pytorch", "sampler": "dual_clock_euler", "scheduler": "native_flow",
    }


def prepare_request(core, evidence_root):
    config = recipe()
    folders = {"unet": "diffusion_models", "clip": "text_encoders", "video_vae": "vae",
               "audio_vae": "vae", "lora": "loras"}
    weights = {}
    for key, folder in folders.items():
        path = core / "models" / folder / config[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        weights[key] = {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}
    runtime_source = evidence_root / "runtime-source"
    shutil.copytree(ROOT / "h3_t8", runtime_source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime_fallback = evidence_root / "runtime-root-fallback"
    runtime_fallback.mkdir()
    for path in ROOT.glob("*.py"):
        if not (runtime_source / path.name).exists():
            shutil.copy2(path, runtime_fallback / path.name)
    source_paths = list(runtime_source.rglob("*.py")) + list(runtime_fallback.glob("*.py")) + [Path(__file__)]
    source_paths += [core / relative for relative in (
        "nodes.py", "comfy/ldm/minimax/model.py", "comfy/model_patcher.py",
        "comfy/model_management.py", "comfy/samplers.py", "comfy/sd.py",
        "comfy_extras/nodes_custom_sampler.py", "comfy_extras/nodes_audio.py")]
    sources = {str(path): digest(path) for path in source_paths}
    return {"schema": "t8.h3.owned_cold_warm.v1", "core": str(core), "recipe": config,
            "runtime_source": str(runtime_source),
            "runtime_fallback": str(runtime_fallback),
            "weights": weights, "sources": sources,
            "source_identity": source_content_identity(sources, runtime_source, runtime_fallback, core, __file__),
            "source_identity_schema": "logical_source_content_v1",
            "exclusive_worker_authorized": True}


def worker(request_path):
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != "t8.h3.owned_cold_warm.v1" or request.get("exclusive_worker_authorized") is not True:
        raise ValueError("Use the explicit --run controller")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("Unexpected CUDA device mask; refusing ambiguous GPU identity")
    for path, expected in request["sources"].items():
        if digest(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    for item in request["weights"].values():
        if Path(item["path"]).stat().st_size != item["size"] or digest(item["path"]) != item["sha256"]:
            raise ValueError("Frozen weight changed: " + item["path"])
    core, config = Path(request["core"]), request["recipe"]
    sys.path.insert(0, str(core))
    # Set policy before importing model_management. No user Core is affected.
    import comfy.cli_args
    comfy.cli_args.args.reserve_vram = config["reserve_vram_gib"]
    comfy.cli_args.args.disable_pinned_memory = config["disable_pinned_memory"]
    comfy.cli_args.args.use_pytorch_cross_attention = True
    import torch
    import nodes as core_nodes
    import folder_paths
    import comfy.ldm.modules.attention as attention
    import comfy.model_management as mm
    import comfy.memory_management as memory_management
    from comfy_extras import nodes_custom_sampler as custom
    torch.set_num_threads(2)
    for kind, directory in (("diffusion_models", "diffusion_models"), ("text_encoders", "text_encoders"),
                            ("vae", "vae"), ("loras", "loras")):
        folder_paths.add_model_folder_path(kind, str(core / "models" / directory), is_default=True)
    output = request_path.parent / "output"
    output.mkdir()
    folder_paths.set_output_directory(str(output))
    public = importlib.import_module(package(request["runtime_source"], request["runtime_fallback"]) + ".nodes")
    compat = importlib.import_module(package() + ".h3_lora_compat_advanced")
    delivery = importlib.import_module(package() + ".nodes_h3_av_delivery")
    measurement = importlib.import_module(package() + ".acceleration_measurement")
    runtime = {"gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
               "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
               "cuda_version": torch.version.cuda,
               "driver_version": subprocess.check_output([
                   "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
               "attention_backend": attention.optimized_attention.__name__,
               "aimdo_enabled": bool(memory_management.aimdo_enabled),
               "comfy_compiler_disabled": bool(comfy.cli_args.args.disable_comfy_compiler),
               "pinned_memory_disabled": bool(comfy.cli_args.args.disable_pinned_memory),
               "extra_reserved_vram_bytes": int(mm.EXTRA_RESERVED_VRAM)}
    weights = request["weights"]
    workload = {"model_sha256": weights["unet"]["sha256"], "conditioning_sha256": json_sha({
        "prompt": config["prompt"], "clip": weights["clip"]["sha256"], "task_type": "T2VA"}),
        "video_vae_sha256": weights["video_vae"]["sha256"], "audio_vae_sha256": weights["audio_vae"]["sha256"],
        "width": config["width"], "height": config["height"], "frames": config["length"],
        "fps_num": 24, "fps_den": 1, "seed": config["seed"], "nfe": config["steps"], "cfg": 1.,
        "recipe_sha256": json_sha(config), "core_revision": "actual-source-sha256:" + request["source_identity"],
        "lora_sha256": weights["lora"]["sha256"], "delivered_frames": config["seconds"] * 24}
    loaded = None
    try:
        for cache_state in ("cold", "warm"):
            meter = measurement.SynchronizedCudaMeasurement(workload, cache_state=cache_state,
                        runtime_identity=runtime, exclusive_process=True)
            class DecodeProxy:
                def __init__(self, wrapped, stage):
                    self.wrapped, self.stage = wrapped, stage
                def __getattr__(self, name):
                    return getattr(self.wrapped, name)
                def decode(self, latent):
                    with meter.stage(self.stage):
                        return self.wrapped.decode(latent)
            try:
                with torch.inference_mode():
                    with meter.stage("model_file_loading_or_loader_object_reuse"):
                        if loaded is None:
                            model = core_nodes.UNETLoader().load_unet(config["unet"], "default")[0]
                            video_vae = core_nodes.VAELoader().load_vae(config["video_vae"])[0]
                            audio_vae = core_nodes.VAELoader().load_vae(config["audio_vae"])[0]
                            clip = core_nodes.CLIPLoader().load_clip(config["clip"], "minimax", "default")[0]
                            model, lora_report = compat.load_minimax_h3_lora_model(model, weights["lora"]["path"], config["strength"])
                            loaded = (model, video_vae, audio_vae, clip)
                        else:
                            model, video_vae, audio_vae, clip = loaded
                    with meter.stage("conditioning"):
                        conditioned = public.MiniMaxH3AudioConditioningT8.execute(
                            clip, video_vae, audio_vae, config["prompt"], config["width"], config["height"],
                            config["length"], "T2VA", "native", 1., False, 0, True, "match", "official_2_to_15s").result
                    forward_calls = [0]
                    def count_forward(module, args):
                        forward_calls[0] += 1
                    handle = model.model.diffusion_model.register_forward_pre_hook(count_forward)
                    try:
                        with meter.stage("sampling"):
                            setup = public.MiniMaxH3DualClockSamplerT8.execute(
                                model, conditioned[1], config["steps"], config["shift_video"], config["shift_audio"]).result
                            noise = custom.RandomNoise.execute(config["seed"]).result[0]
                            guider = custom.BasicGuider.execute(setup[0], conditioned[0]).result[0]
                            sampled = custom.SamplerCustomAdvanced.execute(noise, guider, setup[1], setup[2], conditioned[1]).result
                    finally:
                        handle.remove()
                    if forward_calls[0] != config["steps"]:
                        raise RuntimeError(f"Actual forwards {forward_calls[0]} != expected {config['steps']}")
                    decoded = public.MiniMaxH3AVDecodeT8.execute(sampled[0],
                        DecodeProxy(video_vae, "video_vae"), DecodeProxy(audio_vae, "audio_vae")).result
                    if tuple(decoded[0].shape[:3]) != (config["length"], config["height"], config["width"]):
                        raise RuntimeError("Decoded geometry does not match the frozen workload")
                    with meter.stage("trim_and_delivery"):
                        trimmed = public.MiniMaxH3OutputTrimT8.execute(decoded[0], 0., float(config["seconds"]), 24., decoded[1]).result
                        saved = delivery.MiniMaxH3SafeAVSaveT8Advanced.execute(trimmed[0], trimmed[1], cache_state, 18).result
                report = meter.finish()
                report.update(actual_forwards=forward_calls[0], delivery=json.loads(saved[2]),
                    loader_objects_reused=cache_state == "warm", graph_outputs_reused=False,
                    cold_definition="fresh_worker_process_not_OS_file_cache_flush",
                    stage_scope="actual_public_node_calls; device transfers stay in their calling stage",
                    lora_report=json.loads(lora_report))
                write_new(request_path.parent / (cache_state + ".json"), report)
                print(json.dumps({"case": cache_state, "seconds": report["elapsed_seconds"], "output": saved[1]}), flush=True)
                del conditioned, setup, noise, guider, sampled, decoded, trimmed, saved
            except BaseException as error:
                meter.failed = True
                report = meter.finish() if not meter.closed else {"status": "failed"}
                report["error"] = f"{type(error).__name__}: {error}"
                write_new(request_path.parent / (cache_state + "-failed.json"), report)
                raise
    finally:
        mm.unload_all_models()  # This worker only; never user Core.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--pairs", type=int, default=3)
    options = parser.parse_args()
    if options.worker:
        worker(options.worker.resolve())
        return
    if not options.run:
        print(json.dumps(recipe(), indent=2))
        return
    if options.pairs != 3:
        raise ValueError("This qualification requires exactly three cold/warm pairs")
    root = ROOT / "artifacts/development" / ("h3-cold-warm-" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, exist_ok=False)
    request = prepare_request(options.core.resolve(), root)
    write_new(root / "frozen-request.json", request)
    print("Evidence: " + str(root), flush=True)
    env = dict(os.environ, PYTHONUTF8="1", OMP_NUM_THREADS="2")
    for index in range(1, options.pairs + 1):
        case = root / f"pair-{index}"
        case.mkdir()
        write_new(case / "request.json", request)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (case / "worker.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-X", "utf8", __file__, "--worker", str(case / "request.json")],
                cwd=options.core, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            try:
                code = process.wait(timeout=1800)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
            write_new(case / "exit.json", {"pid": process.pid, "exit_code": code, "owned_worker": True})
        if code:
            raise RuntimeError(f"Owned worker failed: {case}; inspect retained log")
        print(f"Pair {index} complete", flush=True)
    measure = importlib.import_module(package() + ".acceleration_measurement")
    reports = {state: [
        json.loads((root / f"pair-{index}" / (state + ".json")).read_text(encoding="utf-8"))
        for index in range(1, 4)] for state in ("cold", "warm")}
    summary = {state: measure.summarize_repeated_measurements(rows) for state, rows in reports.items()}
    hashes = [report["delivery"]["source_rgb8_sha256"] for rows in reports.values() for report in rows]
    summary["repeatability_observation"] = {"decoded_rgb8_all_equal": len(set(hashes)) == 1,
        "rgb8_sha256": hashes, "quality_acceptance": False,
        "note": "Equality is observed after RGB8 conversion; not float-latent parity or human AV review."}
    write_new(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
