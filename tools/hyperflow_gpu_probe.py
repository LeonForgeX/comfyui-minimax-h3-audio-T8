"""One-owned-server original HyperFlow GPU acceptance probe.

Does not modify the user's running Core. Every generated file stays in a
unique evidence directory and only this controller's Popen is stopped.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_progressive_pilot as pilot  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


def recipe(prefix: str, *, length: int = 22, mode: str = "single") -> dict:
    graph = json.loads((ROOT / "tests/fixtures/api/dual_clock_4step_api.json").read_text(encoding="utf-8"))
    graph["1"]["inputs"]["vae_name"] = "minimax_h3_video_vae_int8_convrot.safetensors"
    graph["2"]["inputs"]["vae_name"] = "minimax_h3_audio_vae_fp32.safetensors"
    graph["3"]["inputs"]["clip_name"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    graph["4"]["inputs"]["unet_name"] = "minimax_h3_fl2va_int8_convrot.safetensors"
    graph["5"] = {"class_type": "MiniMaxH3HyperFlowLoaderT8Advanced", "inputs": {
        "model": ["4", 0], "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors"}}
    graph["6"]["inputs"].update(
        prompt="One stable wide shot of a candle flame flickering on a wooden desk. Soft room tone; no speech or music.",
        width=256, height=256, length=length,
    )
    graph["7"] = {"class_type": "MiniMaxH3HyperFlowPlanT8Advanced", "inputs": {"model": ["5", 0]}}
    graph["15"] = {"class_type": "MiniMaxH3HyperFlowSamplerT8Advanced", "inputs": {
        "model": ["5", 0], "hyperflow_plan": ["7", 0], "av_latent": ["6", 1]}}
    graph["9"]["inputs"]["model"] = ["15", 0]
    graph["10"]["inputs"].update(sampler=["15", 1], sigmas=["15", 2])
    graph["12"]["inputs"]["filename_prefix"] = prefix
    if mode in ("single-fl2va", "single-ref2va"):
        # This checked-in local input is used only for an isolated Core probe;
        # the model and the conditioning task must match the tested route.
        graph["25"] = {"class_type": "LoadImage", "inputs": {"image": "10A.jpg"}}
        if mode == "single-fl2va":
            graph["26"] = {"class_type": "LoadImage", "inputs": {"image": "10b.jpg"}}
            graph["6"]["inputs"].update(task_type="FL2VA", first_frame=["25", 0], last_frame=["26", 0])
        else:
            graph["4"]["inputs"]["unet_name"] = "minimax_h3_ref2va_int8_convrot.safetensors"
            graph["6"]["inputs"].update(
                task_type="Ref2VA",
                prompt="Use <Picture 1> as the visual reference. A subtle cinematic portrait with natural room ambience, no speech or music.",
                **{"ref_images.ref_image_0": ["25", 0]},
            )
    if mode in ("split", "upscale", "parity"):
        graph["16"] = {"class_type": "MiniMaxH3HyperFlowLoaderT8Advanced", "inputs": {
            "model": ["4", 0], "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors"}}
    if mode == "split":
        graph["10"] = {"class_type": "MiniMaxH3HyperFlowSplitT8Advanced", "inputs": {
            "model_low": ["5", 0], "model_high": ["16", 0],
            "positive": ["6", 0], "av_latent": ["6", 1],
            "seed": 123456789, "split_interval": 4, "cfg": 1.0}}
    elif mode == "repeat":
        graph["23"] = {"class_type": "SamplerCustomAdvanced", "inputs": dict(graph["10"]["inputs"])}
        graph["24"] = {"class_type": "MiniMaxH3HyperFlowLatentParityT8Advanced", "inputs": {
            "single_av_latent": ["10", 0], "split_av_latent": ["23", 0]}}
        del graph["11"]
        del graph["12"]
    elif mode == "parity":
        graph["23"] = {"class_type": "MiniMaxH3HyperFlowSplitT8Advanced", "inputs": {
            "model_low": ["5", 0], "model_high": ["16", 0],
            "positive": ["6", 0], "av_latent": ["6", 1],
            "seed": 123456789, "split_interval": 4, "cfg": 1.0}}
        graph["24"] = {"class_type": "MiniMaxH3HyperFlowLatentParityT8Advanced", "inputs": {
            "single_av_latent": ["10", 0], "split_av_latent": ["23", 0]}}
        del graph["11"]
        del graph["12"]
    elif mode == "upscale":
        graph["17"] = {"class_type": "MiniMaxH3LearnedLatentUpscaleT8Advanced", "inputs": {
            "av_latent": ["10", 0], "model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
            "size_mode": "scale_by", "scale_by": 2.0, "target_megapixels": 0.26,
            "target_width": 512, "target_height": 512, "aspect_policy": "preserve_source",
            "max_anisotropy": 1.05, "precision": "fp16", "release_policy": "offload_after"}}
        graph["18"] = {"class_type": "MiniMaxH3HyperFlowTailPlanT8Advanced", "inputs": {"model": ["16", 0]}}
        graph["19"] = {"class_type": "MiniMaxH3HyperFlowRefineSamplerT8Advanced", "inputs": {
            "model": ["16", 0], "av_latent": ["17", 0], "hyperflow_plan": ["18", 0]}}
        graph["20"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 123456790}}
        graph["21"] = {"class_type": "BasicGuider", "inputs": {
            "model": ["19", 0], "conditioning": ["6", 0]}}
        graph["22"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["20", 0], "guider": ["21", 0], "sampler": ["19", 1],
            "sigmas": ["19", 2], "latent_image": ["17", 0]}}
        graph["11"]["inputs"]["av_latent"] = ["22", 0]
    return graph


def request_json(server, method: str, path: str, **kwargs):
    server.assert_port_owner()
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, server.url + path, timeout=(4, 45), **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code} {response.text[:4000]}")
        return response.json()


def director_recipe(evidence: Path, variant: str, *, multi_lora: bool = False,
                    distinct_lora: bool = False,
                    output_mp: float = 0.2, ratio: str = "1:1") -> tuple[dict, dict]:
    # Controller-only import of the exact checkout; no user Core is contacted.
    sys.path.insert(0, str(CORE))
    import comfy.cli_args

    comfy.cli_args.args.disable_comfy_compiler = True
    name = "h3_audio_t8_hf_probe"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    assert spec.loader is not None
    spec.loader.exec_module(package)
    from h3_audio_t8_hf_probe.director_generation import build_director_generation_prompt
    from h3_audio_t8_hf_probe.director_project import ProjectStore, new_project

    store = ProjectStore(evidence / "director-user", evidence / "director-input")
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot.update(simplePrompt="A steady cinematic close-up of a candle flame on a wooden desk, soft room tone.",
                ratio=ratio, ownRatio=ratio, manualDuration=1.0, autoDuration=False,
                duration=1.0, end=1.0)
    project["doc"]["ratio"] = ratio
    project["doc"]["generation"].update(
        unet="minimax_h3_fl2va_int8_convrot.safetensors",
        clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
        audio_vae="minimax_h3_audio_vae_fp32.safetensors", lora_mode="none")
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": variant,
        "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors",
        "output_mp": output_mp, "upscaler": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "low_loras": [], "high_loras": [],
    }
    if multi_lora or distinct_lora:
        # Distinct mode adds a second installed H3 content adapter. Header
        # inspection is not proof of semantic compatibility; the GPU probe
        # establishes only load/execute/media integrity, not visual benefit.
        name = "minimax_h3_five_view_512_s1500.safetensors"
        second = str(Path("minimax") / "H3-World" / "step-10000.safetensors") if distinct_lora else name
        project["doc"]["sampling"]["low_loras"] = [
            {"id": "low-a", "name": name, "strength": 0.10, "enabled": True},
            {"id": "low-b", "name": second, "strength": 0.05, "enabled": True},
        ]
        project["doc"]["sampling"]["high_loras"] = [
            {"id": "high-a", "name": name, "strength": 0.07, "enabled": True},
            {"id": "high-b", "name": second, "strength": 0.04, "enabled": True},
        ]
    built = build_director_generation_prompt(project, shot["id"], store, seed=123456789)
    return built["prompt"], {"recipe": built["recipe"], "sampling": built["sampling"], "report": built["report"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8860)
    parser.add_argument("--length", type=int, default=22)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--mode", choices=("single", "single-fl2va", "single-ref2va", "split", "repeat", "parity", "upscale", "director-upscale", "director-upscale-loras", "director-upscale-distinct-loras", "director-partial"), default="single")
    parser.add_argument("--mp", type=float, choices=(0.2, 0.4, 0.5), default=0.2)
    parser.add_argument("--ratio", choices=("1:1", "16:9", "9:16"), default="1:1")
    parser.add_argument("--split-interval", type=int, choices=tuple(range(1, 8)), default=4)
    parser.add_argument("--disable-comfy-compiler", action="store_true")
    parser.add_argument("--same-owner", action="store_true", help="Diagnostic: split stages use one HF loader")
    options = parser.parse_args()
    evidence = ROOT / "artifacts" / "development" / ("hyperflow-gpu-" + uuid.uuid4().hex[:10])
    evidence.mkdir(parents=True, exist_ok=False)
    config = probe_resource_config(CORE, ROOT)
    config["t8_runtime_models"]["hyperflow"] = str(CORE / "models" / "hyperflow" / "loras")
    pilot.write_json(evidence / "paths.json", config)
    pilot.CORE = CORE
    server = pilot.OwnedServer(evidence, options.port, False)
    if options.mode in ("director-upscale", "director-upscale-loras", "director-upscale-distinct-loras", "director-partial"):
        graph, director_report = director_recipe(
            evidence, "upscale4plus4" if options.mode == "director-partial" else "upscale8plus4",
            multi_lora=options.mode == "director-upscale-loras",
            distinct_lora=options.mode == "director-upscale-distinct-loras",
            output_mp=options.mp, ratio=options.ratio,
        )
        pilot.write_json(evidence / "director-report.json", director_report)
    else:
        graph = recipe("MiniMaxH3/HyperFlowEXP/" + options.mode, length=options.length, mode=options.mode)
        if options.mode in ("split", "parity"):
            split_node = "10" if options.mode == "split" else "23"
            graph[split_node]["inputs"]["split_interval"] = options.split_interval
    if options.same_owner and options.mode in ("split", "parity"):
        target = "10" if options.mode == "split" else "23"
        graph[target]["inputs"]["model_high"] = ["5", 0]
    pilot.write_json(evidence / "workflow.json", graph)
    try:
        base_command = pilot.server_command

        def command_with_video_helper(*args, **kwargs):
            command = base_command(*args, **kwargs)
            index = command.index("--extra-model-paths-config")
            command[index:index] = ["ComfyUI-VideoHelperSuite"]
            if options.disable_comfy_compiler:
                command.append("--disable-comfy-compiler")
            return command

        pilot.server_command = command_with_video_helper
        server.start()
        pilot.wait_ready(server, lambda: None, seconds=240)
        nodes = request_json(server, "GET", "/object_info/MiniMaxH3HyperFlowLoaderT8Advanced")
        if not nodes:
            raise RuntimeError("HyperFlow node unavailable in isolated Core")
        submitted = request_json(server, "POST", "/prompt", json={"prompt": graph, "client_id": "hyperflow-probe"})
        pilot.write_json(evidence / "submitted.json", submitted)
        if submitted.get("node_errors"):
            raise RuntimeError("Comfy graph validation failed: " + json.dumps(submitted["node_errors"]))
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + options.timeout
        while time.monotonic() < deadline:
            history = request_json(server, "GET", "/history/" + prompt_id)
            if prompt_id in history:
                pilot.write_json(evidence / "history.json", history[prompt_id])
                status = history[prompt_id]["status"]
                print(json.dumps({"evidence": str(evidence), "status": status}, ensure_ascii=False), flush=True)
                if status.get("status_str") != "success":
                    return 1
                return 0
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during inference")
            time.sleep(2)
        raise TimeoutError("HyperFlow GPU prompt did not complete by deadline")
    finally:
        server.stop()
        pilot.write_json(evidence / "owned-stop.json", server.stop_receipt)
        print("Evidence:", evidence, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
