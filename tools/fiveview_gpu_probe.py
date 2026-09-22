"""Run one five-view EXP sheet in an owned Core from a clear full-body reference.

The result is a visual candidate, not an automatic angle/identity qualification.
This controller never submits to or stops the user's running Core.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_progressive_pilot as pilot  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


DEFAULT_PROMPT = (
    "Exactly five full-body turnaround views of the same standing adult man "
    "from <Picture 1>, in order: direct front, front three-quarter, exact side "
    "profile, rear three-quarter, direct 180-degree back. Keep the same pale "
    "yellow patterned polo shirt, light blue jeans, dark shoes, body and face "
    "identity. Neutral standing pose, equal scale, fixed camera distance and "
    "light gray studio background in every view. The last view must show the "
    "back of his head and shirt with no visible face. No walking, no scene "
    "motion, no text."
)


def build_prompt(reference_name: str, prompt: str, seed: int) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
        "2": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0],
            "lora_name": "minimax_h3_five_view_512_s1500.safetensors",
            "strength_model": 1.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "MiniMaxH3FiveViewConditioningEXPT8", "inputs": {
            "clip": ["4", 0], "video_vae": ["5", 0], "ref_image": ["1", 0],
            "prompt": prompt, "size": 512}},
        "7": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "model": ["3", 0], "av_latent": ["6", 1], "steps": 28,
            "shift_video": 12.0, "shift_audio": 3.0,
            "sampler_name": "res_multistep", "scheduler": "simple"}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "BasicGuider", "inputs": {
            "model": ["7", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["8", 0], "guider": ["9", 0], "sampler": ["7", 1],
            "sigmas": ["7", 2], "latent_image": ["6", 1]}},
        "11": {"class_type": "MiniMaxH3FiveViewDecodeEXPT8", "inputs": {
            "five_view_latent": ["10", 0], "video_vae": ["5", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {
            "images": ["11", 0], "filename_prefix": "MiniMaxH3_FiveView_EXP/views"}},
        "13": {"class_type": "SaveImage", "inputs": {
            "images": ["11", 1], "filename_prefix": "MiniMaxH3_FiveView_EXP/sheet"}},
    }


def request_json(server, method: str, path: str, **kwargs):
    server.assert_port_owner()
    with requests.Session() as session:
        session.trust_env = False
        response = session.request(method, server.url + path, timeout=(4, 90), **kwargs)
        if not response.ok:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code} {response.text[:4000]}")
        return response.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=26092205)
    options = parser.parse_args()
    reference = options.reference.resolve(strict=True)
    if reference.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("five-view reference must be a supported still image")
    token = uuid.uuid4().hex[:12]
    evidence = ROOT / "artifacts" / "development" / f"fiveview-redo-{token}"
    (evidence / "input").mkdir(parents=True, exist_ok=False)
    reference_name = "fiveview_reference" + reference.suffix.lower()
    shutil.copy2(reference, evidence / "input" / reference_name)
    pilot.write_json(evidence / "paths.json", probe_resource_config(CORE, ROOT))
    graph = build_prompt(reference_name, options.prompt, options.seed)
    pilot.write_json(evidence / "workflow.json", graph)
    pilot.write_json(evidence / "run.json", {
        "source_reference": str(reference), "prompt": options.prompt, "seed": options.seed,
        "quality_gate": "inspect all five angles; output may fail rear/identity despite GPU success",
    })
    pilot.CORE = CORE
    server = pilot.OwnedServer(evidence, options.port, False)
    base_command = pilot.server_command

    def command_with_owned_input(*args, **kwargs):
        command = base_command(*args, **kwargs)
        index = command.index("--input-directory")
        command[index + 1] = str(evidence / "input")
        command.append("--disable-comfy-compiler")
        return command

    pilot.server_command = command_with_owned_input
    try:
        server.start()
        pilot.wait_ready(server, lambda: None, seconds=240)
        submitted = request_json(server, "POST", "/prompt", json={
            "prompt": graph, "client_id": "fiveview-redo-owned"})
        pilot.write_json(evidence / "submitted.json", submitted)
        if submitted.get("node_errors"):
            raise RuntimeError("five-view graph validation failed: " + json.dumps(submitted["node_errors"]))
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + options.timeout
        while time.monotonic() < deadline:
            history = request_json(server, "GET", "/history/" + prompt_id)
            if prompt_id in history:
                pilot.write_json(evidence / "history.json", history[prompt_id])
                print(json.dumps({"evidence": str(evidence),
                                  "status": history[prompt_id]["status"]},
                                 ensure_ascii=False), flush=True)
                return 0 if history[prompt_id]["status"].get("status_str") == "success" else 1
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during five-view inference")
            time.sleep(3)
        raise TimeoutError("five-view prompt did not complete by deadline")
    finally:
        server.stop()
        pilot.write_json(evidence / "owned-stop.json", server.stop_receipt)
        print("Evidence:", evidence, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
