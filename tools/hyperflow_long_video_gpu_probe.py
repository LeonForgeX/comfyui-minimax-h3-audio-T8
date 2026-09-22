"""Owned-Core P7 probe for the isolated HyperFlow long-video EXP recipe.

This controller never contacts or stops the user's existing Core. It creates a
unique chain and evidence directory, and stops only its own Popen process.
"""
from __future__ import annotations

import argparse
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
from build_hyperflow_long_video_workflow import NODE, build_prompt  # noqa: E402
from build_hyperflow_single8_long_video_workflow import (  # noqa: E402
    NODE as SINGLE8_NODE, build_prompt as build_single8_prompt,
)
from vdn_probe_environment import probe_resource_config  # noqa: E402


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
    parser.add_argument("--port", type=int, default=8867)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--variant", choices=("dual4plus4", "single8"), default="dual4plus4")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--low-width", type=int, default=448)
    parser.add_argument("--low-height", type=int, default=224)
    parser.add_argument("--prompt", default=None,
                        help="Optional scene prompt; omit to keep the established comparison prompt")
    options = parser.parse_args()
    if options.timeout < 600:
        raise ValueError("P7 probe timeout must allow a full two-segment run")
    if (any(value < 32 or value % 32 for value in (options.width, options.height))
            or (options.variant == "dual4plus4" and
                ((options.width, options.height) !=
                 (2 * options.low_width, 2 * options.low_height)
                 or any(value < 32 or value % 32 for value in
                        (options.low_width, options.low_height))))):
        raise ValueError("Probe requires 32-grid dimensions; dual mode requires exact 2x LOW/HIGH")
    token = uuid.uuid4().hex[:12]
    stem = "hyperflow-long-video-p7" if options.variant == "dual4plus4" else "hyperflow-single8-long-video"
    evidence = ROOT / "artifacts" / "development" / f"{stem}-{token}"
    evidence.mkdir(parents=True, exist_ok=False)
    chain_id = (f"hf_long_video_p7_{token}" if options.variant == "dual4plus4"
                else f"hf_single8_long_video_{token}")
    config = probe_resource_config(CORE, ROOT)
    config["t8_runtime_models"]["hyperflow"] = str(CORE / "models" / "hyperflow" / "loras")
    pilot.write_json(evidence / "paths.json", config)
    pilot.CORE = CORE
    server = pilot.OwnedServer(evidence, options.port, False)
    base_command = pilot.server_command

    def command_with_video_helper(*args, **kwargs):
        command = base_command(*args, **kwargs)
        index = command.index("--extra-model-paths-config")
        command[index:index] = ["ComfyUI-VideoHelperSuite"]
        command.append("--disable-comfy-compiler")
        return command

    pilot.server_command = command_with_video_helper
    try:
        server.start()
        pilot.wait_ready(server, lambda: None, seconds=240)
        selected_node = NODE if options.variant == "dual4plus4" else SINGLE8_NODE
        info = request_json(server, "GET", f"/object_info/{selected_node}")
        if selected_node not in info:
            raise RuntimeError(f"{selected_node} unavailable in owned Core")
        if options.variant == "dual4plus4":
            graph = build_prompt(
                info, chain_id=chain_id,
                **({"prompt": options.prompt} if options.prompt is not None else {}),
                width=options.width, height=options.height,
                low_width=options.low_width, low_height=options.low_height)
        else:
            comparison_prompt = options.prompt or (
                "A steady cinematic shot of a quiet candle-lit room, natural motion and room tone.")
            graph = build_single8_prompt(
                info, chain_id=chain_id, prompt=comparison_prompt,
                width=options.width, height=options.height)
        pilot.write_json(evidence / "workflow.json", graph)
        pilot.write_json(evidence / "run.json", {
            "chain_id": chain_id,
            "expected_chain_root": str(evidence / "output" / "minimax_h3_t8_long_video" / chain_id),
            "expected_final_folder": str(evidence / "output" / "minimax_h3_t8_long_video" / chain_id / "assembled"),
            "variant": options.variant,
            "output_geometry": [options.width, options.height],
            "low_geometry": ([options.low_width, options.low_height]
                             if options.variant == "dual4plus4" else None),
            "quality_gate": "manual picture/audio/seam review required",
        })
        submitted = request_json(server, "POST", "/prompt", json={
            "prompt": graph, "client_id": "hyperflow-long-video-p7-owned"})
        pilot.write_json(evidence / "submitted.json", submitted)
        if submitted.get("node_errors"):
            raise RuntimeError("P7 graph validation failed: " + json.dumps(submitted["node_errors"]))
        prompt_id = submitted["prompt_id"]
        deadline = time.monotonic() + options.timeout
        while time.monotonic() < deadline:
            history = request_json(server, "GET", "/history/" + prompt_id)
            if prompt_id in history:
                pilot.write_json(evidence / "history.json", history[prompt_id])
                status = history[prompt_id]["status"]
                print(json.dumps({"evidence": str(evidence), "chain_id": chain_id,
                                  "status": status}, ensure_ascii=False), flush=True)
                return 0 if status.get("status_str") == "success" else 1
            if server.process.poll() is not None:
                raise RuntimeError("Owned Core exited during P7 inference")
            time.sleep(3)
        raise TimeoutError("P7 prompt did not complete by deadline")
    finally:
        server.stop()
        pilot.write_json(evidence / "owned-stop.json", server.stop_receipt)
        print("Evidence:", evidence, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
