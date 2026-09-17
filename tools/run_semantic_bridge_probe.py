"""Serial, content-bound three-way Semantic Bridge samples on an owned server.

No production deployment, no user UI, no package installs. Every case keeps the
same encoder/model/sampler/prompt; only Bridge and the declared seed vary.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from run_h3_memory_node_probe import build_graph as memory_graph, audit_media  # noqa: E402
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402

MODELS = {
    "original": "original/MiniMaxH3_SemanticBridge_v1.safetensors",
    "bunny": "bunny/BUNNY_H3_ActionLogic_Bridge_V1.safetensors",
}
PROMPT = (
    "A single continuous medium shot with a locked camera. Two adults stand behind a plain wooden table. "
    "The woman on the left wears a red jacket and holds a white ceramic cup in her right hand. "
    "The man on the right wears a blue jacket and keeps both hands empty on the table. "
    "The woman slowly lifts her own cup to chest height; the man watches without touching it. "
    "Exactly two people and one cup, no extra hands, stable object ownership and left-right positions. "
    "Soft constant daylight, no moving lights, no scene cuts, no text. Quiet room ambience, "
    "subtle cloth rustle, no dialogue and no music."
)


def collect_reports(history, reports):
    result = {}
    for name, node in reports.items():
        if name != "builder":
            result[name] = transport.preview_report(history, node)
            continue
        texts = history["outputs"][node]["text"]
        if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
            raise RuntimeError("Unexpected plain-text conditioning report")
        text = texts[0]
        bridges = [line[len("semantic_bridge="):] for line in text.splitlines()
                   if line.startswith("semantic_bridge=")]
        if len(bridges) > 1:
            raise RuntimeError("Repeated Bridge application report")
        result[name] = {"text": text, "semantic_bridge": json.loads(bridges[0]) if bridges else None}
    return result


def probe_sources():
    sources = transport.source_snapshot()
    root = Path(__file__).resolve().parents[1]
    for name in ("tools/run_semantic_bridge_probe.py", "tools/semantic_bridge_probe_extension/__init__.py"):
        sources[name] = file_identity(root / name)["sha256"]
    return sources


def build_graph(variant, seed, *, encoder="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                scenario="constraints"):
    if variant not in ("native", *MODELS):
        raise ValueError("Unknown Bridge comparison arm")
    graph, reports = memory_graph(head_chunks=1, ffn_chunks=1)
    graph.pop("3")  # Native Core default attention; no external attention owner.
    graph["10"]["inputs"].update(model=["2", 0], steps=8)
    graph["6"]["inputs"]["clip_name"] = encoder
    graph["9"]["inputs"].update(prompt=PROMPT, width=832, height=480, length=73)
    if scenario != "constraints":
        if scenario not in ("i2va_mandarin", "ref2va_korean"):
            raise ValueError("Unknown fixed Bridge qualification scenario")
        graph["41"] = {"class_type": "LoadImage", "inputs": {
            "image": "reference_a89496c9bdc8cada29af5951.png"}}
        values = graph["9"]["inputs"]
        values.update(width=512, height=768, task_type="auto")
        if scenario == "i2va_mandarin":
            values.update(first_frame=["41", 0], prompt=(
                "A single continuous close-up, beginning with <Picture 1> as the exact first frame. "
                "The same blonde woman wears black sunglasses, a black jacket and a black top, "
                "beside the same pale textured wall. Constant soft daylight, locked camera. "
                "She naturally opens her mouth and clearly says once in Mandarin: <d>你好，很高兴见到你。</d> "
                "Clear calm adult female voice, natural lip motion, quiet ambience, no music, no subtitles, "
                "no extra people, no light sweeps, no scene changes."))
        else:
            graph["1"]["inputs"]["unet_name"] = "minimax_h3_ref2va_int8_convrot.safetensors"
            values.update({"ref_images.ref_image_0": ["41", 0], "prompt": (
                "<Subject 1> is the blonde woman in <Picture 1>, wearing the same black sunglasses, "
                "black jacket and black top. A single continuous close-up beside a pale textured wall, "
                "steady soft daylight, locked camera, no scene changes. She gently sings one short Korean "
                "line once: <d>[Korean] 작은 새가 노래해요</d>. Clear gentle female singing, "
                "a soft sustained piano accompaniment, natural synchronized lips, "
                "no other voices, no subtitles or screen text.")})
    graph["11"]["inputs"]["noise_seed"] = seed
    graph["16"]["inputs"]["filename_prefix"] = f"SemanticBridge/{seed}/{variant}"
    graph["30"] = {"class_type": "T8ProgressiveConditionAudit", "inputs": {
        "positive": ["9", 0], "av_latent": ["9", 1]}}
    graph["12"]["inputs"]["conditioning"] = ["30", 0]
    graph["31"] = {"class_type": "PreviewAny", "inputs": {"source": ["30", 2]}}
    graph["32"] = {"class_type": "PreviewAny", "inputs": {"source": ["9", 5]}}
    reports.update(conditioning="31", builder="32")
    graph["34"] = {"class_type": "T8SemanticBridgeLatentCapture", "inputs": {"av_latent": ["13", 0]}}
    graph["14"]["inputs"]["av_latent"] = ["34", 0]
    graph["35"] = {"class_type": "PreviewAny", "inputs": {"source": ["34", 1]}}
    reports["latent_capture"] = "35"
    if variant != "native":
        graph["33"] = {"class_type": "MiniMaxH3SemanticBridgeConfigT8", "inputs": {
            "model_name": MODELS[variant], "enabled": True, "alpha": .10,
            "magnitude_match": "per_token", "token_scope": "all_tokens", "device": "auto",
            "chunk_tokens": 256}}
        graph["9"]["inputs"]["semantic_bridge"] = ["33", 0]
    return graph, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--variant", choices=("native", *MODELS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--encoder", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--scenario", choices=("constraints", "i2va_mandarin", "ref2va_korean"), default="constraints")
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a fresh artifact directory inside this checkout")
    transport.CORE, transport.PROJECT = args.core.resolve(), project
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        result.insert(result.index("--whitelist-custom-nodes") + 1, "semantic_bridge_probe_extension")
        return result
    transport.server_command = command
    graph, reports = build_graph(args.variant, args.seed, encoder=args.encoder, scenario=args.scenario)
    root.mkdir(parents=True)
    expected = {"core": verify_core_source(args.core), "sources": probe_sources(),
                "mode": "gpu", "pilot_graphs": {"semantic_bridge": graph}}
    assets = [args.core / "models" / category / graph[node]["inputs"][field]
              for node, field, category in (("1", "unet_name", "diffusion_models"),
              ("2", "lora_name", "loras"), ("6", "clip_name", "text_encoders"),
              ("7", "vae_name", "vae"), ("8", "vae_name", "vae"))]
    if args.variant in MODELS:
        assets.append(args.core / "models/semantic_bridge" / MODELS[args.variant])
    if "41" in graph:
        assets.append(args.core / "input" / graph["41"]["inputs"]["image"])
    # Record actual complete asset hashes before the live server owns GPU work.
    asset_identities = [file_identity(path) for path in assets]
    if "41" in graph and asset_identities[-1]["sha256"] != "a89496c9bdc8cada29af5951ecf2f1f4f421142ce68ad3801e6ffe35797bf774":
        raise ValueError("The exact user-selected 2:3 reference image has changed")
    transport.write_json(root / "assets.json", asset_identities)
    # Bridge resolves its own model list; the standard five assets use Core lookup.
    expected["assets"] = asset_identities[:5]
    if "41" in graph:
        expected["assets"].append(asset_identities[-1])
    transport.write_json(root / "expected.json", expected)
    paths = probe_resource_config(args.core, project)
    paths["t8_runtime_models"]["semantic_bridge"] = str(args.core / "models/semantic_bridge")
    transport.write_json(root / "paths.json", paths)
    result = {"status": "incomplete", "variant": args.variant, "seed": args.seed,
              "encoder": args.encoder, "scenario": args.scenario,
              "prompt": graph["9"]["inputs"]["prompt"], "quality_accepted": False}
    guard, server, monitor = ResourceGuard(), transport.OwnedServer(root, args.port, False, 0), None
    lease = args.core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError("Startup resource guard: " + reason)
            server.start()
            monitor = transport.ContinuousGuard(reader, guard, root / "resources.jsonl", server)
            cleanup.callback(monitor.close)
            monitor.start()
            transport.wait_ready(server, monitor.check)
            history, _ = transport.execute_graph(server, {
                "90": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                "91": {"class_type": "PreviewAny", "inputs": {"source": ["90", 0]}}}, root / "environment", monitor.check)
            result["environment"] = transport.preview_report(history, "91")
            info = server.request("GET", "/object_info")
            for node in ("MiniMaxH3SemanticBridgeConfigT8", "MiniMaxH3SemanticBridgeApplyT8"):
                if node not in info:
                    raise RuntimeError("Bridge node registration missing: " + node)
            transport.write_json(root / "object-info.json", info)
            started = time.perf_counter()
            history, timing = transport.execute_graph(server, graph, root / "generation", monitor.check, timeout=1800)
            result.update(timing=timing, wall_seconds=time.perf_counter() - started,
                          reports=collect_reports(history, reports),
                          media=audit_media(root))
            media = result["media"]
            expected_canvas = (graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"], 73, "24/1")
            if (media["width"], media["height"], media["frames"], media["fps"]) != expected_canvas:
                raise RuntimeError("Unexpected Bridge media canvas/frame count/fps")
            observed = probe_sources()
            if observed != expected["sources"]:
                raise RuntimeError("Source changed during generation")
            for identity in asset_identities:
                if file_identity(Path(identity["path"]))["sha256"] != identity["sha256"]:
                    raise RuntimeError("Model or reference asset changed during generation")
            result["status"] = "mechanical_pass_human_pending"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report())
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"status": result["status"], "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
