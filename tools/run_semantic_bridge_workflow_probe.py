"""Execute one explicit candidate on an owned serial server; never deploy or publish."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def graph_assets(graph, core):
    assets = set()
    loader_fields = {"UNETLoader": ("unet_name", "diffusion_models"),
        "CLIPLoader": ("clip_name", "text_encoders"), "VAELoader": ("vae_name", "vae"),
        "MiniMaxH3LoRACompatibilityLoaderT8Advanced": ("lora_name", "loras"),
        "MiniMaxH3SemanticBridgeConfigT8": ("model_name", "semantic_bridge")}
    for node in graph.values():
        spec = loader_fields.get(node["class_type"])
        if spec:
            name, category = spec
            assets.add(core / "models" / category / node["inputs"][name])
        if "upscaler_model" in node["inputs"]:
            assets.add(core / "models/latent_upscale_models" / node["inputs"]["upscaler_model"])
    return [file_identity(path) for path in sorted(assets)]


def audit_video(path, output, expected):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(output.resolve()):
        raise RuntimeError("Video escaped the owned output directory")
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = next(item for item in streams if item["codec_type"] == "video")
    audio = next(item for item in streams if item["codec_type"] == "audio")
    actual = [video["width"], video["height"], int(video["nb_read_frames"]), video["avg_frame_rate"]]
    if actual != expected:
        raise RuntimeError(f"Unexpected geometry/frames/rate: {actual} != {expected}")
    if abs(float(audio["duration"]) - expected[2]/24) > .1:
        raise RuntimeError("Audio duration mismatch")
    for selection in ("0:v:0", "0:a:0"):
        result = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
            "-i", str(path), "-map", selection, "-f", "null", "-"], capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f"Strict decode failed ({selection}, exit={result.returncode}): "
                               + (result.stderr.strip() or "decoder returned no stderr"))
    return {"path": str(path), "sha256": file_identity(path)["sha256"], "geometry": actual,
            "audio_seconds": float(audio["duration"]), "strict_decode": True}


def instrument(graph):
    """Only read report outputs; leave sampling nodes and conditioning untouched."""
    if "8" in graph and "LongVideo" in graph["8"]["class_type"]:
        graph["90"] = {"class_type": "PreviewAny", "inputs": {"source": ["8", 5]}}
        values = graph["8"]["inputs"]
        return "loop", [values["width"], values["height"], round(values["total_duration_seconds"]*24), "24/1"]
    if graph["9"]["class_type"] == "MiniMaxH3PromptRelayConditioningT8Advanced":
        if graph["9"]["inputs"]["execution_mode"] != "apply_exp":
            raise ValueError("Relay qualification must actually apply the timeline")
        graph["90"] = {"class_type": "PreviewAny", "inputs": {"source": ["9", 6]}}
        recipe = "relay"
    elif "40" in graph and graph["40"]["class_type"] == "MiniMaxH3SemanticBridgeApplyT8":
        graph["90"] = {"class_type": "PreviewAny", "inputs": {"source": ["40", 1]}}
        recipe = "external"
    else:
        raise ValueError("Only external, active Relay, or in-node loop candidates are accepted")
    values = graph["9"]["inputs"]
    return recipe, [values["width"], values["height"], 73, "24/1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8221)
    parser.add_argument("--resume-check", action="store_true")
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a fresh artifact directory in this checkout")
    graph = json.loads(args.graph.read_text(encoding="utf8"))
    recipe, expected_media = instrument(graph)
    if args.resume_check and recipe != "loop":
        raise ValueError("Resume check is only for persistent loops")
    transport.CORE, transport.PROJECT = args.core.resolve(), project
    root.mkdir(parents=True)
    sources = transport.source_snapshot()
    sources["tools/run_semantic_bridge_workflow_probe.py"] = file_identity(Path(__file__))["sha256"]
    assets = graph_assets(graph, args.core)
    expected = {"core": verify_core_source(args.core), "sources": sources, "mode": "gpu",
                "pilot_graphs": {recipe: graph}}
    transport.write_json(root / "assets.json", assets)
    transport.write_json(root / "expected.json", expected)
    paths = probe_resource_config(args.core, project)
    paths["t8_runtime_models"]["semantic_bridge"] = str(args.core / "models/semantic_bridge")
    transport.write_json(root / "paths.json", paths)
    result = {"status": "incomplete", "recipe": recipe, "quality_accepted": False,
              "workflow_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest()}
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
            env_history, _ = transport.execute_graph(server, {
                "98": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                "99": {"class_type": "PreviewAny", "inputs": {"source": ["98", 0]}}}, root / "environment", monitor.check)
            result["environment"] = transport.preview_report(env_history, "99")
            history, timing = transport.execute_graph(server, graph, root / "generation", monitor.check, timeout=3600)
            report = transport.preview_report(history, "90")
            result.update(timing=timing, report=report)
            if recipe == "loop":
                if report["status"] != "complete" or report["segment_count"] != 2:
                    raise RuntimeError("Expected two complete sequential segments")
                path = report["final_video_path"]
            else:
                files = list((root / "output").rglob("*.mp4"))
                if len(files) != 1:
                    raise RuntimeError("Expected one completed video")
                path = files[0]
                if recipe == "relay" and not report.get("attention_patch_installed"):
                    raise RuntimeError("Relay attention route was not actually installed")
                if recipe == "external" and not report.get("applied"):
                    raise RuntimeError("External Bridge was not applied")
            result["media"] = audit_video(path, root / "output", expected_media)
            if args.resume_check:
                before = {str(path.relative_to(root / "output")): file_identity(path)["sha256"]
                          for path in (root / "output").rglob("*") if path.is_file()
                          and path.suffix in {".mp4", ".latent", ".safetensors"}}
                resumed, resume_timing = transport.execute_graph(server, graph, root / "resume", monitor.check, timeout=600)
                resumed_report = transport.preview_report(resumed, "90")
                after = {str(path.relative_to(root / "output")): file_identity(path)["sha256"]
                         for path in (root / "output").rglob("*") if path.is_file()
                         and path.suffix in {".mp4", ".latent", ".safetensors"}}
                if before != after or resumed_report.get("final_video_sha256") != report.get("final_video_sha256"):
                    raise RuntimeError("Cache-only resume changed completed latent/media assets")
                result["resume"] = {"report": resumed_report, "timing": resume_timing,
                                    "latent_and_media_files_unchanged": True}
            for name, sha in sources.items():
                if file_identity(project / name)["sha256"] != sha:
                    raise RuntimeError("Generation source changed: " + name)
            for asset in assets:
                if file_identity(Path(asset["path"]))["sha256"] != asset["sha256"]:
                    raise RuntimeError("Generation model asset changed")
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
