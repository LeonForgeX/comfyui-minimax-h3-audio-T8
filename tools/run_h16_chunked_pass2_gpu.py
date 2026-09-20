"""Owned H16-3 73-frame 4+4 GPU qualification; output still needs human review."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_progressive_pilot as transport  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    NvmlResourceReader,
    ResourceGuard,
    SerialProbeLease,
    file_identity,
)
from run_h3_memory_node_probe import audit_media  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


PROMPT = (
    "One continuous locked medium close-up of the same adult woman from the first frame. "
    "She says clearly exactly once: <d>H sixteen audio refinement test.</d> "
    "After speaking she closes her lips and remains still. Natural synchronized mouth motion, "
    "clean dry voice, no added words, no repetition, no music, no subtitles, no cuts."
)


def build_graph(project: Path) -> dict:
    """Compile the saved formal workflow into its exact API graph for this probe."""
    path = (
        project
        / "examples/workflows/13-latent-upscale"
        / "2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    widget_names = {
        1: ["vae_name"],
        2: ["vae_name"],
        3: ["clip_name", "type", "device"],
        4: ["unet_name", "weight_dtype"],
        5: ["lora_name", "strength_model"],
        6: ["image"],
        7: [
            "prompt", "width", "height", "length", "task_type", "audio_mode",
            "audio_denoise_strength", "add_source_as_reference",
            "prompt_primary_audio_ordinal", "strict_prompt_tags", "ref_image_size",
            "reference_video_policy", "allow_above_reference_area",
        ],
        8: ["steps", "shift_video", "shift_audio", "sampler_name", "scheduler"],
        9: ["base_steps", "coarse_steps", "refine_steps"],
        10: [],
        11: ["noise_seed"],
        12: [],
        13: [
            "model_name", "size_mode", "scale_by", "target_megapixels",
            "target_width", "target_height", "aspect_policy", "max_anisotropy",
            "precision", "release_policy",
        ],
        14: [
            "prompt", "width", "height", "length", "task_type", "audio_mode",
            "audio_denoise_strength", "add_source_as_reference",
            "prompt_primary_audio_ordinal", "strict_prompt_tags", "ref_image_size",
            "reference_video_policy", "allow_above_reference_area",
        ],
        15: ["audio_policy", "second_pass_audio_source", "second_pass_audio_strength"],
        16: [
            "shift_video", "shift_audio", "enable_tail", "extra_tail_steps",
            "tail_spacing", "enable_model_time_bias", "bias", "bias_start_progress",
            "bias_end_progress", "bias_domain", "enable_stg", "stg_scale",
            "stg_double_blocks", "stg_start_progress", "stg_end_progress",
            "enable_restart", "restart_video_sigma", "restart_steps", "restart_seed",
        ],
        17: [],
        18: ["noise_seed"],
        19: [
            "temporal_strategy", "temporal_chunk_frames", "temporal_overlap_frames",
            "anchor_strength", "audio_output",
        ],
        20: [],
    }
    graph = {}
    for node_id, names in widget_names.items():
        node = nodes[node_id]
        values = node.get("widgets_values", [])
        if len(values) < len(names):
            raise RuntimeError(
                f"saved widget contract changed for node {node_id}: {len(values)} < {len(names)}"
            )
        graph[str(node_id)] = {
            "class_type": node["type"],
            # RandomNoise stores the UI-only control_after_generate value as a
            # trailing widget; the API input remains noise_seed only.
            "inputs": dict(zip(names, values[: len(names)], strict=True)),
        }
    for _link, source, source_slot, target, target_slot, _kind in workflow["links"]:
        if str(target) not in graph:
            continue
        name = nodes[target]["inputs"][target_slot]["name"]
        graph[str(target)]["inputs"][name] = [str(source), source_slot]

    # Fixed short qualification: enough for multiple guarded windows plus a tail,
    # while remaining below the already-qualified 0.4 MP high-resolution budget.
    for node_id in ("7", "14"):
        graph[node_id]["inputs"].update(prompt=PROMPT, length=73)
    graph["7"]["inputs"].update(width=416, height=224)
    graph["13"]["inputs"].update(
        size_mode="scale_by",
        scale_by=2.0,
        target_width=832,
        target_height=448,
        target_megapixels=0.37,
        release_policy="offload_after",
    )
    graph["19"]["inputs"].update(
        temporal_strategy="guarded_overlap_exp",
        temporal_chunk_frames=34,
        temporal_overlap_frames=17,
        anchor_strength=0.999,
        audio_output="refined_exp",
    )
    # Use only Core video nodes in the isolated process. This does not alter the
    # saved public workflow, whose VHS output remains compatible for end users.
    graph["21"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["20", 0], "audio": ["20", 1], "fps": 24.0},
    }
    graph["22"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["21", 0],
            "filename_prefix": "MiniMaxH3/H16_3_Chunked_PASS2_GPU_v1",
            "format": "mp4",
            "codec": "h264",
        },
    }
    graph["23"] = {"class_type": "PreviewAny", "inputs": {"source": ["13", 3]}}
    graph["24"] = {"class_type": "PreviewAny", "inputs": {"source": ["15", 2]}}
    graph["25"] = {"class_type": "PreviewAny", "inputs": {"source": ["16", 5]}}
    return graph


def pcm_evidence(path: Path) -> dict:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
            "stream=sample_rate,channels,duration", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("H16-3 qualification expected exactly one audio stream")
    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-i",
            str(path), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    if len(decoded) < 4 or len(decoded) % 4:
        raise RuntimeError("H16-3 decoded audio is empty or malformed")
    import numpy as np

    pcm = np.frombuffer(decoded, dtype="<f4")
    if not np.isfinite(pcm).all() or float(np.max(np.abs(pcm))) <= 1e-6:
        raise RuntimeError("H16-3 decoded audio is silent or non-finite")
    return {
        **streams[0],
        "decoded_f32le_bytes": len(decoded),
        "decoded_pcm_sha256": hashlib.sha256(decoded).hexdigest(),
        "rms": float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64)))),
        "peak": float(np.max(np.abs(pcm))),
        "finite": True,
        "non_silent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8856)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a new artifact directory inside this project")
    graph = build_graph(project)
    transport.CORE = args.core.resolve()
    transport.PROJECT = project
    root.mkdir(parents=True)
    expected = {
        "core": verify_core_source(args.core),
        "sources": transport.source_snapshot(),
        "controller": file_identity(Path(__file__)),
        "workflow": file_identity(
            project
            / "examples/workflows/13-latent-upscale"
            / "2026-09-20_H3_H16_3_Chunked_PASS2_I2VA_Advanced_EXP.json"
        ),
        "mode": "gpu",
        "pilot_graphs": {"h16_3": graph},
        "qualification_scope": "73_frames_416x224_to_832x448_guarded34_overlap17_refined_exp",
    }
    transport.write_json(root / "paths.json", probe_resource_config(args.core, project))
    transport.write_json(root / "expected.json", expected)
    result = {
        "status": "incomplete",
        "human_quality_accepted": False,
        "seed": 2608193401,
        "expected_frames": 73,
    }
    guard = ResourceGuard()
    server = transport.OwnedServer(root, args.port, False, 2)
    monitor = None
    lease = (
        args.core
        / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    )
    try:
        with SerialProbeLease(lease), NvmlResourceReader() as reader, ExitStack() as cleanup:
            reason = guard.observe(reader.sample(), startup=True)
            if reason:
                raise RuntimeError("Startup resource guard: " + reason)
            server.start()
            monitor = transport.ContinuousGuard(
                reader, guard, root / "resources.jsonl", server
            )
            cleanup.callback(monitor.close)
            monitor.start()
            transport.wait_ready(server, monitor.check)
            object_info = server.request("GET", "/object_info")
            for node_id in {node["class_type"] for node in graph.values()}:
                if node_id not in object_info:
                    raise RuntimeError("Missing node registration: " + node_id)
            transport.write_json(root / "object-info.json", object_info)
            started = time.perf_counter()
            history, timing = transport.execute_graph(
                server, graph, root / "generation", monitor.check, timeout=2400
            )
            executed = {row["node"] for row in timing["node_intervals"]}
            if "19" not in executed:
                raise RuntimeError("Production DeciiaChunkedPass2Sampler did not execute")
            media = audit_media(root)
            if (media["width"], media["height"], media["frames"], media["fps"]) != (
                832,
                448,
                73,
                "24/1",
            ):
                raise RuntimeError("Unexpected H16-3 delivery geometry/timeline: " + repr(media))
            audio = pcm_evidence(Path(media["path"]))
            logs = "\n".join(
                path.read_text(encoding="utf8", errors="replace")
                for path in (root / "server.stdout.log", root / "server.stderr.log")
            )
            marker = "H16-3 refined audio merge complete:"
            if marker not in logs or "H16-3 refined audio merge failed" in logs:
                raise RuntimeError("Refined audio merge success marker missing or fallback occurred")
            reports = {
                "learned_upscale": transport.preview_report(history, "23"),
                "reconcile": transport.preview_report(history, "24"),
                "detail_mixer": transport.preview_report(history, "25"),
            }
            if transport.source_snapshot() != expected["sources"]:
                raise RuntimeError("Runtime source changed during H16-3 inference")
            result.update(
                status="mechanical_pass_human_pending",
                wall_seconds=time.perf_counter() - started,
                production_node_executed=True,
                refined_audio_merge="success_no_fallback",
                media=media,
                audio=audio,
                reports=reports,
                timing=timing,
            )
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
