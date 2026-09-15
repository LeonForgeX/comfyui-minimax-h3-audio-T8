"""Run one isolated, serial, three-second native H3 memory-node probe."""

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
)
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


def build_graph(*, head_chunks: int, ffn_chunks: int) -> tuple[dict, dict[str, str]]:
    if head_chunks not in (1, 4) or ffn_chunks not in (1, 2):
        raise ValueError("Probe supports only the declared h1/c1 and h4/c2 settings")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
            "weight_dtype": "default",
        }},
        "2": {"class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced", "inputs": {
            "model": ["1", 0],
            "lora_name": "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors",
            "strength_model": 1.0,
        }},
        "3": {"class_type": "PathchSageAttentionKJ", "inputs": {
            "model": ["2", 0], "sage_attention": "auto", "allow_compile": False,
        }},
        "6": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default",
        }},
        "7": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors",
        }},
        "8": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors",
        }},
        "9": {"class_type": "MiniMaxH3AudioConditioningT8", "inputs": {
            "prompt": (
                "A continuous cinematic medium shot of a woman in an old concert hall. "
                "She looks toward the camera and clearly says in Mandarin: <d>你在哪里</d>. "
                "Soft classical piano and strings, clean speech, natural mouth motion, no subtitles."
            ),
            "width": 1024, "height": 512, "length": 73,
            "task_type": "T2VA", "audio_mode": "native",
            "audio_denoise_strength": 1.0, "add_source_as_reference": False,
            "prompt_primary_audio_ordinal": 0, "strict_prompt_tags": True,
            "ref_image_size": "match", "reference_video_policy": "official_2_to_15s",
            "clip": ["6", 0], "video_vae": ["7", 0], "audio_vae": ["8", 0],
        }},
        "10": {"class_type": "MiniMaxH3DualClockSamplerT8", "inputs": {
            "steps": 4, "shift_video": 12.0, "shift_audio": 3.0,
            "model": ["3", 0], "av_latent": ["9", 1],
        }},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2609032101}},
        "12": {"class_type": "BasicGuider", "inputs": {
            "model": ["10", 0], "conditioning": ["9", 0],
        }},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["12", 0],
            "sampler": ["10", 1], "sigmas": ["10", 2], "latent_image": ["9", 1],
        }},
        "14": {"class_type": "MiniMaxH3AVDecodeT8", "inputs": {
            "av_latent": ["13", 0], "video_vae": ["7", 0], "audio_vae": ["8", 0],
        }},
        "15": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0], "fps": 24.0, "audio": ["14", 1],
        }},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0], "filename_prefix": "MiniMaxH3/H3_Memory_Probe",
            "format": "mp4", "codec": "h264",
        }},
    }
    source = "3"
    reports = {}
    if head_chunks != 1:
        graph["4"] = {"class_type": "MiniMaxH3LowVRAMAttentionT8Advanced", "inputs": {
            "model": [source, 0], "head_chunks": head_chunks,
        }}
        graph["17"] = {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}}
        reports["attention"] = "17"
        source = "4"
    if ffn_chunks != 1:
        graph["5"] = {"class_type": "MiniMaxH3ChunkFeedForwardT8Advanced", "inputs": {
            "model": [source, 0], "chunks": ffn_chunks, "seq_threshold": 4096,
        }}
        graph["18"] = {"class_type": "PreviewAny", "inputs": {"source": ["5", 1]}}
        reports["ffn"] = "18"
        source = "5"
    graph["10"]["inputs"]["model"] = [source, 0]
    return graph, reports


def audit_media(root: Path) -> dict:
    paths = list((root / "output").rglob("*.mp4"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one output MP4, found {len(paths)}")
    path = paths[0]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    payload = json.loads(probe.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in payload["streams"] if stream["codec_type"] == "audio")
    for selection in (["-map", "0:v:0"], ["-map", "0:a:0"], ["-map", "0:v:0", "-map", "0:a:0"]):
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(path),
             *selection, "-f", "null", "-"], capture_output=True, text=True,
        )
        if decoded.returncode or decoded.stderr.strip():
            raise RuntimeError("Strict media decode failed: " + decoded.stderr)
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": int(video["width"]), "height": int(video["height"]),
        "frames": int(video["nb_read_frames"]), "fps": video["avg_frame_rate"],
        "video_seconds": video.get("duration"), "audio_seconds": audio.get("duration"),
        "container_seconds": payload["format"].get("duration"),
        "strict_decode": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--head-chunks", type=int, choices=(1, 4), required=True)
    parser.add_argument("--ffn-chunks", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a new artifact directory inside this worktree")
    graph, report_nodes = build_graph(
        head_chunks=args.head_chunks, ffn_chunks=args.ffn_chunks,
    )
    transport.CORE = args.core.resolve()
    transport.PROJECT = project
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        result.insert(result.index("--whitelist-custom-nodes") + 1, "ComfyUI-KJNodes")
        return result

    transport.server_command = command
    root.mkdir(parents=True)
    expected = {
        "core": verify_core_source(args.core),
        "sources": transport.source_snapshot(),
        "mode": "gpu",
        "pilot_graphs": {"h3_memory": graph},
    }
    transport.write_json(root / "paths.json", probe_resource_config(args.core, project))
    transport.write_json(root / "expected.json", expected)
    result = {
        "status": "incomplete", "head_chunks": args.head_chunks,
        "ffn_chunks": args.ffn_chunks, "seed": 2609032101,
    }
    guard = ResourceGuard()
    server = transport.OwnedServer(root, args.port, False, 0)
    monitor = None
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
                "90": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {
                    "expected_json": json.dumps(expected),
                }},
                "91": {"class_type": "PreviewAny", "inputs": {"source": ["90", 0]}},
            }, root / "environment", monitor.check)
            result["environment"] = transport.preview_report(history, "91")
            object_info = server.request("GET", "/object_info")
            for node_id in (
                "MiniMaxH3LowVRAMAttentionT8Advanced",
                "MiniMaxH3ChunkFeedForwardT8Advanced",
            ):
                if node_id not in object_info:
                    raise RuntimeError("Missing T8 memory node registration: " + node_id)
            transport.write_json(root / "object-info.json", object_info)
            started = time.perf_counter()
            history, timing = transport.execute_graph(
                server, graph, root / "generation", monitor.check, timeout=1200,
            )
            result.update(
                status="mechanical_audit_pass_human_pending",
                wall_seconds=time.perf_counter() - started,
                timing=timing,
                reports={name: transport.preview_report(history, node)
                         for name, node in report_nodes.items()},
                media=audit_media(root),
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
