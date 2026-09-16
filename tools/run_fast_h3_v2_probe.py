"""Source-bound, isolated FastH3 V2 smoke/media probe; never controls user UI."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from run_h3_memory_node_probe import audit_media, build_graph as memory_graph  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    GuardPolicy, MIB, NvmlResourceReader, ResourceGuard, SerialProbeLease, file_identity,
)
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402

REVISION = "0de92ab26fcb74ee47596332d93e55d20cddfd45"
MODEL = "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors"
MODEL_BYTES = 22128378696
MODEL_SHA256 = "0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8"
PROFILES = ("trained_vsa_exp", "dense_compat_exp", "official_comfy_template_exp")


def model_identity(core):
    from safetensors import safe_open

    path = core / "models/diffusion_models" / MODEL
    if path.stat().st_size != MODEL_BYTES:
        raise ValueError("V2 checkpoint incomplete or wrong export; do not load it")
    identity = file_identity(path)
    if identity["sha256"] != MODEL_SHA256:
        raise ValueError("V2 checkpoint SHA256 differs from pinned official export")
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        gates = [key for key in keys if ".to_gate_compress." in key]
        blocks = {int(match.group(1)) for key in gates
                  if (match := re.search(r"(?:blocks|transformer_blocks)\.(\d+)\.", key))}
        if blocks != set(range(50)):
            raise ValueError("Pinned V2 must contain learned gates for all50 blocks")
        identity.update(tensor_keys=len(keys), gate_keys=len(gates), gate_blocks=sorted(blocks),
                        metadata=reader.metadata(), repository="FastVideo/FastVideo-FastH3-Comfy",
                        revision=REVISION, header_only_structure_audit=True)
    return identity


def build_graph(*, profile="trained_vsa_exp", head_chunks=1, ffn_chunks=1,
                backend="pytorch", frames=73, width=832, height=480, min_tokens=0,
                instrument=True, first_frame=None):
    if profile not in PROFILES or backend not in ("pytorch", "kj_sage", "sol"):
        raise ValueError("Only named, source-bound V2 recipes/backends are supported")
    if not 32 <= width <= 8192 or not 32 <= height <= 8192 or width % 32 or height % 32 or frames not in (73, 124):
        raise ValueError("Probe sizes must be multiples32 and frames73/124")
    graph, reports = memory_graph(head_chunks=head_chunks, ffn_chunks=ffn_chunks)
    graph.pop("2")  # The full student is NOT an EMA/Turbo LoRA on the old model.
    graph["1"]["inputs"]["unet_name"] = MODEL
    if backend == "pytorch":
        graph.pop("3")
        source = "1"
    else:
        graph["3"]["inputs"]["model"] = ["1", 0]
        if backend == 'sol':
            graph['3'] = {'class_type': 'SolAttentionPatch', 'inputs': {
                'model': ['1', 0], 'enabled': True, 'tau': .5, 'min_tokens': 256,
                'strict': True, 'thresh_type': 'diag', 'int8_qk': False, 'int8_pv': False}}
        source = "3"
    for node in ("4", "5"):
        if node in graph:
            graph[node]["inputs"]["model"] = [source, 0]
            source = node
    graph["9"]["inputs"].update(width=width, height=height, length=frames)
    graph["10"] = {"class_type": "MiniMaxH3FastH3V2SetupEXPT8", "inputs": {
        "model": [source, 0], "av_latent": ["9", 1], "profile": profile,
        "min_tokens": min_tokens,
    }}
    graph["19"] = {"class_type": "MiniMaxH3FastH3V2RuntimeAuditEXPT8", "inputs": {
        "model": ["10", 0], "sampled_av_latent": ["13", 0],
    }}
    graph["14"]["inputs"]["av_latent"] = ["19", 0]
    graph["20"] = {"class_type": "PreviewAny", "inputs": {"source": ["19", 1]}}
    graph["21"] = {"class_type": "PreviewAny", "inputs": {"source": ["10", 3]}}
    graph["16"]["inputs"]["filename_prefix"] = "MiniMaxH3/FastH3_V2_Probe"
    reports.update(runtime="20", recipe="21")
    if first_frame is not None:
        relative = Path(first_frame)
        if relative.is_absolute() or '..' in relative.parts or not first_frame.strip():
            raise ValueError("First frame must be a Core input-relative filename")
        graph["23"] = {"class_type": "LoadImage", "inputs": {"image": relative.as_posix()}}
        graph["9"]["inputs"].update(task_type="I2VA", first_frame=["23", 0])
    if instrument:
        graph.pop("11")
        graph.pop("12")
        graph["13"] = {"class_type": "T8FastH3V2SamplerProbe", "inputs": {
            "model": ["10", 0], "positive": ["9", 0], "av_latent": ["9", 1],
            "sampler": ["10", 1], "sigmas": ["10", 2], "seed": 2609032101}}
        graph["22"] = {"class_type": "PreviewAny", "inputs": {"source": ["13", 1]}}
        reports["network"] = "22"
    return graph, reports


def validate_media_contract(media, *, width, height, frames):
    if (media["width"], media["height"], media["frames"]) != (width, height, frames):
        raise RuntimeError("V2 media dimensions/frame count differ from the requested canvas")
    if Fraction(media["fps"]) != 24 or not media["strict_decode"]:
        raise RuntimeError("V2 media must strictly decode at24fps")
    expected = frames / 24.
    if abs(float(media["video_seconds"]) - expected) > 1/24.:
        raise RuntimeError("V2 video duration differs from its frame count")
    if abs(float(media["audio_seconds"]) - expected) > .1:
        raise RuntimeError("V2 audio and video durations diverge")


def audit_v2_media(root, *, width, height, frames):
    media = audit_media(root)  # All three video/audio/combined strict decode paths.
    validate_media_contract(media, width=width, height=height, frames=frames)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", media["path"]],
                           capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != 1 or video[0]["codec_name"] != "h264":
        raise RuntimeError("V2 delivery must contain one H264 video and one audio track")
    return {**media, "video_codec": "h264", "audio_codec": audio[0]["codec_name"],
            "canvas_and_av_duration_verified": True, "quality_accepted": False}


def verify_final_inputs(expected, core):
    current = transport.source_snapshot()
    current['tools/run_fast_h3_v2_probe.py'] = file_identity(Path(__file__))['sha256']
    current['tools/fast_h3_v2_backend_audit.py'] = file_identity(Path(__file__).with_name('fast_h3_v2_backend_audit.py'))['sha256']
    if current != expected['sources'] or verify_core_source(core) != expected['core']:
        raise RuntimeError("Probe implementation changed during execution")
    for asset in expected.get('assets', []):
        if file_identity(Path(asset['path']))['sha256'] != asset['sha256']:
            raise RuntimeError("Probe model/reference asset changed during execution")


def isolated_backend_command(command, backend, headroom_gib=0):
    if headroom_gib not in (0, 2):
        raise ValueError('Only documented isolated-server headroom0/2 is supported')
    result = list(command)
    if backend in ('kj_sage', 'sol'):
        plugin = 'ComfyUI-KJNodes' if backend == 'kj_sage' else 'ComfyUI-sol-attn'
        result.insert(result.index('--whitelist-custom-nodes') + 1, plugin)
    elif backend != 'pytorch':
        raise ValueError('Unknown probe backend')
    # Isolate the MODEL selector. Do not silently change Qwen/VAE/global Core
    # attention to automatic Sage merely because a custom MODEL node is tested.
    if '--use-pytorch-cross-attention' not in result:
        result.append('--use-pytorch-cross-attention')
    if headroom_gib:
        if '--vram-headroom' in result:
            raise ValueError('Do not silently override existing server headroom')
        result.extend(['--vram-headroom', str(headroom_gib)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cpu", action="store_true", help="Registration/environment only; no model loading")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    parser.add_argument("--backend", choices=("pytorch", "kj_sage", "sol"), default="pytorch")
    parser.add_argument("--head-chunks", type=int, choices=(1, 4), default=1)
    parser.add_argument("--ffn-chunks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--frames", type=int, choices=(73, 124), default=73)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--headroom-gib", type=int, choices=(0, 2), default=0,
                        help="Native DynamicVRAM headroom for this owned test server only")
    parser.add_argument("--first-frame", help="Core input-relative file; matching aspect ratio required")
    args = parser.parse_args()
    project, core, root = Path(__file__).resolve().parents[1], args.core.resolve(), args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts"):
        raise ValueError("Use a new artifact directory in this development worktree")
    graph, reports = build_graph(profile=args.profile, backend=args.backend, head_chunks=args.head_chunks,
                                 ffn_chunks=args.ffn_chunks, frames=args.frames, width=args.width,
                                 height=args.height, first_frame=args.first_frame)
    transport.CORE, transport.PROJECT = core, project
    original = transport.server_command

    def command(*values):
        return isolated_backend_command(original(*values), args.backend, args.headroom_gib)

    transport.server_command = command
    root.mkdir(parents=True)
    expected = {"core": verify_core_source(core), "sources": transport.source_snapshot(),
                "mode": "cpu-smoke" if args.cpu else "gpu",
                "pilot_graphs": {} if args.cpu else {"fasth3_v2": graph}}
    expected['sources']['tools/run_fast_h3_v2_probe.py'] = file_identity(Path(__file__))['sha256']
    expected['sources']['tools/fast_h3_v2_backend_audit.py'] = file_identity(Path(__file__).with_name('fast_h3_v2_backend_audit.py'))['sha256']
    transport.write_json(root / "paths.json", probe_resource_config(core, project))
    result = {"status": "incomplete", "profile": args.profile, "backend": args.backend,
              "head_chunks": args.head_chunks, "ffn_chunks": args.ffn_chunks,
              "runtime_options": {"reserve_vram_gib": 5, "headroom_gib": args.headroom_gib},
              "small_probe_min_tokens": 0, "trained_default_min_tokens_changed": False}
    server, monitor = transport.OwnedServer(root, args.port, args.cpu), None
    # No fabricated12GB startup requirement. Retain sustained/critical safety
    # observation after previous host crashes; this is not a node admission gate.
    guard = ResourceGuard(GuardPolicy(startup_free_gpu_bytes=512 * MIB,
                                      startup_free_ram_bytes=4096 * MIB))
    lease = core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
    try:
        if not args.cpu:
            result["checkpoint"] = model_identity(core)
            expected['assets'] = [result['checkpoint']]
        if args.first_frame:
            from PIL import Image
            image_path = (core / 'input' / args.first_frame).resolve(strict=True)
            if not image_path.is_relative_to((core / 'input').resolve()):
                raise ValueError("Reference leaves Core input scope")
            with Image.open(image_path) as image:
                if image.width * args.height != image.height * args.width:
                    raise ValueError("Reference aspect differs from the canvas; refusing distortion")
            expected.setdefault('assets', []).append(file_identity(image_path))
        transport.write_json(root / "expected.json", expected)
        with ExitStack() as stack:
            if not args.cpu:
                stack.enter_context(SerialProbeLease(lease))
                reader = stack.enter_context(NvmlResourceReader())
                reason = guard.observe(reader.sample(), startup=True)
                if reason:
                    raise RuntimeError("Measured resource protection: " + reason)
            server.start()
            if not args.cpu:
                monitor = transport.ContinuousGuard(reader, guard, root / "resources.jsonl", server)
                stack.callback(monitor.close)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            transport.wait_ready(server, check)
            info = server.request("GET", "/object_info")
            for node in ("MiniMaxH3FastH3V2SetupEXPT8", "MiniMaxH3FastH3V2RuntimeAuditEXPT8"):
                if node not in info:
                    raise RuntimeError("V2 node missing from live registration: " + node)
            transport.write_json(root / "object-info.json", info)
            history, _ = transport.execute_graph(server, {
                "90": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(expected)}},
                "91": {"class_type": "PreviewAny", "inputs": {"source": ["90", 0]}},
            }, root / "environment", check)
            result["environment"] = transport.preview_report(history, "91")
            if args.cpu:
                result["status"] = "live_registration_pass_no_inference"
            else:
                started = time.perf_counter()
                history, timing = transport.execute_graph(server, graph, root / "generation", check, timeout=1200)
                runtime = transport.preview_report(history, reports["runtime"])
                if args.profile != "dense_compat_exp" and not runtime["actual_vsa_dispatched"]:
                    raise RuntimeError("No VSA dispatched; Dense output is not VSA qualification")
                result.update(status="mechanical_media_pass_human_pending", wall_seconds=time.perf_counter()-started,
                              timing=timing, reports={name: transport.preview_report(history, node)
                                                     for name, node in reports.items()},
                              media=audit_v2_media(root, width=args.width, height=args.height, frames=args.frames))
            verify_final_inputs(expected, core)
            result['source_and_asset_inputs_unchanged'] = True
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if monitor:
            monitor.close()
        server.stop()
        result.update(server_stop=server.stop_receipt, resources=guard.report() if not args.cpu else "CUDA disabled")
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"status": result["status"], "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
