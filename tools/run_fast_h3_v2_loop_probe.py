"""Owned FastH3 V2 two-window EXP probe; CPU validates, GPU is explicit.

No downloads, frontend operations, unrelated process termination or automatic
retry. The reference and learned-upscale4+4 loop are NOT distilled support.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_progressive_pilot as transport  # noqa: E402
from run_fast_h3_v2_probe import MODEL, model_identity  # noqa: E402
from run_dual_model_pilot import (  # noqa: E402
    BUND_KOREAN_MV_GLOBAL_PROMPT, BUND_KOREAN_MV_LOCAL_PROMPTS,
)
from tools.build_dual_model_workflows import defaults  # noqa: E402
from progressive_probe_control import (  # noqa: E402
    GuardPolicy, MIB, NvmlResourceReader, ResourceGuard, SerialProbeLease,
    file_identity,
)
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402

NODE = "MiniMaxH3FastH3V2DualModelLongVideoEXPT8"
PLAN = "MiniMaxH3PromptRelayPlanT8Advanced"
LORA = "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
LOWVRAM = "MiniMaxH3LowVRAMAttentionT8Advanced"
FFN = "MiniMaxH3ChunkFeedForwardT8Advanced"
PROFILES = ("dense_compat_exp", "trained_vsa_exp")
REFERENCE = Path("F:/ComfyUI_00110_iphsk_1789210737 (1).png")
MODEL_ASSETS = (("diffusion_models", MODEL),
    ("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    ("vae", "minimax_h3_video_vae_fp16.safetensors"),
    ("vae", "minimax_h3_audio_vae_fp32.safetensors"),
    ("latent_upscale_models", "minimax_h3_latent_upscaler_3d_fp16.safetensors"))
SOURCE_HELPERS = ("run_fast_h3_v2_loop_probe.py", "run_fast_h3_v2_probe.py",
    "run_dual_model_pilot.py", "run_h3_memory_node_probe.py", "build_dual_model_workflows.py",
    "api_to_frontend_workflow.py", "frontend_workflow_compat.py")


def build_graph(info, *, first_frame, chain_id, profile="dense_compat_exp",
                lora1=None, lora2=None, strength1=0., strength2=0.):
    if profile not in PROFILES or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", chain_id):
        raise ValueError("Unknown V2 profile or unsafe chain_id")
    reference = Path(first_frame)
    if (not first_frame or reference.is_absolute() or reference.drive
            or any(part in ("", ".", "..") for part in reference.parts)
            or ":" in first_frame or "\\" in first_frame):
        raise ValueError("Reference must be Core input-relative")
    required = {NODE, LOWVRAM, FFN, "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "PreviewAny"}
    if profile == "dense_compat_exp":
        required.add(PLAN)
    if required - info.keys():
        raise RuntimeError("Missing actual Core registrations: " + ", ".join(sorted(required-info.keys())))
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODEL_ASSETS[1][1], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_ASSETS[2][1]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_ASSETS[3][1]}},
        "23": {"class_type": "LoadImage", "inputs": {"image": first_frame}},
    }
    for base, lora, strength, lora_node, attention_node, ffn_node in (
            ("1", lora1, strength1, "30", "21", "24"),
            ("2", lora2, strength2, "31", "22", "25")):
        if not math.isfinite(strength) or not -20 <= strength <= 20:
            raise ValueError("LoRA strength must be finite within -20..20")
        source = base
        # A disabled optional slot must not try to open a nonexistent LoRA.
        if strength != 0:
            if not lora or LORA not in info:
                raise ValueError("An active independent LoRA requires a filename and registered loader")
            graph[lora_node] = {"class_type": LORA, "inputs": {"model": [base, 0],
                "lora_name": lora, "strength_model": strength}}
            source = lora_node
        graph[attention_node] = {"class_type": LOWVRAM,
            "inputs": {"model": [source, 0], "head_chunks": 4}}
        graph[ffn_node] = {"class_type": FFN,
            "inputs": {"model": [attention_node, 0], "chunks": 2, "seq_threshold": 4096}}
    inputs = {**defaults(info[NODE]), "profile": profile,
        "model_pass1": ["24", 0], "model_pass2": ["25", 0],
        "clip": ["4", 0], "video_vae": ["5", 0], "audio_vae": ["6", 0],
        "chain_id": chain_id, "total_duration_seconds": 8.,
        "low_width": 256, "low_height": 384, "width": 512, "height": 768,
        "render_window_frames": 124, "context_frames": 22,
        "upscaler_model": MODEL_ASSETS[4][1], "first_frame": ["23", 0],
        "task_type": "I2VA", "first_frame_reuse": "segment0_only",
        "audio_mode": "native", "audio_denoise_strength": 1.,
        "add_source_as_reference": False, "context_audio": "video_and_audio",
        "eav_mode": "disabled", "segment_prompts_json": "",
        "prompt_relay_mode": "apply_exp" if profile == "dense_compat_exp" else "disabled",
        "global_prompt": "" if profile == "dense_compat_exp" else (
            BUND_KOREAN_MV_GLOBAL_PROMPT + " She sings softly throughout this continuous eight-second excerpt, with no additional dialogue."),
        "second_audio_source": "auto", "second_audio_strength": 0.,
        "low_context_source": "accepted_picture_low_context_v1",
        "video_context_mode": "high_native_mask_ramp_exp",
        "color_match": True, "color_match_mode": "bounded_motion_color_exp",
        "resume_existing": True, "minimum_free_vram_mib": 512,
        "base_seed": 2609152201, "seed_policy": "increment",
        "filename_prefix": "FastH3V2_Loop_8s_4plus4_EXP",
        "audio_seam_policy": "cosine_bridge", "bridge_ms": 5., "bit_depth": 8, "crf": 18}
    if profile == "dense_compat_exp":
        graph["7"] = {"class_type": PLAN, "inputs": {**defaults(info[PLAN]),
            "global_prompt": BUND_KOREAN_MV_GLOBAL_PROMPT,
            "local_prompts": BUND_KOREAN_MV_LOCAL_PROMPTS, "length": 193,
            "timing_mode": "percent", "time_ranges": "0-43.75\n43.75-87.5\n87.5-100"}}
        inputs["prompt_relay_plan"] = ["7", 0]
    graph["8"] = {"class_type": NODE, "inputs": inputs}
    graph["50"] = {"class_type": "PreviewAny", "inputs": {"source": ["8", 5]}}
    return graph


def pending_materials(info, graph):
    """Diagnostic only; Core validate_prompt remains the authoritative gate."""
    missing = []
    for node_id, node in graph.items():
        schema = info[node["class_type"]]["input"]
        specs = {**schema.get("required", {}), **schema.get("optional", {})}
        for name, value in node["inputs"].items():
            spec = specs.get(name)
            if spec and isinstance(spec[0], list) and not isinstance(value, list) and value not in spec[0]:
                missing.append({"node": node_id, "input": name, "value": value})
    return missing


def material_validation_failed(folder, error):
    """Do not relabel a crash, ownership error or source mismatch as materials."""
    marker = "pilot graph failed Core validation"
    if marker in str(error):
        return True
    history_path = folder / "history.json"
    if not history_path.is_file():
        return False
    history = json.loads(history_path.read_text(encoding="utf-8"))
    messages = history.get("status", {}).get("messages", [])
    return any(isinstance(message, list) and len(message) == 2
        and message[0] == "execution_error" and isinstance(message[1], dict)
        and marker in message[1].get("exception_message", "") for message in messages)


def import_reference(core, source):
    from PIL import Image
    original = file_identity(source)
    with Image.open(source) as image:
        width, height = image.size
        if width * 3 != height * 2:
            raise ValueError("Reference must have exact2:3 aspect ratio; it will not be stretched")
    directory = (core / "input").resolve(strict=True)
    filename = "t8_fasth3_v2_loop_" + original["sha256"][:24] + Path(source).suffix.lower()
    destination = directory / filename
    if destination.exists():
        copied = file_identity(destination)
        if copied["sha256"] != original["sha256"] or copied["bytes"] != original["bytes"]:
            raise RuntimeError("Existing task reference differs; refusing overwrite")
    else:
        # An exclusive, byte-identical task-scoped import; never edits original.
        with Path(source).open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 8*MIB)
        copied = file_identity(destination)
        if copied["sha256"] != original["sha256"]:
            raise RuntimeError("Reference import changed bytes")
    if file_identity(source) != original:
        raise RuntimeError("Reference changed while importing")
    return filename, original, copied


def frozen_sources(project):
    sources = transport.source_snapshot()
    for name in SOURCE_HELPERS:
        sources["tools/" + name] = file_identity(project / "tools" / name)["sha256"]
    return sources


def validate_loop_report(report, profile):
    if report.get("status") != "complete" or report.get("segment_count") != 2 or report.get("accepted_count") != 2:
        raise RuntimeError("V2 loop did not complete exactly two accepted segments")
    audits = report.get("segment_audits", [])
    if len(audits) != 2 or [audit.get("segment_index") for audit in audits] != [0, 1]:
        raise RuntimeError("Missing or reordered actual segment audits")
    stages, vsa_calls = [], 0
    for audit in audits:
        unsigned = dict(audit)
        claimed = unsigned.pop("audit_sha256", None)
        digest = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != claimed or audit.get("contract_sha256") != report.get("contract_sha256"):
            raise RuntimeError("Segment audit hash or job contract differs")
        dual = audit["sampling_plan"]["dual_model"]
        for stage_name, start, end in (("first_pass", 0, 4), ("second_pass", 4, 8)):
            stage = dual[stage_name]
            if stage.get("nfe") != 4 or stage.get("completed_network_forwards") != 4:
                raise RuntimeError("Actual stage is not four completed network forwards")
            schedule = stage.get("schedule", {})
            if (schedule.get("profile") != profile or schedule.get("stage_start") != start
                    or schedule.get("stage_end") != end or len(stage.get("sigmas", [])) != 5
                    or schedule.get("video_shift") != 10. or schedule.get("audio_shift") != 3.
                    or schedule.get("rungs") != [999, 874, 749, 624, 500, 375, 250, 125]):
                raise RuntimeError("Actual stage profile/ladder interval differs")
            rungs = [999, 874, 749, 624, 500, 375, 250, 125, 0]
            expected_sigmas = [10*(r/1000)/(1+9*(r/1000)) for r in rungs[start:end+1]]
            if any(not math.isfinite(sigma) or abs(sigma-expected) > 1e-6
                   for sigma, expected in zip(stage["sigmas"], expected_sigmas, strict=True)):
                raise RuntimeError("Actual shifted DMD ladder differs")
            runtime = stage["backend"].get("fasth3_v2", {})
            if runtime.get("profile") != profile or runtime.get("head_chunks") != 4:
                raise RuntimeError("Actual V2 owner or head grouping differs")
            counts = runtime.get("counts", {})
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise RuntimeError("Invalid dispatch counts")
            actual = counts.get("vsa", 0)
            if runtime.get("actual_vsa_dispatched") is not (actual > 0):
                raise RuntimeError("VSA dispatch claim differs from actual counts")
            if profile == "dense_compat_exp" and actual:
                raise RuntimeError("Explicit dense path unexpectedly dispatched sparse attention")
            vsa_calls += actual
            memory = stage.get("memory_composition") or {}
            if memory.get("kind") != "t8_h3_memory" or memory.get("head_chunks") != 4 or memory.get("ffn_settings") != [2, 4096]:
                raise RuntimeError("Actual T8 memory composition differs from h4/c2")
            relay = stage.get("prompt_relay_execution")
            if profile == "dense_compat_exp":
                if not relay or relay.get("completed_forwards") != 4 or relay.get("routed_attention_calls", 0) <= 0:
                    raise RuntimeError("Relay global-time routing did not execute")
            elif relay is not None:
                raise RuntimeError("Trained VSA EXP must not silently combine Relay")
            stages.append({"segment_index": audit["segment_index"], "stage": stage_name,
                "completed_network_forwards": 4, "profile": profile, "dispatch": deepcopy(runtime),
                "memory": deepcopy(memory), "relay": deepcopy(relay)})
    if profile == "trained_vsa_exp" and vsa_calls == 0:
        raise RuntimeError("Trained VSA probe never actually dispatched VSA")
    return {"actual_network_forwards": 16, "stages": stages, "vsa_calls": vsa_calls,
        "scope": "mechanical contract, not trained reference support or human quality acceptance"}


def audit_media(path, evidence):
    path = Path(path).resolve(strict=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
        "-show_format", "-of", "json", str(path)], capture_output=True, text=True)
    transport.write_json(evidence / "ffprobe-terminal.json", {"returncode": probe.returncode,
        "stdout": probe.stdout, "stderr": probe.stderr})
    if probe.returncode or probe.stderr.strip():
        raise RuntimeError("ffprobe failed; see persisted terminal receipt")
    transport.write_json(evidence / "ffprobe.json", json.loads(probe.stdout))
    payload = json.loads(probe.stdout)
    videos = [s for s in payload["streams"] if s["codec_type"] == "video"]
    audios = [s for s in payload["streams"] if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError("Expected exactly one video and one audio track")
    video, audio = videos[0], audios[0]
    if (video["codec_name"] != "h264" or video["width"] != 512 or video["height"] != 768
            or int(video["nb_read_frames"]) != 192 or Fraction(video["avg_frame_rate"]) != 24):
        raise RuntimeError("Delivery is not exact H264512x768/192frames/24fps")
    for name, stream in (("video", video), ("audio", audio), ("container", payload["format"])):
        duration = float(stream["duration"])
        if not math.isfinite(duration) or abs(duration - 8.) > 1/24:
            raise RuntimeError(name + " duration differs from eight-second AV contract")
    for name, maps in (("video", ["-map", "0:v:0"]), ("audio", ["-map", "0:a:0"]),
                       ("av", ["-map", "0:v:0", "-map", "0:a:0"])):
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
            "-i", str(path), *maps, "-f", "null", "-"], capture_output=True, text=True)
        transport.write_json(evidence / ("decode-" + name + ".json"),
            {"returncode": decoded.returncode, "stdout": decoded.stdout, "stderr": decoded.stderr})
        if decoded.returncode or decoded.stderr.strip():
            raise RuntimeError("Strict " + name + " decode failed")
    return {**file_identity(path), "width": 512, "height": 768, "frames": 192,
        "fps": "24/1", "video_codec": "h264", "audio_codec": audio["codec_name"],
        "strict_decode": True, "quality_accepted": False}


class _Tee:
    def __init__(self, original, saved):
        self.original, self.saved = original, saved
    def write(self, value):
        self.saved.write(value)
        self.saved.flush()
        return self.original.write(value)
    def flush(self):
        self.saved.flush()
        self.original.flush()


def run(args, project, core, root):
    transport.CORE, transport.PROJECT = core, project
    source = frozen_sources(project)
    result = {"status": "incomplete", "profile": args.profile, "human_review": "pending",
        "reference_and_loop": "EXP_not_distilled_support", "stop_after_first": "not_supported_by_actual_runner",
        "interrupt_qualification": "not_tested_by_this_controller", "root": str(root)}
    server, monitor = transport.OwnedServer(root, args.port, args.cpu), None
    guard = ResourceGuard(GuardPolicy(startup_free_gpu_bytes=512*MIB, startup_free_ram_bytes=4096*MIB))
    original_command = transport.server_command
    resume_expected, output = None, root / "output"
    try:
        if args.resume_from:
            if args.cpu:
                raise ValueError("Cross-process resume is an explicit GPU cache audit")
            previous = args.resume_from.resolve(strict=True)
            if previous == root or not previous.is_relative_to(project / "artifacts"):
                raise ValueError("Resume evidence must be another exact owned artifact directory")
            resume_expected = json.loads((previous / "expected.json").read_text(encoding="utf-8"))
            previous_terminal = json.loads((previous / "terminal.json").read_text(encoding="utf-8"))
            if previous_terminal.get("status") != "mechanical_8s_AV_stage_contract_pass_human_review_pending":
                raise ValueError("This resume probe certifies only a previously completed mechanical8s run")
            output = Path(previous_terminal["output_directory"]).resolve(strict=True)
            if not output.is_relative_to(previous) or not output.is_dir():
                raise ValueError("Resume output leaves verified previous controller scope")
        result["output_directory"] = str(output)
        def command(*values):
            cmd = original_command(*values)
            cmd.append("--use-pytorch-cross-attention")
            cmd[cmd.index("--output-directory")+1] = str(output)
            return cmd
        transport.server_command = command
        filename, original_image, copied_image = import_reference(core, args.reference)
        result["reference"] = original_image
        chain_id = args.chain_id or (resume_expected["pilot_graphs"]["v2_loop"]["8"]["inputs"]["chain_id"]
            if resume_expected else "v2_loop_" + hashlib.sha256(root.name.encode()).hexdigest()[:20])
        expected = {"core": verify_core_source(core), "sources": source,
            "mode": "cpu-smoke" if args.cpu else "gpu", "pilot_graphs": {}, "assets": [copied_image],
            "runtime_options": {"reserve_vram_gib": 5, "headroom_gib": 0}}
        if not args.cpu:
            result["checkpoint"] = model_identity(core)
            expected["assets"].append(result["checkpoint"])
            for category, name in MODEL_ASSETS[1:]:
                expected["assets"].append(file_identity(core / "models" / category / name))
            for name, strength in ((args.lora1, args.strength1), (args.lora2, args.strength2)):
                if strength != 0:
                    if not name:
                        raise ValueError("Active LoRA needs filename")
                    path = (core / "models/loras" / name).resolve(strict=True)
                    if not path.is_relative_to(core / "models/loras"):
                        raise ValueError("LoRA leaves model scope")
                    expected["assets"].append(file_identity(path))
        transport.write_json(root / "paths.json", probe_resource_config(core, project))
        transport.write_json(root / "initial-expected.json", expected)
        lease = core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
        with ExitStack() as cleanup:
            if not args.cpu:
                cleanup.enter_context(SerialProbeLease(lease))
                reader = cleanup.enter_context(NvmlResourceReader())
                reason = guard.observe(reader.sample(), startup=True)
                if reason:
                    raise RuntimeError("Startup resource observation: " + reason)
            server.start()
            if not args.cpu:
                monitor = transport.ContinuousGuard(reader, guard, root / "resources.jsonl", server)
                cleanup.callback(monitor.close)
                monitor.start()
            check = monitor.check if monitor else lambda: None
            transport.wait_ready(server, check)
            info = server.request("GET", "/object_info")
            transport.write_json(root / "object-info.json", info)
            graph = build_graph(info, first_frame=filename, chain_id=chain_id, profile=args.profile,
                lora1=args.lora1, lora2=args.lora2, strength1=args.strength1, strength2=args.strength2)
            def environment(payload, name):
                history, _ = transport.execute_graph(server, {
                    "90": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(payload)}},
                    "91": {"class_type": "PreviewAny", "inputs": {"source": ["90", 0]}}}, root / name, check)
                return transport.preview_report(history, "91")
            result["registration"] = environment(expected, "registration")
            expected["pilot_graphs"] = {"v2_loop": graph}
            transport.write_json(root / "expected.json", expected)
            if resume_expected and resume_expected != expected:
                raise RuntimeError("Resume graph/Core/source/assets differ; use a new chain")
            missing = pending_materials(info, graph)
            try:
                result["graph_validation"] = environment(expected, "graph-validation")
            except RuntimeError as error:
                if not args.cpu or not missing or not material_validation_failed(root / "graph-validation", error):
                    raise
                result.update(status="live_registration_pass_graph_pending_material_no_inference",
                    pending_material=missing, graph_validated=False, graph_validation_error=str(error))
            else:
                result["graph_validated"] = True
                if args.cpu:
                    result["status"] = "live_Core_registration_and_complete_graph_validation_pass_no_inference"
                else:
                    started = time.perf_counter()
                    history, timing = transport.execute_graph(server, graph, root / "generation", check, timeout=args.timeout)
                    report = transport.preview_report(history, "50")
                    transport.write_json(root / "loop-report.json", report)
                    result.update(wall_seconds=time.perf_counter()-started, timing=timing,
                        stage_audit=validate_loop_report(report, args.profile))
                    path = Path(report["final_video_path"]).resolve(strict=True)
                    if not path.is_relative_to(output):
                        raise RuntimeError("Final video leaves controller output scope")
                    result["media"] = audit_media(path, root)
                    if report.get("final_video_sha256") != result["media"]["sha256"]:
                        raise RuntimeError("Final report media SHA differs")
                    if resume_expected:
                        if report.get("resume_action") != "returned_verified_existing_final":
                            raise RuntimeError("Cross-process complete-cache return was not verified")
                        result["resume"] = {"action": report["resume_action"],
                            "new_network_forwards": 0, "stage_counts_scope": "verified original persisted execution",
                            "partial_interruption": "not_qualified"}
                    result["status"] = "mechanical_8s_AV_stage_contract_pass_human_review_pending"
            if frozen_sources(project) != source:
                raise RuntimeError("Runtime sources changed during owned run")
            if file_identity(args.reference) != original_image:
                raise RuntimeError("Original reference changed during run")
            for asset in expected["assets"]:
                identity = file_identity(Path(asset["path"]))
                if any(identity[key] != asset[key] for key in ("path", "bytes", "mtime_ns", "sha256")):
                    raise RuntimeError("Frozen asset changed during run")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        for action in ([monitor.close] if monitor else []) + [server.stop]:
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        transport.server_command = original_command
        result.update(server_stop=server.stop_receipt, resources=guard.report(), cleanup_errors=cleanup_errors)
        if cleanup_errors:
            result["status"] = "failed_owned_cleanup"
        transport.write_json(root / "terminal.json", result)
        print(json.dumps({"status": result["status"], "root": str(root)}), flush=True)
        if cleanup_errors:
            raise RuntimeError("Owned cleanup failed: " + "; ".join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu", action="store_true")
    modes.add_argument("--gpu", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default=PROFILES[0])
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--chain-id")
    parser.add_argument("--resume-from", type=Path, help="New evidence, exact prior output; no interrupted-run qualification")
    parser.add_argument("--lora1")
    parser.add_argument("--lora2")
    parser.add_argument("--strength1", type=float, default=0.)
    parser.add_argument("--strength2", type=float, default=0.)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    project, core, root = Path(__file__).resolve().parents[1], args.core.resolve(strict=True), args.root.resolve()
    if root.exists() or not root.is_relative_to(project / "artifacts") or not 1 <= args.port <= 65535 or args.timeout <= 0:
        raise ValueError("Use a new artifact directory, valid port and positive timeout")
    root.mkdir(parents=True)
    with (root / "controller.stdout.log").open("x", encoding="utf-8") as stdout, (root / "controller.stderr.log").open("x", encoding="utf-8") as stderr:
        with redirect_stdout(_Tee(sys.stdout, stdout)), redirect_stderr(_Tee(sys.stderr, stderr)):
            try:
                run(args, project, core, root)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
