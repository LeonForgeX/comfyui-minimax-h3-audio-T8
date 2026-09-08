"""One serial, isolated real-model VDN compatibility/refinement probe."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_openvdn_h3_validation as base  # noqa: E402
from build_vdn_two_pass_workflows import build_prompt, build_frontend  # noqa: E402
import run_openvdn_h3_multimodal_validation as multimodal  # noqa: E402
from vdn_probe_environment import probe_resource_config, verify_core_source  # noqa: E402


class TimelineGpuMonitor(base.clipprobe.GpuPeakMonitor):
    """Keep samples so a resource failure can be located instead of guessed."""
    def __init__(self):
        super().__init__(interval_seconds=0.25)
        self.timeline = []
        self.prompt_start = None

    def _run(self):
        while not self._stop.is_set():
            sample = base.shared.gpu_memory_mib()
            if sample.get("available"):
                used, free = int(sample["used_mib"]), int(sample["free_mib"])
                self.peak_used_mib = used if self.peak_used_mib is None else max(self.peak_used_mib, used)
                self.minimum_free_mib = free if self.minimum_free_mib is None else min(self.minimum_free_mib, free)
                self.samples += 1
                self.timeline.append({"monotonic": time.monotonic(), "used_mib": used, "free_mib": free})
            self._stop.wait(self.interval_seconds)

    def stop(self):
        result = super().stop()
        result["timeline"] = [{"elapsed_seconds": round(s["monotonic"] - self.prompt_start, 4),
                               "used_mib": s["used_mib"], "free_mib": s["free_mib"]}
                              for s in self.timeline] if self.prompt_start is not None else []
        result["timing_scope"] = "relative to caller submission start; websocket event clock starts slightly later"
        return result


def decoded_audio_sha256(video, ffmpeg):
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-threads", "1", "-i", str(video), "-map", "0:a:0",
         "-c:a", "pcm_f32le", "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, check=True, timeout=120,
    )
    value = result.stdout.strip().removeprefix("SHA256=")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError("FFmpeg did not return a valid decoded-audio SHA256")
    return value


def finalize_resources(report, telemetry):
    report["gpu_monitor"] = telemetry
    minimum = telemetry.get("minimum_free_mib")
    report["resource_checks"] = {
        "telemetry_available": minimum is not None and telemetry.get("samples", 0) > 0,
        "minimum_free_vram_at_least_512_mib": minimum is not None and minimum >= 512,
    }
    if report.get("status") == "mechanical_pass_human_review_pending" and not all(report["resource_checks"].values()):
        report["status"] = "media_pass_resource_gate_failed"


def main(argv=None):
    parser = base._parser()
    parser.add_argument("--runtime-root", type=Path, help="Installed models and input assets, independently of --comfy-root source")
    parser.add_argument("--core-git-repository", type=Path)
    parser.add_argument("--core-source-revision")
    parser.add_argument("--use-sage-attention", action="store_true")
    parser.add_argument("--disable-dynamic-vram", action="store_true")
    parser.add_argument("--attention", choices=["default", "pytorch_before", "sparse_before", "sparse_after"], default="default")
    parser.add_argument("--two-pass", action="store_true")
    parser.add_argument("--safe-probe-save", action="store_true")
    parser.add_argument("--video-save", choices=["native", "isolated"], default="isolated")
    parser.add_argument("--scale-by", type=float, default=2.)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--refine-backend", choices=["vdn", "native_h3"], default="vdn")
    parser.add_argument("--native-refine-lora", default="minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors")
    parser.add_argument("--image")
    parser.add_argument("--variant", choices=["t2va", *multimodal.VARIANTS], default="t2va")
    parser.add_argument("--first-image", default=multimodal.DEFAULT_FIRST_IMAGE)
    parser.add_argument("--last-image", default=multimodal.DEFAULT_LAST_IMAGE)
    parser.add_argument("--ref-image-1", default=multimodal.DEFAULT_REF_IMAGE_1)
    parser.add_argument("--ref-image-2", default=multimodal.DEFAULT_REF_IMAGE_2)
    parser.add_argument("--ref-video", default=multimodal.DEFAULT_REF_VIDEO)
    args = parser.parse_args(argv)
    args.runtime_root = (args.runtime_root or args.comfy_root).resolve()
    args.input_directory = args.runtime_root / "input"
    provenance = verify_core_source(args.comfy_root, repository=args.core_git_repository,
                                    revision=args.core_source_revision)
    if args.variant != "t2va":
        args.image = args.image or multimodal.DEFAULT_IMAGE
        for name in multimodal._required_inputs(args):
            if not (args.input_directory / name).is_file():
                raise FileNotFoundError(f"required input asset missing: {name}")
    if args.recheck_report:
        raise ValueError("Use the existing report as evidence; this probe does not support --recheck-report")
    if args.refine_backend == "native_h3" and (not args.two_pass or args.refine_steps not in (3, 4, 5)):
        raise ValueError("native_h3 comparator requires --two-pass and 3/4/5 refine steps")
    if not base.shutil.which("ffmpeg") or not base.shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError("positive 32px-aligned input dimensions required")
    if args.frame_count < 5 or (args.frame_count - 5) % 17:
        raise ValueError("H3 frame count must follow 17n+5")
    if base.shared.port_is_listening(args.host, args.port):
        raise RuntimeError("probe port is occupied")
    memory = base.shared.gpu_memory_mib()
    if not memory.get("available") or memory.get("free_mib", 0) < args.min_free_vram_mib:
        raise RuntimeError(f"insufficient free GPU memory: {memory}")
    resource_args = copy.copy(args)
    resource_args.comfy_root = args.runtime_root
    for path in [args.comfy_root / "main.py", *base._asset_paths(resource_args)[1:]]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.two_pass:
        upscaler = args.runtime_root / "models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors"
        if not upscaler.is_file():
            raise FileNotFoundError(upscaler)
        if args.refine_backend == "native_h3":
            lora = args.runtime_root / "models/loras" / args.native_refine_lora
            if not lora.is_file():
                raise FileNotFoundError(lora)
    args.native_probe = args.refine_backend == "native_h3"
    args.extra_whitelist_custom_nodes = ("vdn_probe_extension",)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = args.artifact_root
    if not root.is_absolute():
        root = args.project_root / root
    root = root.resolve() / run_id
    root.mkdir(parents=True, exist_ok=False)
    args.extra_model_paths_config = root / "probe_paths.yaml"
    args.extra_model_paths_config.write_text(json.dumps(probe_resource_config(args.runtime_root, args.project_root),
                                                       indent=2), encoding="utf-8")
    graph = build_prompt(args, run_id)
    (root / "prompt.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    commit = provenance["core_commit"]
    source_hashes = {name: hashlib.sha256((args.project_root / name).read_bytes()).hexdigest() for name in (
        "vdn_h3_advanced.py", "h3_core_compat.py", "vdn_attention_compat.py", "vdn_sdpa_backend.py", "vdn_two_pass.py",
        "nodes_vdn_two_pass.py", "sampling.py", "multikeyframe_advanced.py",
        "tools/build_vdn_two_pass_workflows.py", "tools/run_vdn_core_compat_probe.py",
        "tools/vdn_probe_environment.py",
        "tools/vdn_probe_extension/__init__.py", "tools/vdn_probe_extension/media.py",
        "long_video_delivery.py", "h3_world_advanced.py", "h3_av_delivery.py", "nodes_h3_av_delivery.py")}
    report = {"core_commit": commit, "core_provenance": provenance, "runtime_root": str(args.runtime_root),
              "sources": source_hashes, "attention": args.attention,
              "sage_cli": args.use_sage_attention, "two_pass": args.two_pass, "preflight_gpu": memory,
              "disable_dynamic_vram": args.disable_dynamic_vram, "reserve_vram_gib": args.reserve_vram_gib,
              "refine_backend": args.refine_backend,
              "variant": args.variant,
              "safe_probe_save": args.safe_probe_save,
              "video_save": "probe_isolated" if args.safe_probe_save else args.video_save,
              "status": "started", "run_root": str(root)}
    (root / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    monitor = TimelineGpuMonitor()
    try:
        with base.shared.IsolatedServer(args, root, "vdn_core"):
            environment_prompt = {
                "1": {"class_type": "T8VDNCoreEnvironmentProbe", "inputs": {"expected_json": json.dumps(provenance)}},
                "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}},
            }
            environment_phase = asyncio.run(base.pdd._submit_prompt_capture(
                server=f"http://{args.host}:{args.port}", prompt=environment_prompt, timeout_seconds=120))
            report["environment_phase"] = environment_phase
            if environment_phase.get("terminal", {}).get("type") != "execution_success":
                raise RuntimeError("live Core provenance check failed before model generation")
            report["live_core"] = json.loads(base.pdd._phase_text(environment_phase, "2"))
            if report["live_core"].get("status") != "pass":
                raise RuntimeError("live Core module identity did not pass")
            # Survives a host/app interruption without masquerading as generation success.
            (root / "environment_check.json").write_text(json.dumps({
                "status": "environment_pass_only", "generation_accepted": False,
                "core_provenance": provenance, "live_core": report["live_core"],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.two_pass:
                with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
                    info = json.load(response)
                kinds = {node["class_type"] for node in graph.values()} | {"SaveVideo"}
                selected_info = {kind: info[kind] for kind in kinds}
                (root / "object_info.json").write_text(json.dumps(selected_info, ensure_ascii=False, indent=2), encoding="utf-8")
                workflow = build_frontend(graph, selected_info, refine_backend=args.refine_backend)
                (root / "workflow_EXP.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
            monitor.start()
            monitor.prompt_start = time.monotonic()
            phase = asyncio.run(base.pdd._submit_prompt_capture(
                server=f"http://{args.host}:{args.port}", prompt=graph, timeout_seconds=args.timeout_seconds))
        report["phase"] = phase
        if phase.get("terminal", {}).get("type") != "execution_success":
            raise RuntimeError("Comfy execution failed; inspect phase and server logs")
        if args.safe_probe_save or args.video_save == "isolated":
            report["safe_encoder"] = json.loads(base.pdd._phase_text(phase, "88"))
            if args.two_pass:
                report["first_safe_encoder"] = json.loads(base.pdd._phase_text(phase, "89"))
        video = next((root / "output" / "VDN_Core_Compat").glob("*.mp4"))
        media = base.stable_media_report(video, ffmpeg=base.shutil.which("ffmpeg"), ffprobe=base.shutil.which("ffprobe"))
        audio = base.pdd._audio_numeric(video, base.shutil.which("ffmpeg"))
        report.update(output_video=str(video), media=media, audio_numeric=audio)
        composition = json.loads(base.pdd._phase_text(phase, "14"))
        stderr = (root / "logs" / "vdn_core.stderr.log").read_text(encoding="utf-8", errors="replace")
        report["composition"] = composition
        report["adapter_checks"] = base.adapter_integrity_checks(composition, stderr)
        width, height = args.width, args.height
        if args.two_pass:
            report["refine"] = json.loads(base.pdd._phase_text(phase, "68"))
            if args.refine_backend == "native_h3":
                report["native_reference_schedule"] = report["refine"]
                report["refine"] = {"backend": "native_h3", "first_pass_nfe": 8,
                                    "refine_nfe": args.refine_steps, "total_nfe": 8 + args.refine_steps,
                                    "reference_coarse_sigmas_used": False,
                                    "refine_sigmas": report["native_reference_schedule"]["refine_video_sigmas"]}
                report["native_refine_lora"] = json.loads(base.pdd._phase_text(phase, "83"))
                report["native_branch_runtime"] = json.loads(base.pdd._phase_text(phase, "87"))
                if report["native_branch_runtime"].get("status") != "pass":
                    raise RuntimeError("native branch ownership audit failed")
                lora_audit = report["native_refine_lora"]
                if not lora_audit.get("applied_patch_count", 0) or lora_audit.get("missed_patch_target_count") != 0:
                    raise RuntimeError("native refinement LoRA did not apply completely")
            upscale = json.loads(base.pdd._phase_text(phase, "69"))
            report["upscale"] = upscale
            width, height = upscale["geometry"]["output_width"], upscale["geometry"]["output_height"]
            report["two_pass_audio_audit"] = json.loads(base.pdd._phase_text(phase, "74"))
            first_video = next((root / "output" / "VDN_Core_Compat_First_Pass").glob("*.mp4"))
            first_media = base.stable_media_report(first_video, ffmpeg=base.shutil.which("ffmpeg"), ffprobe=base.shutil.which("ffprobe"))
            first_audio = base.pdd._audio_numeric(first_video, base.shutil.which("ffmpeg"))
            report["first_pass"] = {"output_video": str(first_video), "media": first_media,
                                    "checks": base.pdd._media_checks(first_media, first_audio, width=args.width, height=args.height, frame_count=args.frame_count)}
            if not all(report["first_pass"]["checks"].values()):
                raise RuntimeError("first-pass media checks failed")
            first_pcm = decoded_audio_sha256(first_video, base.shutil.which("ffmpeg"))
            final_pcm = decoded_audio_sha256(video, base.shutil.which("ffmpeg"))
            report["audio_delivery_checks"] = {"decoded_pcm_identical": first_pcm == final_pcm,
                                               "first_pcm_sha256": first_pcm, "final_pcm_sha256": final_pcm}
            if first_pcm != final_pcm:
                raise RuntimeError("delivered second-pass audio differs from first-pass audio")
        report["media_checks"] = base.pdd._media_checks(media, audio, width=width, height=height, frame_count=args.frame_count)
        if not all(report["adapter_checks"].values()) or not all(report["media_checks"].values()):
            raise RuntimeError("adapter or media checks failed")
        report["status"] = "mechanical_pass_human_review_pending"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        finalize_resources(report, monitor.stop())
        report["core_sources_unchanged"] = all(
            hashlib.sha256((args.comfy_root / name).read_bytes()).hexdigest() == digest
            for name, digest in provenance["core_files_sha256"].items())
        report["sources_unchanged"] = all(
            hashlib.sha256((args.project_root / name).read_bytes()).hexdigest() == digest
            for name, digest in source_hashes.items())
        if report["status"] == "mechanical_pass_human_review_pending" and not (
            report["sources_unchanged"] and report["core_sources_unchanged"]
        ):
            report["status"] = "failed"
            report["error"] = "Runtime sources changed during probe; rerun from frozen sources"
        (root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: report.get(key) for key in ("status", "error", "run_root", "output_video")}
    summary["gpu_monitor"] = {key: value for key, value in report["gpu_monitor"].items() if key != "timeline"}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "mechanical_pass_human_review_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
