"""Run one serial DLSS-NR 2x post-process on a completed H3 outpaint video.

This is an independent post-processing gate: it loads no H3 diffusion, CLIP or
VAE model.  The source file and its outpaint receipt are hash-bound before the
external runtime is audited and invoked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

try:
    from tools import run_dlss_nr_validation as validation
except ModuleNotFoundError:
    import run_dlss_nr_validation as validation


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "b837fc50dd1ade76353a68752d7599c89bf7106602ddf64d243846253e6c1755"


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex(("127.0.0.1", int(port))) == 0


def run(args):
    dlss = validation.dlss
    source_path = args.source.resolve(strict=True)
    source_path.relative_to(ROOT / "artifacts")
    source_hash = validation._sha256_file(source_path)
    expected = args.source_sha256.lower()
    if source_hash != expected:
        raise ValueError(f"completed outpaint source hash mismatch: {source_hash}")
    receipt_path = Path(f"{source_path}.outpaint.json")
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("schema") != "t8.h3.video_outpaint.final_file/v1" or receipt.get("sha256") != source_hash:
        raise ValueError("completed outpaint receipt does not bind the source video")
    if not all((
        receipt.get("pre_encode_pixel_evidence_verified") is True,
        receipt.get("selected_first_frame_verified_before_encoding") is True,
        receipt.get("padding_delivered") is False,
        receipt.get("audio_regenerated") is False,
    )):
        raise ValueError("completed outpaint receipt lacks required delivery evidence")
    output_root = args.output.resolve()
    checks = {
        "new_output_root": not output_root.exists(),
        "source_hash_bound": source_hash == expected,
        "outpaint_receipt_bound": True,
        "no_user_gpu_server": not _port_open(8188),
        "no_upstream_gpu_server": not _port_open(8190),
        "no_h3_model_requested": True,
        "external_license_accepted": args.accept_external_runtime_license is True,
    }
    preflight = {
        "schema": "t8.h3.video_outpaint.dlss_postprocess_probe/v1",
        "status": "preflight",
        "checks": checks,
        "source": str(source_path),
        "source_sha256": source_hash,
        "outpaint_receipt": str(receipt_path),
        "outpaint_receipt_sha256": validation._sha256_file(receipt_path),
        "runtime_version": args.runtime_version,
        "mode": "sr_nr",
        "scale": 2.0,
        "quality_profile": "standard",
        "sr_preset": "default",
        "motion_engine_requested": "auto",
        "h3_diffusion_loaded": False,
        "h3_sampling_called": False,
        "human_acceptance": False,
    }
    if not args.confirm_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return preflight
    if not all(checks.values()):
        raise RuntimeError(f"DLSS-after-outpaint preflight failed: {checks}")
    output_root.mkdir(parents=True, exist_ok=False)
    validation._write_json_atomic(output_root / "report.json", preflight)
    report = dict(preflight)
    started = time.perf_counter()
    try:
        ready, audit = dlss.audit_dlss_nr_runtime(
            dlss.runtime_root(args.models_dir, args.runtime_version),
            args.runtime_version,
            accept_external_runtime_license=args.accept_external_runtime_license,
            probe_mode="feature_probe_1_frame",
            dxgi_adapter_index=args.dxgi_adapter_index,
            cuda_device_index=args.cuda_device_index,
        )
        report["runtime_audit"] = audit
        validation._write_json_atomic(output_root / "runtime_audit.json", audit)
        if not ready:
            raise RuntimeError("DLSS-NR runtime did not pass the one-frame feature gate")
        runtime = dlss.runtime_handle_from_report(audit)
        if runtime is None:
            raise RuntimeError("READY runtime audit did not return a handle")
        revalidation = dlss.revalidate_runtime_handle(runtime)
        gpu_index = validation._nvidia_index(revalidation)
        source = validation.FileBackedVideo(source_path)
        output = output_root / "selected_outpaint_dlss_sr_nr_2x.mp4"

        def execute():
            return dlss.process_video_file(
                runtime,
                source,
                output_path=output,
                mode="sr_nr",
                scale=2.0,
                quality_profile="standard",
                sr_preset="default",
                motion_engine="auto",
                crf=18.0,
            )

        (published, returned_source, process), memory = validation._measure_vram(gpu_index, execute)
        if returned_source is not source or published != output.resolve():
            raise RuntimeError("DLSS post-process returned another source or output")
        video = process["video"]
        audio = process["audio"]
        validated = process["validation"]
        expected_dimensions = [source.width * 2, source.height * 2]
        if (video.get("input_frame_count") != source.frame_count
                or video.get("output_frame_count") != source.frame_count
                or video.get("output_dimensions") != expected_dimensions
                or video.get("single_persistent_dlss_process") is not True
                or video.get("full_image_batch_materialized") is not False):
            raise RuntimeError("DLSS post-process geometry/frame/stream contract failed")
        if not all(audio.get(name) is True for name in (
                "packet_payload_exact", "packet_timeline_exact", "decoded_pcm_exact",
                "decoded_timeline_exact")) or audio.get("reencoded") is not False:
            raise RuntimeError("DLSS post-process changed source audio")
        if (validated.get("decoded_video_frames") != source.frame_count
                or validated.get("strict_ffmpeg_decode") is not True
                or process.get("atomic_publish") is not True
                or process.get("source_overwritten") is not False
                or process.get("no_bilinear_rgb_fallback") is not True
                or process.get("motion_engine_resolved") not in {"nvof", "lk"}):
            raise RuntimeError("DLSS post-process publication validation failed")
        screen = validation._video_screen(source_path, published, hard_cut=False)
        if screen["black_regression_frames"] or screen["freeze_regression_frames"]:
            raise RuntimeError("DLSS post-process introduced mechanical black/freeze regression")
        report.update(
            status="real_dlss_after_outpaint_mechanical_pass_human_review_pending",
            runtime_revalidation=revalidation,
            process=process,
            memory=memory,
            screening=screen,
            output=str(published),
            output_sha256=validation._sha256_file(published),
            output_bytes=published.stat().st_size,
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        validation._write_json_atomic(output_root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output_root / "report.json"),
                      "output": report["output"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=ROOT.parents[1] / "models")
    parser.add_argument("--runtime-version", default="1.3")
    parser.add_argument("--dxgi-adapter-index", type=int, default=0)
    parser.add_argument("--cuda-device-index", type=int, default=0)
    parser.add_argument("--accept-external-runtime-license", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    run(parser.parse_args())
