"""Read-only real-file media isolation probe; no sampling, GPU, or media rewrite."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Only this newly launched diagnostic process.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    name = "t8_media_isolation_probe"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    media = importlib.import_module(name + ".video_outpaint_media")
    delivery = importlib.import_module(name + ".video_outpaint_delivery")
    from comfy_api.input_impl import VideoFromFile
    baseline = delivery.read_delivery_report(args.video)
    args.run_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "t8.outpaint.media_isolation_probe/v1", "status": "running",
              "source": str(args.source.resolve()), "video": str(args.video.resolve()),
              "human_acceptance": False, "generation_or_hardware_fix_claimed": False}
    try:
        source = media.inspect_outpaint_source(VideoFromFile(str(args.source.resolve())))
        if source["sha256"] != baseline["source_sha256"]:
            raise ValueError("diagnostic source differs from the delivery source")
        expected = baseline["plan"]["output"]
        parameters = dict(frame_count=expected["frames"], width=expected["width"], height=expected["height"],
                          rate=expected["fps"], source_audio_packets=source["audio_packets"], source_audio_pcm=source["audio_pcm"])
        report["source_inspection"] = source
        report["video_sha256_before"] = media._sha256_file(args.video)
        report["media"] = media._validate_final_file(args.video, **parameters)
        report["same_media_evidence_as_previous_internal_validator"] = report["media"] == baseline["media"]
        report["video_sha256_after"] = media._sha256_file(args.video)
        if report["video_sha256_before"] != report["video_sha256_after"] or not report["same_media_evidence_as_previous_internal_validator"]:
            raise ValueError("media identity/evidence changed across process isolation")
        if args.bad_video:
            # This explicit corrupt-video-only control tests decoder rejection,
            # independently from whether its audio matches a different source.
            negative = {**parameters, "source_audio_packets": [], "source_audio_pcm": []}
            try:
                media._validate_final_file(args.bad_video, **negative)
            except RuntimeError as error:
                detail = str(error)
                if "strict Skin Finish H.264 validation" not in detail and "InvalidDataError" not in detail:
                    raise RuntimeError("bad-video control failed for an unrelated reason: " + detail) from error
                report["bad_video_control"] = {"path": str(args.bad_video.resolve()),
                    "sha256": media._sha256_file(args.bad_video), "decode_rejected": True, "error": detail}
            else:
                raise RuntimeError("known corrupt video unexpectedly passed the isolated validator")
        report["status"] = "isolated_media_contract_pass_not_generation_acceptance"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (args.run_root / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({"status": report["status"], "report": str(args.run_root / "report.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bad-video", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    run(parser.parse_args())
