"""CPU encode verified RGB and copy the provenance-bound original audio to both routes."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


def source_media_identity(document, entry):
    """Accept the two explicitly known historical/current audit formats."""
    if document.get("status") == "mechanical_media_pass_human_quality_unverified":
        if not document.get("strict_av_decode"):
            raise ValueError("Source media did not pass strict decoding")
        item = document["file"]
        return {"path": item["path"], "file_sha256": item["sha256"]}
    if document.get("schema") == "h3_latent_resizer_split/v1":
        item = document["media"][entry]
        if not item.get("strict_decode_passed"):
            raise ValueError("Historical source did not pass strict decoding")
        return item
    raise ValueError("Unknown media provenance format")


def inspect_video(path):
    import av

    with av.open(str(path), options={"err_detect": "explode"}) as container:
        if len(container.streams.video) != 1:
            raise ValueError("One video stream required")
        stream = container.streams.video[0]
        if not stream.average_rate:
            raise ValueError("Missing exact video FPS")
        fps = Fraction(stream.average_rate)
        if fps <= 0 or fps > 120:
            raise ValueError("Unsupported review FPS")
        count, origin = 0, None
        stamps = []
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded frame missing timestamp")
            if (frame.width, frame.height) != (stream.width, stream.height):
                raise ValueError("Geometry changes during decode")
            stamp = Fraction(frame.pts) * frame.time_base
            if origin is None:
                origin = stamp
            if stamp != origin + Fraction(count) / fps:
                raise ValueError("Video is not exact CFR")
            stamps.append(str(stamp))
            count += 1
        if not count or (stream.frames and stream.frames != count):
            raise ValueError("Incomplete video decode")
        duration = Fraction(count) / fps
        if stream.duration is None or stream.time_base * stream.duration != duration:
            raise ValueError("Declared and actual duration disagree")
        return {"width": stream.width, "height": stream.height, "frames": count,
                "fps": str(fps), "origin": str(origin), "duration": str(duration), "pts": stamps,
                "colors": {key: int(getattr(stream.codec_context, key)) for key in
                           ("color_primaries", "color_trc", "colorspace", "color_range")}}


def encode_rgb(path, rgb_path, expected, source):
    import av
    import torch
    from safetensors import safe_open

    path = Path(path)
    with safe_open(rgb_path, framework="pt", device="cpu") as tensors:
        rgb = tensors.get_slice("rgb")
        if tuple(rgb.get_shape()) != tuple(expected):
            raise ValueError("RGB source geometry mismatch")
        fps, origin = Fraction(source["fps"]), Fraction(source["origin"])
        if origin != 0:
            raise ValueError("Initial review route requires zero-origin source")
        clock = 1 / fps
        with path.open("xb") as file, av.open(file, "w", format="mp4", options={"movflags": "faststart"}) as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = expected[4], expected[3], "yuv420p"
            stream.time_base = clock
            stream.codec_context.time_base = clock
            stream.codec_context.thread_count = 2
            stream.codec_context.max_b_frames = 0
            stream.options = {"crf": "18", "preset": "medium", "threads": "2"}
            for key, value in source["colors"].items():
                setattr(stream.codec_context, key, value)
            for index in range(expected[2]):
                value = rgb[0, :, index]
                if not bool(torch.isfinite(value).all()) or value.min() < 0 or value.max() > 1:
                    raise ValueError("Invalid normalized RGB")
                pixels = value.permute(1, 2, 0).mul(255).round().to(torch.uint8).contiguous().numpy()
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = index, clock
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-entry", default="draft")
    parser.add_argument("--saved-reference", action="store_true", help="New scoped candidate plus previous provenance-bound native review video")
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    audit_path = root / ("independent-saved-reference-audit.json" if args.saved_reference else "independent-video-audit.json")
    audit = json.loads(audit_path.read_text())
    expected_status = ("complete_short_scoped_decode_against_bound_native_not_human_qualified" if args.saved_reference
                       else "complete_rgb_evidence_verified_not_human_or_speed_qualification")
    if audit["status"] != expected_status:
        raise ValueError("Independent RGB audit required")
    for name, digest in audit["files"].items():
        if Path(name).name != name or digest_file(root / name) != digest:
            raise ValueError("Audited artifact changed")
    source_report = args.source_report.resolve(strict=True)
    provenance_digest = digest_file(source_report)
    provenance = source_media_identity(json.loads(source_report.read_text()), args.source_entry)
    source = Path(provenance["path"]).resolve(strict=True)
    if digest_file(source) != provenance["file_sha256"].lower():
        raise ValueError("Original media provenance hash mismatch")
    import torch
    from dlss_nr_advanced import (
        _audio_packet_digests, _audio_pcm_digests, _packet_copy_video_and_audio, _validate_audio_identity,
    )
    torch.set_num_threads(2)
    original = inspect_video(source)
    native_review = None
    if args.saved_reference:
        request = json.loads((root / "request.json").read_text())
        from tools.trt_vae_saved_reference import bind_reference
        if bind_reference(request["reference_rgb"], request["latent"], request["native_vae"]) != audit["reference_sources"]:
            raise ValueError("Native reference binding changed")
        previous = json.loads((Path(request["reference_rgb"]).parent / "media-audit.json").read_text())
        native_review = previous["routes"]["native"]
        if (previous["status"] != "complete_video_and_original_audio_verified_not_human_qualification"
                or Path(previous["source"]).resolve() != source or previous["source_sha256"] != digest_file(source)
                or digest_file(native_review["path"]) != native_review["sha256"]):
            raise ValueError("Previous native review provenance differs")
    shape = (1,3,73,512,1024) if args.saved_reference else tuple(audit["output_shape"])
    if (shape[0], shape[1], shape[2], shape[3], shape[4]) != (1, 3, original["frames"], original["height"], original["width"]):
        raise ValueError("Original media does not match full latent output geometry")
    packets, pcm = _audio_packet_digests(source), _audio_pcm_digests(source)
    if not packets or not pcm:
        raise ValueError("This audio-bound test requires actual original audio")
    target = root / "media"
    target.mkdir(exist_ok=False)
    results = {}
    for route in ("native", "trt"):
        video_only = target / (route + "-video.mp4")
        final = target / (route + ".mp4")
        if final.exists():
            raise FileExistsError(final)
        if route == "native" and native_review is not None:
            shutil.copyfile(native_review["path"], final)
            if digest_file(final) != native_review["sha256"]:
                raise ValueError("Copied native review identity changed")
        else:
            encode_rgb(video_only, root / (route + "-rgb.safetensors"), shape, original)
            _packet_copy_video_and_audio(video_only, source, final)
        actual = inspect_video(final)
        if actual != original:
            raise ValueError("Final geometry, full PTS, color tags or duration differ from original")
        got_packets, got_pcm = _audio_packet_digests(final), _audio_pcm_digests(final)
        audio = _validate_audio_identity(packets, got_packets, pcm, got_pcm)
        results[route] = {"path": str(final), "sha256": digest_file(final), "video": actual,
                          "audio": audio, "audio_packets": got_packets, "audio_pcm": got_pcm}
    if digest_file(source) != provenance["file_sha256"].lower() or digest_file(source_report) != provenance_digest:
        raise ValueError("Original media/provenance changed while preparing review")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU media preparation initialized CUDA")
    report = {"status": "complete_video_and_original_audio_verified_not_human_qualification",
              "source_report": str(source_report), "source_report_sha256": provenance_digest,
              "source": str(source), "source_sha256": provenance["file_sha256"].lower(), "routes": results,
              "rgb_audit_sha256": digest_file(audit_path), "native_review_reused": native_review is not None,
              "preparation_source_sha256": digest_file(__file__),
              "audio_implementation_sha256": digest_file(PROJECT / "dlss_nr_advanced.py"),
              "cuda_initialized": False, "encoder": "Identical CPU libx264 CRF18 yuv420p, no sharpening/color correction",
              "limitation": "Lossy review videos; numerical metrics apply to pre-encode float RGB, not MP4 pixels. This source only, not other routes/content or full-plan/human qualification."}
    write_new_json(root / "media-audit.json", report)
    print(json.dumps({"status": report["status"], "routes": {key: {"path": row["path"], "audio": row["audio"]}
                                                            for key, row in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
