"""Read-only video packet comparison in a separate no-Torch/PyAV process.

This checks packet bytes and metadata, not decodability or visual quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av


def byte_difference_summary(left, right):
    """Bounded binary evidence; differing bytes alone do not identify a cause."""
    count = bits = offset = 0
    first = []
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            x, y = a.read(1024**2), b.read(1024**2)
            if not x and not y:
                break
            if x != y:
                for index, (u, v) in enumerate(zip(x, y)):
                    if u != v:
                        count += 1
                        bits += (u ^ v).bit_count()
                        if len(first) < 10:
                            first.append({"offset": offset + index, "left": u, "right": v, "xor": u ^ v})
            offset += max(len(x), len(y))
    return {"different_shared_bytes": count, "different_shared_bits": bits,
            "length_difference": Path(left).stat().st_size - Path(right).stat().st_size,
            "first_differences": first}


def inspect(path):
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected exactly one video stream")
        stream = container.streams.video[0]
        info = {"codec": stream.codec_context.name, "width": stream.width, "height": stream.height,
                "extradata_sha256": hashlib.sha256(stream.codec_context.extradata or b"").hexdigest(),
                "time_base": str(stream.time_base)}
        packets = []
        for packet in container.demux(stream):
            if packet.dts is not None:
                packets.append({"sha256": hashlib.sha256(bytes(packet)).hexdigest(), "bytes": packet.size,
                    "pts": packet.pts, "dts": packet.dts, "duration": packet.duration,
                    "time_base": str(packet.time_base), "keyframe": packet.is_keyframe})
        return {"stream": info, "packets": packets}


def strict_decode(path, *, threads=1, thread_type="SLICE"):
    count = 0
    error = None
    options_left = None
    actual_threads = None
    with av.logging.Capture(local=True) as logs:
        try:
            with av.open(str(path), mode="r") as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = threads
                stream.codec_context.thread_type = thread_type
                stream.codec_context.options = {"err_detect": "explode"}
                for frame in container.decode(stream):
                    count += 1
                    options_left = dict(stream.codec_context.options)
                    actual_threads = stream.codec_context.thread_count
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    errors = [message for level, _, message in logs if level <= av.logging.ERROR]
    return {"av_version": av.__version__, "library_versions": av.library_versions,
            "decoded_frames": count, "error": error, "error_logs": errors[:5], "remaining_options": options_left,
            "requested_threads": threads, "thread_type": thread_type, "actual_threads": actual_threads,
            "strict_decode_passed": error is None and not errors and count > 0 and "err_detect" not in (options_left or {}),
            "scope": "independent_decoder_diagnostic_not_generation_acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path, nargs="?")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--thread-type", choices=("AUTO", "FRAME", "SLICE"), default="SLICE")
    args = parser.parse_args()
    if args.decode:
        if args.threads < 0:
            parser.error("threads cannot be negative")
        print(json.dumps(strict_decode(args.left, threads=args.threads, thread_type=args.thread_type), indent=2))
        raise SystemExit(0)
    if args.right is None:
        parser.error("right input is required for packet comparison")
    left, right = inspect(args.left), inspect(args.right)
    differences = [{"index": i, "left": a, "right": b}
                   for i, (a, b) in enumerate(zip(left["packets"], right["packets"])) if a != b]
    print(json.dumps({"av_version": av.__version__, "left_stream": left["stream"], "right_stream": right["stream"],
        "left_packets": len(left["packets"]), "right_packets": len(right["packets"]),
        "stream_equal": left["stream"] == right["stream"], "packet_records_equal": left["packets"] == right["packets"],
        "difference_count": len(differences), "first_differences": differences[:5],
        "file_byte_comparison": byte_difference_summary(args.left, args.right),
        "scope": "packet_only_not_decode_or_perceptual_acceptance"}, indent=2))
