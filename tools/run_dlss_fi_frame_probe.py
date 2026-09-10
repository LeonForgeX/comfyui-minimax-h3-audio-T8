"""First actual DLSSG evaluation: three existing game frames, not an end-user video.

Strictly serial, no retries or source resampling. Default plan does not start a
worker. This establishes real frame requests and non-copy RGB, not visual/audio
qualification or a complete streaming file route.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss_fi_contract import TwoXTimeline, FrameLedger  # noqa: E402
from dlss_fi_transport import BinarySession, runtime_identity  # noqa: E402
from dlss_fi_guides import MotionGuide  # noqa: E402
from dlss_fi_device_binding import cuda_device_inventory, bind_probe  # noqa: E402
from run_progressive_pilot import RESEARCH, OwnedServer, ContinuousGuard, write_json  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402
from progressive_pilot_analysis import load_run, command_output  # noqa: E402


def rgb_identity(rgba, width, height):
    if not isinstance(rgba, bytes) or len(rgba) != width*height*4:
        raise ValueError("Invalid exact RGBA frame payload")
    pixels = np.frombuffer(rgba, np.uint8).reshape(height, width, 4)[..., :3]
    return hashlib.sha256(pixels.tobytes()).hexdigest(), pixels


def verify_decoder(binary, source):
    if file_identity(binary) != source["media"]["binaries"]["ffmpeg"]:
        raise ValueError("Decode executable must match the source's audited FFmpeg, not FFprobe or another binary")


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "gpu"), default="plan")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    source, identities = load_run(args.source_run), runtime_identity(args.runtime)
    verify_decoder(args.ffmpeg, source)
    if (source["terminal"].get("exploration_case") != "game_2609032101" or source["terminal"]["case"] != "T2VA_native8"):
        raise ValueError("First actual frame probe uses only the predeclared existing game native seed")
    if args.mode == "plan":
        print(json.dumps({"mode": "plan_no_GPU", "source": source["media"]["file"], "inputs": 3, "generated_expected": 2}))
        return
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError("New dedicated research run required")
    width, height, count = 1024, 512, 3
    with SerialProbeLease(RESEARCH / "serial-gpu.lock"):
        root.mkdir(parents=True)
        receipt = {"status": "incomplete", "audio_qualified": False, "quality_qualified": False}
        owner, monitor, session = OwnedServer(root, 0, False), None, None
        try:
            write_json(root / "source.json", {"source_media": source["media"], "runtime": identities,
                "ffmpeg": file_identity(args.ffmpeg), "selected_frames": [0, 1, 2],
                "tools": {name: file_identity(Path(__file__).with_name(name)) for name in
                    ("run_dlss_fi_frame_probe.py", "dlss_fi_transport.py", "dlss_fi_guides.py", "dlss_fi_device_binding.py")}})
            raw = command_output([args.ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-hwaccel", "none",
                "-i", source["media"]["file"]["path"], "-map", "0:v:0", "-an", "-frames:v", str(count),
                "-threads", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"])
            if len(raw) != width*height*4*count:
                raise ValueError("Source decode did not return exactly three original-size frames")
            frames = np.frombuffer(raw, np.uint8).reshape(count, height, width, 4)
            if any(np.array_equal(a[..., :3], b[..., :3]) for a, b in zip(frames, frames[1:])):
                raise ValueError("Selected diagnostic interval is static; cannot qualify non-copy motion")
            with NvmlResourceReader() as reader:
                guard = ResourceGuard()
                startup = reader.sample()
                guard.observe(startup, startup=True)
                write_json(root / "startup.json", startup)
                if guard.reason:
                    raise RuntimeError(guard.reason)
                inventory = cuda_device_inventory()
                write_json(root / "device-inventory.json", inventory)
                def started(process):
                    nonlocal monitor
                    owner.process = process
                    monitor = ContinuousGuard(reader, guard, root / "resources.jsonl", owner)
                    monitor.start()
                    print(f"Owned actual DLSSG frame worker PID {process.pid}", flush=True)
                try:
                    session = BinarySession([identities["dlssg-worker.exe"]["path"], "--serve"],
                        cwd=args.runtime.resolve(), width=width, height=height, frame_count=count, on_start=started)
                    plan = TwoXTimeline(count, Fraction(24), Fraction(0), frozenset())
                    ledger, slots, guide = FrameLedger(plan), iter(plan.slots()), MotionGuide(width, height)
                    hashes, rows, modules = [], [], set()
                    previous_pixels = None
                    for index, frame in enumerate(frames):
                        monitor.check()
                        field, reset = guide.process(frame)
                        color = frame.tobytes()
                        digest, pixels = rgb_identity(color, width, height)
                        response = session.frame(color, field.tobytes(), Fraction(index, 24), reset=reset)
                        row = {"input_index": index, "input_rgb_sha256": digest, "reset": reset,
                            "payload_frames": response["payload_frames"], "usable_generated_frames": response["usable_generated_frames"],
                            "motion_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
                            "motion_max_abs": float(np.abs(field).max())}
                        if index:
                            generated = response["rgba"]
                            generated_hash, generated_pixels = rgb_identity(generated, width, height)
                            if float(generated_pixels.std()) < 1 and min(float(pixels.std()), float(previous_pixels.std())) >= 1:
                                raise ValueError("Textured motion probe returned a constant/blank candidate")
                            ledger.observe(next(slots), payload_sha256=generated_hash, worker_generated=True, endpoint_hashes=(hashes[-1], digest))
                            with (root / f"generated-{index}.rgba").open("xb") as stream:
                                stream.write(generated)
                            row.update(generated_rgb_sha256=generated_hash,
                                mean_abs_previous=float(np.abs(generated_pixels.astype(np.int16)-previous_pixels).mean()),
                                mean_abs_current=float(np.abs(generated_pixels.astype(np.int16)-pixels).mean()),
                                generated_std=float(generated_pixels.std()))
                        ledger.observe(next(slots), payload_sha256=digest, endpoint_hashes=(digest,))
                        hashes.append(digest)
                        previous_pixels = pixels
                        rows.append(row)
                        write_json(root / f"frame-{index}.json", row)
                        modules.update(row.path for row in psutil.Process(session.process.pid).memory_maps(grouped=True) if "dlss" in row.path.lower())
                    ledger.observe(next(slots), payload_sha256=hashes[-1], endpoint_hashes=(hashes[-1],))
                    accounting = ledger.finish()
                    session.close()
                    logs = b"".join(session.logs).decode("utf-8", "replace")
                    write_json(root / "worker-log.json", {"stderr_tail": logs, "stop": session.stop_receipt})
                    binding = bind_probe({"supports_native_2x_by_report": True, "stderr_tail": logs}, inventory, startup["gpu_uuid"])
                    mapped = {Path(path).name: file_identity(path) for path in modules}
                    for name, identity in identities.items():
                        if name not in mapped or any(mapped[name][key] != identity[key] for key in ("path", "sha256", "bytes")):
                            raise ValueError("Frame worker did not map the fixed runtime files")
                    if runtime_identity(args.runtime) != identities or file_identity(source["media"]["file"]["path"]) != source["media"]["file"]:
                        raise ValueError("Runtime or source changed during actual frame evaluation")
                    write_json(root / "result.json", {"device_binding": binding, "mapped": mapped, "ledger": accounting, "frames": rows})
                    receipt.update(status="actual_two_intermediate_frames_noncopy_RGB_device_bound", generated_frames=2)
                finally:
                    if session:
                        session.close()
                    owner.stop()
                    if monitor:
                        monitor.close()
                        monitor.check()
                    receipt["resources"] = guard.report()
        except BaseException as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            if session and not (root / "worker-log.json").exists():
                write_json(root / "worker-log.json", {"stderr_tail": b"".join(session.logs).decode("utf-8", "replace"), "stop": session.stop_receipt})
            raise
        finally:
            owner.stop()
            receipt["server_stop"] = owner.stop_receipt
            write_json(root / "terminal.json", receipt)
            print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
