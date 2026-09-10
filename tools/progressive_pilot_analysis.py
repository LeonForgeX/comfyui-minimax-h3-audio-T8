"""CPU media postflight and evidence-bound pilot comparison; never generates video."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

try:
    from .progressive_probe_control import file_identity, summarize_execution_events
    from .progressive_memory_metrics import validate_completed_interval
    from .progressive_qualification import require_sage
except ImportError:
    from progressive_probe_control import file_identity, summarize_execution_events
    from progressive_memory_metrics import validate_completed_interval
    from progressive_qualification import require_sage


def command_output(command):
    result = subprocess.run(list(map(str, command)), stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name} exited {result.returncode}: {result.stderr.decode('utf-8', 'replace')[-3000:]}")
    return result.stdout


def validate_timeline(probe, frames, *, width, height, count, fps):
    """Only fixed, zero-origin CFR AV output is qualified by this first pilot."""
    streams = probe["streams"]
    video = [stream for stream in streams if stream["codec_type"] == "video"]
    audio = [stream for stream in streams if stream["codec_type"] == "audio"]
    if len(video) != 1 or len(audio) != 1 or len(streams) != 2:
        raise ValueError("Pilot requires exactly one video stream and one audio stream")
    video, audio = video[0], audio[0]
    rate = Fraction(fps)
    if (video["width"], video["height"], len(frames)) != (width, height, count):
        raise ValueError("Actual frame geometry/count differs from the fixed pilot")
    if Fraction(video["avg_frame_rate"]) != rate or Fraction(video["r_frame_rate"]) != rate:
        raise ValueError("Pilot video is not the expected CFR rate")
    time_base = Fraction(video["time_base"])
    if time_base <= 0:
        raise ValueError("Invalid video time base")
    for index, frame in enumerate(frames):
        if "pts" not in frame or abs(int(frame["pts"]) * time_base - Fraction(index, 1) / rate) > time_base / 2:
            raise ValueError("Video PTS gap, offset or duplicate detected")
        if (frame["width"], frame["height"]) != (width, height):
            raise ValueError("Video dimensions changed within the stream")
    duration = Fraction(count, 1) / rate
    sample_rate = int(audio["sample_rate"])
    channels = int(audio["channels"])
    if sample_rate <= 0 or not 1 <= channels <= 8:
        raise ValueError("Invalid audio format")
    # AAC frame rounding is allowed and reported, not used to ignore a missing tail.
    tolerance = Fraction(1, 1) / rate + Fraction(1024, sample_rate)
    if abs(Fraction(audio["start_time"])) > Fraction(1, sample_rate):
        raise ValueError("Audio stream does not start at timeline zero")
    audio_duration = Fraction(audio["duration"])
    if abs(audio_duration - duration) > tolerance:
        raise ValueError("Audio/video duration mismatch")
    return {"width": width, "height": height, "frames": count, "fps": str(rate),
            "video_duration_seconds": float(duration), "audio_duration_seconds": float(audio_duration),
            "duration_tolerance_seconds": float(tolerance), "sample_rate": sample_rate, "channels": channels,
            "video_codec": video["codec_name"], "audio_codec": audio["codec_name"], "all_video_pts_checked": True}


def audit_media(path, *, ffmpeg, ffprobe, width, height, count, fps=24):
    if not 0 < count <= 1024 or not 0 < width * height <= 4096**2:
        raise ValueError("Media probe outside the bounded pilot size")
    before = file_identity(path)
    binaries = {name: file_identity(binary) for name, binary in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))}
    probe = json.loads(command_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path]))
    if abs(float(probe["format"]["duration"]) - count / fps) > .1:
        raise ValueError("Container duration differs from the bounded pilot before full decoding")
    decoded_frames = json.loads(command_output([ffprobe, "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=pts,width,height", "-of", "json", path]))["frames"]
    timeline = validate_timeline(probe, decoded_frames, width=width, height=height, count=count, fps=fps)
    command_output([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1",
                    "-hwaccel", "none", "-i", path, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    raw = command_output([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", path,
                          "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    channels, sample_rate = timeline["channels"], timeline["sample_rate"]
    if not raw or len(raw) % (4 * channels):
        raise ValueError("Decoded audio PCM is empty or malformed")
    pcm = np.frombuffer(raw, dtype="<f4").reshape(-1, channels)
    if not bool(np.isfinite(pcm).all()):
        raise ValueError("Decoded audio PCM contains NaN/Inf")
    peak = float(np.abs(pcm).max())
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
    if peak <= 1e-7:
        raise ValueError("Classical-music/dialogue pilot decoded to silence")
    if abs(len(pcm)/sample_rate - timeline["video_duration_seconds"]) > timeline["duration_tolerance_seconds"]:
        raise ValueError("Decoded audio PCM length does not cover the expected clip")
    if file_identity(path) != before:
        raise RuntimeError("Media changed during postflight")
    return {"status": "mechanical_media_pass_human_quality_unverified", "file": before, "timeline": timeline,
            "strict_av_decode": True, "binaries": binaries,
            "decoded_audio": {"pcm_sha256": hashlib.sha256(raw).hexdigest(), "samples_per_channel": len(pcm),
                "peak": peak, "rms": rms, "rms_dbfs": 20*math.log10(rms), "values_above_full_scale": int((np.abs(pcm)>1).sum())},
            "warnings": ["Audio is very quiet; human listening required"] if rms < .001 else [],
            "limits": "Decode/timeline and signal statistics only. No noise, speech content, lip-sync or visual-quality acceptance."}


def common_graph(graph):
    graph = deepcopy(graph)
    for key in ("10", "13", "21"):
        graph.pop(key, None)
    graph["18"]["inputs"].pop("filename_prefix")
    return graph


def verify_execution_report(case, graph, reports):
    sampler = reports["sampler"]
    if graph.get('106',{}).get('class_type') == 'T8ProgressiveSageBackend':
        audit = sampler.get('attention_audit',{})
        if require_sage(audit.get('counts',{})) != audit:
            raise ValueError('Sage report does not bind actual successful calls')
    if case.endswith("native8"):
        if sampler.get("actual_network_forwards") != 8 or sampler.get("cfg") != 1 or sampler.get("seed") != graph["10"]["inputs"]["seed"]:
            raise ValueError("Native pilot actual network/CFG/seed evidence mismatch")
    elif case.endswith("progressive6plus2"):
        counts = sampler.get("counts", {})
        for name in ("callbacks", "actual_forwards", "apply_model_calls"):
            if counts.get(name) != {"low": 6, "high": 2}:
                raise ValueError("Progressive pilot did not execute the intended 6+2 network calls")
        if sampler.get("pixel_anchor") is not False or sampler.get("highres_tiling") is not False:
            raise ValueError("Pilot unexpectedly used pixel anchoring or tiling")
        if sampler["noise"]["low_seed"] != graph["10"]["inputs"]["seed"]:
            raise ValueError("Progressive pilot seed mismatch")
    else:
        raise ValueError("Unknown fixed pilot route")
    lora = reports["lora"]
    if (lora.get("status") != "applied" or lora.get("selected_strength") != 1 or
            lora.get("applied_patch_count") != 259 or lora.get("missed_patch_target_count") != 0 or
            lora.get("observed_model_patch_targets") != 259):
        raise ValueError("Actual EMA B patch evidence mismatch")


def load_allocator_evidence(root, terminal, expected, live):
    """Old runs remain valid without fabricated allocator metrics; new runs require them."""
    required = "tools/progressive_memory_metrics.py" in {key.replace("\\", "/") for key in expected["sources"]}
    if not required:
        return None
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    report = read("allocator-memory.json")
    end = read("allocator-finish/history.json")
    begin = read("allocator-begin/history.json")
    if (not begin["status"]["completed"] or not end["status"]["completed"] or
            json.loads(end["outputs"]["2"]["text"][0]) != report):
        raise ValueError("Allocator receipt differs from actual completed probe history")
    started = json.loads(begin["outputs"]["2"]["text"][0])
    validate_completed_interval(report, run_id=root.name, pid=live["pid"], device_type="cuda",
                                gpu_uuid=terminal["resource_guard"]["gpu_uuid"])
    if (started.get("run_id") != report["run_id"] or started.get("identity") != report["identity"] or
            started.get("status") != "active" or started.get("reset_count") != 1 or
            started.get("snapshots") != report["snapshots"][:1] or
            started.get("pre_interval_counters") != report["pre_interval_counters"]):
        raise ValueError("Allocator begin/end boundaries do not belong to the same interval")
    expected_terminal = {"file": "allocator-memory.json", "status": report["status"], "counter_scope": report["counter_scope"]}
    if terminal.get("allocator_interval") != expected_terminal:
        raise ValueError("Terminal evidence omits or changes the allocator interval")
    return report


def load_run(root):
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    terminal = read("terminal.json")
    if (terminal.get("mode") != "gpu" or terminal.get("status") != "generation_media_validated_unreviewed" or
            terminal.get("resource_monitor_failure") or terminal["resource_guard"]["status"] != "observations_within_policy" or
            terminal["server_stop"]["owned_children_remaining"]):
        raise ValueError("Run is not a successful owned GPU execution with media postflight")
    graph, timing, history = read("generation/prompt.json"), read("generation/timing.json"), read("generation/history.json")
    prompt_id = read("generation/submission.json")["prompt_id"]
    events = [json.loads(line) for line in (root / "generation/events.jsonl").read_text(encoding="utf-8").splitlines()]
    recomputed = summarize_execution_events(events, prompt_id, graph, timing["elapsed_seconds"])
    if recomputed != timing or not timing["complete_uncached_graph"] or not history["status"]["completed"]:
        raise ValueError("Actual events/history do not support uncached timing")
    reports = read("reports.json")
    for name, node in {"sampler": "21", "conditioning": "101", "decode": "102", "lora": "103", "save": "19"}.items():
        if json.loads(history["outputs"][node]["text"][0]) != reports[name]:
            raise ValueError("Collected report differs from the actual server history")
    verify_execution_report(terminal["case"], graph, reports)
    media = read("media-audit.json")
    output = Path(reports["save"]["output"]).resolve(strict=True)
    if not output.is_relative_to(root / "output") or file_identity(output) != media["file"]:
        raise ValueError("Saved media identity is stale or outside this run")
    if media["file"]["sha256"] != reports["save"]["output_sha256"] or not media["strict_av_decode"]:
        raise ValueError("Save result and media postflight disagree")
    expected, live = read("environment-expected.json"), read("live-environment.json")
    if expected.get('runtime_options') != live.get('runtime_options'):
        raise ValueError('Live runtime options differ from requested qualification')
    if live["status"] != "pass" or live["device"] == "cpu" or live["pid"] != terminal["server_stop"]["pid"]:
        raise ValueError("Live GPU process identity mismatch")
    if live["core"]["core_commit"] != expected["core"]["core_commit"]:
        raise ValueError("Live Core revision differs from the expected source")
    allocator = load_allocator_evidence(root, terminal, expected, live)
    return {"root": str(root), "terminal": terminal, "graph": graph, "timing": timing, "reports": reports,
            "media": media, "environment": expected, "live": live, "assets": read("assets-verified.json"),
            "allocator_memory": allocator}


def compare_runs(baseline, candidate):
    a, b = baseline, candidate
    if (not a["terminal"]["case"].endswith("_native8") or
            b["terminal"]["case"] != a["terminal"]["case"].replace("_native8", "_progressive6plus2")):
        raise ValueError("Compare the same task's native8 and progressive6plus2 routes")
    for field in ("assets",):
        if a[field] != b[field]:
            raise ValueError("Installed model/input identities differ")
    if (a["environment"]["core"] != b["environment"]["core"] or
            a["environment"]["sources"] != b["environment"]["sources"] or common_graph(a["graph"]) != common_graph(b["graph"])):
        raise ValueError("Core, implementation or common recipe differs")
    for key in ("torch_version", "torch_cuda", "device"):
        if a["live"][key] != b["live"][key]:
            raise ValueError("Actual runtime/device differs")
    for key in ("thermal_scope",):
        if not a["terminal"].get(key) or a["terminal"][key] != b["terminal"].get(key):
            raise ValueError("Cold/warm timing scope differs or is missing")
    if a['environment'].get('runtime_options') != b['environment'].get('runtime_options'):
        raise ValueError('Runtime memory reservation differs between comparison routes')
    if a["terminal"]["resource_guard"]["gpu_uuid"] != b["terminal"]["resource_guard"]["gpu_uuid"]:
        raise ValueError("GPU UUID differs")
    if a["reports"]["conditioning"] != b["reports"]["conditioning"]:
        raise ValueError("Actual encoded conditioning/initial latent differs")
    if a["graph"]["10"]["inputs"]["seed"] != b["graph"]["10"]["inputs"]["seed"]:
        raise ValueError("Sampler seeds differ")
    for field in ("width", "height", "frames", "fps", "sample_rate", "channels"):
        if a["media"]["timeline"][field] != b["media"]["timeline"][field]:
            raise ValueError("Actual media workload differs")
    seconds_a, seconds_b = a["timing"]["elapsed_seconds"], b["timing"]["elapsed_seconds"]
    return {"status": "matched_pair_wall_time_only_quality_unreviewed", "baseline": a["root"], "candidate": b["root"],
            "baseline_seconds": seconds_a, "candidate_seconds": seconds_b, "saved_fraction": 1-seconds_b/seconds_a,
            "speedup": seconds_a/seconds_b, "cache_scope": a["terminal"]["thermal_scope"],
            "video_vae_seconds": {label: next(row["seconds"] for row in run["reports"]["decode"]["timings"]
                                               if row["stage"] == "video_vae_decode") for label, run in (("baseline", a), ("candidate", b))},
            "quality": "Human full-video/face/music/speech/lip-sync review required; no winner inferred.",
            "sample_scope": "One serial pair, no statistical significance, OS/file cache and external load remain confounders."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=73)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    research = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909"
    if not args.output.resolve().is_relative_to(research) or args.output.exists():
        raise ValueError("Use a new report within acceleration research")
    if args.media:
        if not args.ffmpeg or not args.ffprobe or args.baseline or args.candidate:
            raise ValueError("Media mode needs explicit FFmpeg/FFprobe and no run pair")
        result = audit_media(args.media, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                             width=args.width, height=args.height, count=args.frames)
        result["scope"] = "Selected existing media only; not proof of new progressive generation"
    else:
        if not args.baseline or not args.candidate:
            raise ValueError("Pair mode needs two actual run directories")
        result = compare_runs(load_run(args.baseline), load_run(args.candidate))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
