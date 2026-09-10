"""Serial real-model continuation of an explicitly named candidate, for testing.

Requires --confirm-test-selection to record the operator's test choice, never
human quality acceptance. Copies the paused candidate and preparation to a new
artifact root. Does not regenerate the candidate or modify the prior evidence.
The optional --exercise-interrupt-resume mode interrupts the owned first process
during the unfinished window, verifies the atomic checkpoint, then finishes it
in a fresh owned ComfyUI process.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
from unittest.mock import patch

import pynvml

try:
    from tools import run_video_outpaint_t8_probe as t8
    from tools.build_video_outpaint_candidate_workflows import _closure
except ModuleNotFoundError:
    import run_video_outpaint_t8_probe as t8
    from build_video_outpaint_candidate_workflows import _closure


ROOT = Path(__file__).resolve().parents[1]


def selected_source_mode(prior):
    """Require the selected preview's declared pixel policy, not a node default."""
    preview = prior.get("preview_report")
    if not isinstance(preview, dict):
        raise ValueError("selected candidate needs a preview report")
    mode = t8.cases.pixel_receipt_module().validate_source_mode(
        preview.get("source_mode", "preserve_source"))
    if (preview.get("source_exact_before_encoding") is not (mode == "preserve_source")
            or preview.get("source_reconstructed", False) is not (mode == "joint_decode")):
        raise ValueError("selected preview source mode contradicts pixel evidence")
    candidates = [node["inputs"] for node in prior["prompt"].values()
                  if node["class_type"] == "MiniMaxH3VideoOutpaintCandidateT8"]
    if len(candidates) != 1:
        raise ValueError("requires exactly one selected candidate generation graph")
    graph_mode = t8.cases.candidate_finish_inputs(candidates[0])["source_mode"]
    if mode != graph_mode:
        raise ValueError("selected preview source mode differs from generation graph")
    return mode


def verify_selected_delivery(prior, sidecar, output_sha):
    return t8.cases.verify_delivery(
        sidecar, output_sha, prior["source_sha256"], prior["plan_sha256"],
        expected_source_mode=selected_source_mode(prior))


async def submit_with_owned_gpu_trigger(
    probe, *, server, prompt, timeout_seconds, trigger_free_below_mib, trigger_timeout_seconds,
):
    """Interrupt the only prompt after its owned GPU load proves sampling started."""
    if trigger_free_below_mib <= 0 or trigger_timeout_seconds <= 0:
        raise ValueError("GPU interrupt trigger and timeout must be positive")

    async def request_interrupt():
        import aiohttp

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        started = time.monotonic()
        observed = None
        while time.monotonic() - started < trigger_timeout_seconds:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            observed = memory.free / 1024**2
            if observed <= trigger_free_below_mib:
                break
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("owned prompt never reached the configured GPU sampling load")
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await probe._json_request(session, "POST", f"{server}/interrupt", json={})
        return {"response": response, "observed_free_mib": observed,
                "elapsed_seconds": round(time.monotonic() - started, 4)}

    execution_task = asyncio.create_task(
        probe.submit_prompt(server=server, prompt=prompt, timeout_seconds=timeout_seconds))
    interrupt_task = asyncio.create_task(request_interrupt())
    done, _ = await asyncio.wait({execution_task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED)
    if execution_task in done and interrupt_task not in done:
        interrupt_task.cancel()
        try:
            await interrupt_task
        except asyncio.CancelledError:
            pass
        execution = await execution_task
        trigger = None
    else:
        trigger = await interrupt_task
        execution = await execution_task
    execution["owned_gpu_interrupt_trigger"] = trigger
    return execution


def build_selected_prompt(prior):
    graph = deepcopy(prior["prompt"])
    candidates = [n["inputs"] for n in graph.values() if n["class_type"] == "MiniMaxH3VideoOutpaintCandidateT8"]
    if len(candidates) != 1:
        raise ValueError("requires exactly one generated candidate in the prior graph")
    candidate = candidates[0]
    identifier = prior["candidate_id"]
    if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("requires an explicit full candidate ID")
    if set(graph) & {"read_candidate", "test_select", "continue_selected", "save_selected", "selection_identifier"}:
        raise ValueError("prior graph uses reserved test node IDs")
    graph["read_candidate"] = {"class_type": "MiniMaxH3VideoOutpaintLoadCandidateT8", "inputs": {
        "prepared": candidate["prepared"], "candidate_name": candidate["candidate_name"], "candidate_id": identifier}}
    graph["test_select"] = {"class_type": "MiniMaxH3VideoOutpaintSelectCandidateT8", "inputs": {
        "candidate": ["read_candidate", 0], "confirm_selection": True}}
    graph["selection_identifier"] = {"class_type": "PreviewAny", "inputs": {"source": ["test_select", 2]}}
    graph["continue_selected"] = {"class_type": "MiniMaxH3VideoOutpaintContinueCandidateT8", "inputs": {
        "model": candidate["model"], "selected": ["test_select", 0]}}
    graph["save_selected"] = {"class_type": "MiniMaxH3VideoOutpaintComposeCandidateT8", "inputs": {
        "sampled": ["continue_selected", 0], "video_vae": candidate["video_vae"], "output_name": "selected_test32"}}
    return _closure(graph, ["selection_identifier", "save_selected"])


def run(args):
    probe, sha, atomic = t8.upstream.probe, t8.upstream._sha256_file, t8.upstream._atomic_json
    prior_root, root = args.prior_root.resolve(strict=True), args.run_root.resolve()
    prior_root.relative_to(ROOT / "artifacts")
    root.relative_to(ROOT / "artifacts")
    prior = json.loads((prior_root / "report.json").read_bytes())
    if prior["status"] != "candidate_generated_human_review_pending":
        raise ValueError("prior candidate is not successfully generated")
    # A live process must never be inferred dead merely from a report.
    pid = prior.get("server_pid")
    if pid and t8.psutil.pid_exists(pid):
        process = t8.psutil.Process(pid)
        if process.create_time() <= (prior_root / "report.json").stat().st_mtime + 1:
            raise RuntimeError("prior candidate server still present")
    graph = build_selected_prompt(prior)
    source_mode = selected_source_mode(prior)
    prior_candidate = Path(prior["cache_root"]).resolve(strict=True)
    prior_candidate.relative_to(prior_root / "output/T8_H3_Outpaint_Cache")
    prior_cache = prior_candidate.parents[1]
    if any(p.is_symlink() for p in prior_cache.rglob("*")):
        raise ValueError("linked candidate/preparation assets are not supported")
    window_path = prior_candidate / "outpaint_windows.json"
    window = json.loads(window_path.read_bytes())
    if window["status"] != "paused" or len(window["committed"]) != 1:
        raise ValueError("expected one normally paused first window")
    source_node = next(n for n in graph.values() if n["class_type"] == "LoadVideo")
    source = prior_root / "input" / source_node["inputs"]["file"]
    pynvml.nvmlInit()
    free = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1024**2
    checks = {"new_artifact_root": not root.exists(), "source_unchanged": sha(source) == prior["source_sha256"],
        "gpu_headroom": free >= args.min_free_mib, "port_free": not probe.port_is_listening(args.host, args.port),
        "no_other_gpu_server": not any(probe.port_is_listening("127.0.0.1", p) for p in (8188, 8190, 8193)),
        "kj_audited": t8.cases.is_audited_kj_source(args.comfy_root / "custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py")}
    report = {"scope": "explicit_operator_test_selection_not_human_quality_acceptance", "status": "preflight",
        "checks": checks, "prompt": graph, "candidate_id": prior["candidate_id"], "prior_root": str(prior_root),
        "source_sha256": prior["source_sha256"], "plan_sha256": prior["plan_sha256"],
        "selected_first_frame_rgb8_sha256": prior["preview_report"]["rgb8_sha256"],
        "first_window_sha256": window["committed"][0]["sha256"], "prior_manifest_sha256": sha(window_path),
        "human_acceptance": False, "gpu_initial_free_mib": free, "source_mode": source_mode,
        "exercise_interrupt_resume": bool(args.exercise_interrupt_resume),
        "interrupt_when_free_below_mib": (args.interrupt_when_free_below_mib
                                           if args.exercise_interrupt_resume else None)}
    if not args.confirm_test_selection:
        print(json.dumps(report, indent=2))
        return report
    if not all(checks.values()):
        raise RuntimeError(f"selected continuation preflight failed: {checks}")
    root.mkdir(parents=True, exist_ok=False)
    atomic(root / "report.json", report)
    (root / "input").mkdir()
    shutil.copyfile(source, root / "input" / source.name)
    models = []
    for old in prior["models"]:
        path = Path(old["path"])
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        if old != actual:
            raise ValueError("actual model changed since candidate generation")
        models.append(actual)
    report["models"] = models
    cache = root / "output/T8_H3_Outpaint_Cache" / prior_cache.name
    shutil.copytree(prior_cache, cache)
    candidate_root = cache / "candidates" / prior_candidate.name
    if sha(candidate_root / "outpaint_windows.json") != report["prior_manifest_sha256"]:
        raise ValueError("copied candidate manifest changed")
    runtime = [*(ROOT / "h3_t8").glob("video_outpaint*.py"), *(ROOT / "h3_t8").glob("nodes_video_outpaint*.py")]
    report["implementation_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in runtime}
    extra = root / "draft_paths.yaml"
    atomic(extra, {"selected_test_only": {"custom_nodes": str(ROOT / "tools")}})
    original_command = probe._server_command

    def command(server_args, run_root):
        result = original_command(server_args, run_root)
        first, last = result.index("--whitelist-custom-nodes") + 1, result.index("--input-directory")
        result[first:last] = ["outpaint_candidate_extension", "ComfyUI-KJNodes"]
        result[result.index("--input-directory") + 1] = str(root / "input")
        return result + ["--extra-model-paths-config", str(extra)]

    server = None
    telemetry = None
    telemetry_rows = []
    active_label = None
    started = time.monotonic()

    def start_phase(label):
        owned = probe.IsolatedServer(args, root, label)
        with patch.object(probe, "_server_command", command):
            owned.start()
        with urllib.request.urlopen(f"http://{args.host}:{args.port}/object_info", timeout=30) as response:
            info = json.load(response)
        used = {n["class_type"] for n in graph.values()}
        if used - set(info):
            owned.stop()
            raise RuntimeError(f"missing selected workflow nodes: {used - set(info)}")
        return owned, info, used

    def stop_phase(owned, sampler, label):
        if sampler is not None:
            sampler.stop()
            sampler.write_csv(root / f"telemetry-{label}.csv")
            telemetry_rows.append({"phase": label, **sampler.summary(),
                                   "journal_error": sampler.journal_error})
        if owned is not None:
            owned.stop()

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            raise RuntimeError("parent disables CUDA")
        first_label = "selected-interrupt" if args.exercise_interrupt_resume else "selected"
        server, info, used = start_phase(first_label)
        active_label = first_label
        atomic(root / "schema_snapshot.json", {k: info[k] for k in used})
        report.update(status="running", server_pid=server.process.pid, cache_root=str(candidate_root))
        atomic(root / "report.json", report)
        telemetry = t8.LiveMemorySampler(server.process.pid, root / f"telemetry-{first_label}.live.jsonl")
        telemetry.start()
        server_url = f"http://{args.host}:{args.port}"
        if args.exercise_interrupt_resume:
            result = asyncio.run(submit_with_owned_gpu_trigger(
                probe, server=server_url, prompt=graph, timeout_seconds=args.timeout,
                trigger_free_below_mib=args.interrupt_when_free_below_mib,
                trigger_timeout_seconds=args.interrupt_trigger_timeout))
        else:
            result = asyncio.run(probe.submit_prompt(
                server=server_url, prompt=graph, timeout_seconds=args.timeout))
        if args.exercise_interrupt_resume:
            report["interrupted_execution"] = result
            if ((result.get("terminal") or {}).get("type") != "execution_interrupted"
                    or result.get("owned_gpu_interrupt_trigger") is None):
                raise RuntimeError("owned interruption was not acknowledged as execution_interrupted")
            interrupted = json.loads((candidate_root / "outpaint_windows.json").read_bytes())
            if (interrupted["status"] != "interrupted"
                    or interrupted["committed"] != window["committed"]):
                raise RuntimeError("interruption changed the selected prefix or published a partial window")
            report["interrupted_checkpoint"] = {
                "status": interrupted["status"],
                "committed_windows": len(interrupted["committed"]),
                "manifest_sha256": sha(candidate_root / "outpaint_windows.json"),
            }
            stop_phase(server, telemetry, first_label)
            server = None
            telemetry = None
            active_label = None
            report.update(status="restart_running", interrupted_server_stopped=True)
            atomic(root / "report.json", report)
            server, _, _ = start_phase("selected-resume")
            active_label = "selected-resume"
            report["resumed_server_pid"] = server.process.pid
            telemetry = t8.LiveMemorySampler(server.process.pid, root / "telemetry-selected-resume.live.jsonl")
            telemetry.start()
            result = asyncio.run(probe.submit_prompt(server=f"http://{args.host}:{args.port}", prompt=graph,
                timeout_seconds=args.timeout))
            report["resumed_execution"] = result
        report["execution"] = result
        if (result.get("terminal") or {}).get("type") != "execution_success":
            raise RuntimeError("selected continuation failed; inspect execution record and retained cache")
        final_window = json.loads((candidate_root / "outpaint_windows.json").read_bytes())
        if final_window["status"] != "sampled" or final_window["committed"][0] != window["committed"][0]:
            raise RuntimeError("continuation did not preserve the selected prefix")
        report["committed_windows"] = len(final_window["committed"])
        report["final_window_sha256"] = [item["sha256"] for item in final_window["committed"]]
        reference = prior.get("reference_complete_window_sha256")
        if reference is not None and report["final_window_sha256"] != reference:
            raise RuntimeError("interrupted/resumed windows differ from the complete learned reference")
        report["matches_complete_learned_reference"] = reference is not None
        outputs = list((root / "output/T8_H3_Outpaint").glob("*.mp4"))
        if len(outputs) != 1:
            raise RuntimeError("expected one selected output")
        sidecar = outputs[0].with_suffix(".mp4.outpaint.json")
        delivery = verify_selected_delivery(prior, sidecar, sha(outputs[0]))
        if (delivery.get("selected_first_frame_rgb8_sha256") != report["selected_first_frame_rgb8_sha256"]
                or delivery.get("selected_first_frame_verified_before_encoding") is not True):
            raise RuntimeError("selected first-frame encoder guard missing or failed")
        report["delivery"] = {"path": str(sidecar), "report": delivery}
        # Inspect actual original stream count rather than guessing from plan.
        streams = json.loads(subprocess.check_output([args.ffprobe, "-v", "error", "-show_streams", "-of", "json", str(source)], timeout=60))["streams"]
        audio_tracks = sum(s["codec_type"] == "audio" for s in streams)
        report["media"] = t8.cases.strict_media(outputs[0], args.ffmpeg, args.ffprobe, audio_tracks, sha)
        if any(sha(ROOT / name) != digest for name, digest in report["implementation_sha256"].items()):
            raise ValueError("implementation changed during the test")
        report.update(status=("interrupt_restart_media_pass_human_review_pending"
                              if args.exercise_interrupt_resume
                              else "selected_continuation_media_pass_human_review_pending"),
                      first_prefix_preserved=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            stop_phase(server, telemetry, active_label or "selected")
            if telemetry_rows:
                report["memory_phases"] = telemetry_rows
                report["memory"] = {
                    "max_gpu_used_mib": max(item["max_gpu_used_mib"] for item in telemetry_rows),
                    "min_gpu_free_mib": min(item["min_gpu_free_mib"] for item in telemetry_rows),
                }
                report["telemetry_error"] = next(
                    (item["journal_error"] for item in telemetry_rows if item["journal_error"]), None)
        finally:
            report["prior_candidate_unchanged"] = sha(window_path) == report["prior_manifest_sha256"]
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            atomic(root / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(root / "report.json")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--confirm-test-selection", action="store_true")
    parser.add_argument("--exercise-interrupt-resume", action="store_true",
                        help="interrupt the unfinished learned window and finish it in a fresh owned ComfyUI process")
    parser.add_argument("--interrupt-when-free-below-mib", type=float, default=9000,
                        help="interrupt only after free VRAM proves the owned prompt entered GPU sampling")
    parser.add_argument("--interrupt-trigger-timeout", type=float, default=300,
                        help="maximum wait for the owned prompt to reach the GPU sampling threshold")
    parser.add_argument("--comfy-root", type=Path, default=ROOT.parents[1])
    parser.add_argument("--python", type=Path, default=ROOT.parents[2] / "python/python.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--reserve-vram-gib", type=float, default=5.0)
    parser.add_argument("--min-free-mib", type=float, default=10000)
    parser.add_argument("--server-start-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=10000)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    if args.interrupt_when_free_below_mib <= 0 or args.interrupt_trigger_timeout <= 0:
        parser.error("GPU interrupt trigger and timeout must be positive")
    run(args)
