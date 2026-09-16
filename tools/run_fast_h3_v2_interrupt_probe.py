"""Owned8s V2 native interruption + new-process resume, never fake completion.

CPU only validates live registration/graph. GPU must be explicit. This tool
does not edit Core, production APIs or previous controller evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_fast_h3_v2_loop_probe as loop  # noqa: E402

transport = loop.transport
SCHEMA = "t8.fasth3_v2.owned_interrupt_probe/v1"
INTERRUPTED = "verified_native_one_segment_interrupted_human_pending"
RESUMED = "verified_native_interruption_new_process_resume_8s_human_pending"
STATE_FOLDER = "minimax_h3_t8_long_video"
STATE_NAME = "in_node_loop_effects_state.json"
EVIDENCE_NAMES = ("expected.json", "native-trigger-state.json", "interrupt-request.json", "native-interrupted-state.json",
                  "first-segment-evidence.json", "first-media-probe.json", "first-media-strict-decode.json",
                  "generation/prompt.json", "generation/submission.json", "generation/events.jsonl",
                  "generation/history.json", "generation/timing.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(root, value):
    root, path = Path(root).resolve(), Path(value)
    if not path.is_absolute():
        if "\\" in str(value) or ":" in str(value) or any(p in ("", ".", "..") for p in str(value).split("/")):
            raise ValueError("Unsafe native evidence path")
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError("Native evidence leaves owned scope")
    return path


def chain_root(output, chain_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", chain_id):
        raise ValueError("Unsafe chain_id")
    return Path(output).resolve() / STATE_FOLDER / chain_id


def frozen_sources(project):
    return {**loop.frozen_sources(project),
            "tools/run_fast_h3_v2_interrupt_probe.py": loop.file_identity(Path(__file__))["sha256"]}


def validate_state(state, chain_id, status):
    expected = {"schema": 1, "format": "minimax_h3_t8_in_node_loop_effects",
                "chain_id": chain_id, "status": status, "segment_count": 2,
                "accepted_count": 1, "current_segment_index": 1, "manifest_revision": 1}
    if any(state.get(key) != value or type(state.get(key)) is not type(value) for key, value in expected.items()):
        raise RuntimeError("Not an authoritative own-chain one-segment " + status + " state")
    if not re.fullmatch(r"[0-9a-f]{64}", state.get("contract_sha256", "")) or state.get("final_video_path"):
        raise RuntimeError("Interrupted state has invalid contract or existing final")


def validate_first_stages(audit, profile):
    """Validate only the actual signed first segment, not invented completion."""
    dual = audit["sampling_plan"]["dual_model"]
    stages, vsa = [], 0
    rungs = [999, 874, 749, 624, 500, 375, 250, 125, 0]
    for name, start, end in (("first_pass", 0, 4), ("second_pass", 4, 8)):
        stage = dual[name]
        schedule = stage.get("schedule", {})
        expected = [10*(r/1000)/(1+9*(r/1000)) for r in rungs[start:end+1]]
        sigmas = stage.get("sigmas", [])
        if (stage.get("nfe") != 4 or stage.get("completed_network_forwards") != 4
                or schedule.get("profile") != profile or schedule.get("stage_start") != start
                or schedule.get("stage_end") != end or schedule.get("video_shift") != 10.
                or schedule.get("audio_shift") != 3. or schedule.get("rungs") != rungs[:-1]
                or len(sigmas) != 5 or any(not math.isfinite(s) or abs(s-e) > 1e-6 for s, e in zip(sigmas, expected))):
            raise RuntimeError("Signed first stage is not exact completed4+4 V2")
        memory = stage.get("memory_composition") or {}
        runtime = stage.get("backend", {}).get("fasth3_v2", {})
        counts = runtime.get("counts", {})
        if (memory.get("kind") != "t8_h3_memory" or memory.get("head_chunks") != 4
                or memory.get("ffn_settings") != [2, 4096] or runtime.get("profile") != profile
                or runtime.get("head_chunks") != 4
                or any(type(n) is not int or n < 0 for n in counts.values())
                or runtime.get("actual_vsa_dispatched") is not (counts.get("vsa", 0) > 0)):
            raise RuntimeError("Signed first memory/dispatch owner differs")
        vsa += counts.get("vsa", 0)
        relay = stage.get("prompt_relay_execution")
        if profile == "dense_compat_exp":
            if counts.get("vsa", 0) or not relay or relay.get("completed_forwards") != 4 or relay.get("routed_attention_calls", 0) <= 0:
                raise RuntimeError("First Dense Relay routing differs")
        elif profile != "trained_vsa_exp" or relay is not None:
            raise RuntimeError("First trained VSA must not combine Relay")
        stages.append({"segment_index": 0, "stage": name, "completed_network_forwards": 4,
                       "profile": profile, "dispatch": deepcopy(runtime), "memory": deepcopy(memory), "relay": deepcopy(relay)})
    if profile == "trained_vsa_exp" and vsa == 0:
        raise RuntimeError("First reference/VSA segment never dispatched actual sparse attention")
    return stages


def first_evidence(output, chain_id, state, profile):
    """Pure read-only signed first-segment storage identity; no tensor loading."""
    root = chain_root(output, chain_id)
    manifest = read_json(inside(root, "manifest.json"))
    if (manifest.get("schema") != 2 or manifest.get("format") != "minimax_h3_t8_accepted_manifest"
            or manifest.get("chain_id") != chain_id or manifest.get("revision") != state["manifest_revision"]
            or len(manifest.get("segments", [])) != state["accepted_count"]):
        raise RuntimeError("Native accepted manifest and state disagree")
    entry = manifest["segments"][0]
    candidate_id = entry.get("candidate_id", "")
    if entry.get("index") != 0 or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", candidate_id):
        raise RuntimeError("First accepted candidate identity invalid")
    candidate_base = f"candidates/segment_00000/{candidate_id}"
    candidate = read_json(inside(root, candidate_base + "/candidate.json"))
    if any(candidate.get(key) != value for key, value in {
            "schema": 1, "status": "candidate", "chain_id": chain_id, "index": 0,
            "candidate_id": candidate_id, "frame_count": 124, "fps": 24,
            "width": 512, "height": 768, "is_final_segment": False}.items()):
        raise RuntimeError("First candidate descriptor scope/geometry differs")
    audit = read_json(inside(root, candidate_base + "/effects_audit.json"))
    unsigned = dict(audit)
    claimed = unsigned.pop("audit_sha256", None)
    if (claimed != digest(unsigned) or audit.get("schema") != 1
            or audit.get("format") != "minimax_h3_t8_in_node_loop_effects_segment"
            or audit.get("segment_index") != 0 or audit.get("candidate_id") != candidate_id
            or audit.get("contract_sha256") != state["contract_sha256"]):
        raise RuntimeError("First segment signed audit differs")
    stages = validate_first_stages(audit, profile)
    dual = audit["sampling_plan"]["dual_model"]
    if dual.get("low_reused") is not False or dual.get("high_reused") is not False:
        raise RuntimeError("First execution unexpectedly reused previous stage counts")
    files = {}
    for name in (candidate_base + "/candidate.json", candidate_base + "/effects_audit.json",
                 entry.get("video_path", ""), entry.get("context_path", ""),
                 candidate.get("video_path", ""), candidate.get("context_path", ""),
                 dual.get("low_context", {}).get("path", "")):
        path = inside(root, name)
        files[path.relative_to(root).as_posix()] = loop.file_identity(path)
    for name, key in (("video_path", "video_sha256"), ("context_path", "context_sha256")):
        identity = loop.file_identity(inside(root, entry[name]))
        if identity["sha256"] != entry[key] or candidate.get(key) != entry[key]:
            raise RuntimeError("Accepted first media/context checksum differs from candidate")
    receipts = []
    for path in sorted((root / "dual_stages/segment_00000").rglob("*.json")):
        receipt = read_json(path)
        if receipt.get("schema") != 1 or receipt.get("contract", {}).get("job") != state["contract_sha256"]:
            raise RuntimeError("First stage receipt contract differs")
        tensor = inside(path.parent, receipt.get("tensor_file", ""))
        identity = loop.file_identity(tensor)
        if identity["sha256"] != receipt.get("tensor_sha256"):
            raise RuntimeError("First stage tensor checksum differs")
        receipts.append(receipt["stage"])
        for selected in (path, tensor):
            files[selected.relative_to(root).as_posix()] = loop.file_identity(selected)
    if sorted(receipts) != ["high_input", "high_output", "low_x0"]:
        raise RuntimeError("Missing or duplicate first native stage cache receipts")
    if list((root / "dual_stages/segment_00001").rglob("*.json")):
        raise RuntimeError("Second segment already completed a stage before interrupt; new8 cannot be certified")
    return {"entry": entry, "audit": audit, "files": files,
            "persisted_first_network_forwards": 8, "dispatch": stages}


def require_interrupted_history(history, events, submission, request):
    prompt_id = submission.get("prompt_id")
    if (not prompt_id or request.get("prompt_id") != prompt_id or request.get("http_status") != 200
            or request.get("method") != "POST" or request.get("endpoint") != "/interrupt"
            or type(request.get("owner_pid")) is not int or request["owner_pid"] <= 0):
        raise RuntimeError("No actual owned targeted interrupt request")
    terminal = [e.get("type") for e in events if e.get("data", {}).get("prompt_id") == prompt_id
                and e.get("type") in ("execution_interrupted", "execution_success", "execution_error")]
    messages = history.get("status", {}).get("messages", [])
    authoritative = [m[0] for m in messages if isinstance(m, list) and len(m) == 2
                     and isinstance(m[1], dict) and m[1].get("prompt_id") == prompt_id
                     and m[0] in ("execution_interrupted", "execution_success", "execution_error")]
    if terminal != ["execution_interrupted"] or authoritative != terminal or history.get("status", {}).get("completed") is not False:
        raise RuntimeError("Missing authoritative execution_interrupted; no interruption qualification")


def validate_trigger(trigger, interrupted, request, chain_id):
    validate_state(trigger, chain_id, "running")
    if (trigger["contract_sha256"] != interrupted["contract_sha256"]
            or request.get("trigger_state_sha256") != digest(trigger)):
        raise RuntimeError("Interrupt request is not bound to the actual one-accepted running state")


def post_owned_interrupt(server, prompt_id):
    """/interrupt has an empty200 response, unlike OwnedServer JSON requests."""
    import requests
    server.assert_port_owner()
    queue = server.request("GET", "/queue").get("queue_running", [])
    if len(queue) != 1 or queue[0][1] != prompt_id:
        raise RuntimeError("Target prompt is not the sole running owned job")
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(server.url + "/interrupt", json={"prompt_id": prompt_id}, timeout=(1, 2))
        response.raise_for_status()
        if response.status_code != 200:
            raise RuntimeError("Owned interrupt did not return HTTP200")
    return {"method": "POST", "endpoint": "/interrupt", "http_status": 200,
            "prompt_id": prompt_id, "owner_pid": server.process.pid}


class InterruptWatcher:
    def __init__(self, server, generation, output, chain_id):
        self.server, self.generation, self.output, self.chain_id = server, generation, output, chain_id
        self.sent = False

    def check(self):
        if self.sent or not (self.generation / "submission.json").is_file():
            return
        path = chain_root(self.output, self.chain_id) / STATE_NAME
        if not path.is_file():
            return
        state = read_json(path)  # Native writes atomically; corrupt evidence fails closed.
        if state.get("accepted_count") != 1 or state.get("current_segment_index") != 1 or state.get("status") != "running":
            return
        validate_state(state, self.chain_id, "running")
        transport.write_json(self.generation.parent / "native-trigger-state.json", state)
        receipt = post_owned_interrupt(self.server, read_json(self.generation / "submission.json")["prompt_id"])
        receipt["trigger_state_sha256"] = digest(state)
        transport.write_json(self.generation.parent / "interrupt-request.json", receipt)
        self.sent = True


def verify_first_unchanged(output, chain_id, frozen):
    root = chain_root(output, chain_id)
    manifest = read_json(root / "manifest.json")
    if not manifest.get("segments") or manifest["segments"][0] != frozen["entry"]:
        raise RuntimeError("Resumed first accepted entry changed")
    for name, identity in frozen["files"].items():
        if loop.file_identity(inside(root, name)) != identity:
            raise RuntimeError("Resumed first accepted bytes/context/stage cache changed")


def previous_evidence(previous, project, root):
    previous = Path(previous).resolve(strict=True)
    if previous == root or not previous.is_relative_to(project / "artifacts"):
        raise ValueError("Resume must use another exact owned interrupted artifact")
    terminal, expected = read_json(previous / "terminal.json"), read_json(previous / "expected.json")
    if (terminal.get("schema") != SCHEMA or terminal.get("status") != INTERRUPTED
            or terminal.get("human_review") != "pending" or terminal.get("cleanup_errors") != []
            or terminal.get("server_stop", {}).get("owned_children_remaining") != []):
        raise ValueError("Previous artifact is not a verified interrupted owned run")
    if terminal.get("expected_sha256") != digest(expected):
        raise RuntimeError("Previous expected configuration changed")
    output, input_scope = Path(terminal["output_directory"]).resolve(strict=True), Path(terminal["input_directory"]).resolve(strict=True)
    if output != previous / "output" or input_scope != previous / "input":
        raise ValueError("Previous input/output leaves owned artifact scope")
    if not output.is_dir() or not input_scope.is_dir():
        raise ValueError("Previous input/output missing")
    evidence_files = terminal.get("evidence_files", {})
    if set(evidence_files) != set(EVIDENCE_NAMES):
        raise RuntimeError("Previous interruption evidence manifest is incomplete")
    for name, identity in evidence_files.items():
        if loop.file_identity(inside(previous, name)) != identity:
            raise RuntimeError("Previous immutable interruption evidence changed")
    graph = expected["pilot_graphs"]["v2_loop"]
    chain_id = graph["8"]["inputs"]["chain_id"]
    state = read_json(chain_root(output, chain_id) / STATE_NAME)
    validate_state(state, chain_id, "interrupted")
    validate_trigger(read_json(previous / "native-trigger-state.json"), state,
                     read_json(previous / "interrupt-request.json"), chain_id)
    if read_json(previous / "generation/prompt.json") != graph:
        raise RuntimeError("Previous generation graph changed or differs from frozen expected")
    timing = read_json(previous / "generation/timing.json")
    if timing.get("terminal") != "execution_interrupted" or timing.get("complete_uncached_graph") is not False:
        raise RuntimeError("Previous transport timing did not preserve its real interrupted terminal")
    require_interrupted_history(read_json(previous / "generation/history.json"),
        [json.loads(line) for line in (previous / "generation/events.jsonl").read_text(encoding="utf8").splitlines()],
        read_json(previous / "generation/submission.json"), read_json(previous / "interrupt-request.json"))
    frozen = read_json(previous / "first-segment-evidence.json")
    current = first_evidence(output, chain_id, state, terminal["profile"])
    if current != frozen:
        raise RuntimeError("Previous first native segment evidence changed")
    return previous, terminal, expected, output, input_scope, chain_id, frozen


def validate_resume(report, frozen, profile):
    qualified = loop.validate_loop_report(report, profile)
    if report.get("resume_action") != "generated_or_resumed_then_completed" or report["segment_audits"][0] != frozen["audit"]:
        raise RuntimeError("Not a native incomplete resume with unchanged persisted first audit")
    dual = report["segment_audits"][1]["sampling_plan"]["dual_model"]
    if dual.get("low_reused") is not False or dual.get("high_reused") is not False:
        raise RuntimeError("Persisted second-stage counts cannot be claimed as new network forwards")
    return {"persisted_first_network_forwards": 8, "new_second_network_forwards": 8,
            "final_persisted_successful_network_forwards": qualified["actual_network_forwards"],
            "previous_partial_second_attempt_forwards": "not_measured_do_not_infer_zero",
            "stage_audit": qualified, "human_quality_accepted": False}


def validate_final_native(output, chain_id, report):
    root = chain_root(output, chain_id)
    state, manifest = read_json(root / STATE_NAME), read_json(root / "manifest.json")
    fields = {"chain_id": chain_id, "status": "complete", "segment_count": 2,
              "accepted_count": 2, "current_segment_index": None, "manifest_revision": 2,
              "contract_sha256": report["contract_sha256"], "final_video_path": report["final_video_path"],
              "final_video_sha256": report["final_video_sha256"]}
    if any(state.get(k) != v or report.get(k) != v for k, v in fields.items()):
        raise RuntimeError("Final native state and successful report disagree")
    if manifest.get("chain_id") != chain_id or manifest.get("revision") != 2 or len(manifest.get("segments", [])) != 2:
        raise RuntimeError("Final native manifest is not two accepted segments")
    if [entry.get("candidate_id") for entry in manifest["segments"]] != [audit.get("candidate_id") for audit in report["segment_audits"]]:
        raise RuntimeError("Final native accepted candidates differ from signed audits")


def stage_media(path, evidence):
    """First accepted segment has124f/24fps, not the final192f/8s canvas."""
    from fractions import Fraction
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True)
    transport.write_json(evidence / "first-media-probe.json", {"returncode": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr})
    if probe.returncode or probe.stderr.strip():
        raise RuntimeError("First accepted media probe failed")
    payload = json.loads(probe.stdout)
    video = [v for v in payload["streams"] if v["codec_type"] == "video"]
    audio = [v for v in payload["streams"] if v["codec_type"] == "audio"]
    if (len(video) != 1 or len(audio) != 1 or video[0]["codec_name"] != "h264"
            or (video[0]["width"], video[0]["height"]) != (512, 768)
            or int(video[0]["nb_read_frames"]) != 124 or Fraction(video[0]["avg_frame_rate"]) != 24):
        raise RuntimeError("First accepted segment is not exact124f H264+audio2:3")
    for stream in (video[0], audio[0], payload["format"]):
        duration = float(stream["duration"])
        if not math.isfinite(duration) or abs(duration - 124/24) > 1/24:
            raise RuntimeError("First accepted AV duration differs")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], capture_output=True, text=True)
    transport.write_json(evidence / "first-media-strict-decode.json", {"returncode": decoded.returncode, "stdout": decoded.stdout, "stderr": decoded.stderr})
    if decoded.returncode or decoded.stderr.strip():
        raise RuntimeError("First accepted AV strict decode failed")
    return {**loop.file_identity(path), "frames": 124, "fps": "24/1", "strict_decode": True, "quality_accepted": False}


def run(args, project, core, root):
    transport.CORE, transport.PROJECT = core, project
    original_command = transport.server_command
    result = {"schema": SCHEMA, "status": "incomplete", "profile": args.profile,
              "human_review": "pending", "root": str(root), "interrupt_qualification": "pending"}
    # This probe appends its exact4GiB option below. Keep the shared transport
    # on its declared zero-headroom profile rather than broadening its API.
    server, monitor, guard = transport.OwnedServer(root, args.port, args.cpu, 0), None, loop.ResourceGuard(
        loop.GuardPolicy(startup_free_gpu_bytes=512*loop.MIB, startup_free_ram_bytes=4096*loop.MIB))
    output, input_scope, previous, frozen, previous_expected = root / "output", root / "input", None, None, None
    try:
        if args.resume_from:
            if args.cpu:
                raise ValueError("Native interrupted resume requires explicit GPU")
            previous, old, previous_expected, output, input_scope, chain_id, frozen = previous_evidence(args.resume_from, project, root)
            if old["profile"] != args.profile or (args.chain_id and args.chain_id != chain_id):
                raise ValueError("Resume profile/chain differs")
            original_image = old["reference"]
            if loop.file_identity(args.reference) != original_image:
                raise RuntimeError("Resume original reference differs")
            copied = previous_expected["owned_reference"]
            filename = Path(copied["path"]).name
            immutable_previous = {name: loop.file_identity(previous / name) for name in ("terminal.json", "expected.json")}
        else:
            chain_id = args.chain_id or "v2_interrupt_" + hashlib.sha256(root.name.encode()).hexdigest()[:20]
            input_scope.mkdir()
            original_image = loop.file_identity(args.reference)
            from PIL import Image
            with Image.open(args.reference) as image:
                if image.width * 3 != image.height * 2:
                    raise ValueError("Reference must be exact2:3, no stretching")
            filename = "reference_" + original_image["sha256"][:24] + args.reference.suffix.lower()
            with args.reference.open("rb") as src, (input_scope / filename).open("xb") as dst:
                shutil.copyfileobj(src, dst, 8*loop.MIB)
            copied = loop.file_identity(input_scope / filename)
            if copied["sha256"] != original_image["sha256"]:
                raise RuntimeError("Owned reference import changed bytes")
        result.update(reference=original_image, output_directory=str(output), input_directory=str(input_scope))
        source = frozen_sources(project)
        expected = {"core": loop.verify_core_source(core), "sources": source,
            "mode": "cpu-smoke" if args.cpu else "gpu", "pilot_graphs": {}, "assets": [],
            "owned_reference": copied,
            "runtime_options": {"reserve_vram_gib": 5, "headroom_gib": 4}}
        if not args.cpu:
            expected["assets"].append(loop.model_identity(core))
            expected["assets"].extend(loop.file_identity(core / "models" / category / name) for category, name in loop.MODEL_ASSETS[1:])
        transport.write_json(root / "paths.json", loop.probe_resource_config(core, project))
        def command(*values):
            cmd = original_command(*values)
            cmd.append("--use-pytorch-cross-attention")
            if "--vram-headroom" in cmd:
                if cmd.count("--vram-headroom") != 1 or cmd[cmd.index("--vram-headroom")+1] != "4":
                    raise RuntimeError("Owned interruption service requires exact4GiB native headroom")
            else:
                cmd.extend(["--vram-headroom", "4"])
            cmd[cmd.index("--output-directory")+1] = str(output)
            cmd[cmd.index("--input-directory")+1] = str(input_scope)
            return cmd
        transport.server_command = command
        with ExitStack() as cleanup:
            if not args.cpu:
                lease = core / "custom_nodes/minimax-h3-audio-T8/artifacts/acceleration-research-20260909/serial-gpu.lock"
                cleanup.enter_context(loop.SerialProbeLease(lease))
                reader = cleanup.enter_context(loop.NvmlResourceReader())
                reason = guard.observe(reader.sample(), startup=True)
                if reason:
                    raise RuntimeError("Startup resource margin: " + reason)
            server.start()
            if not args.cpu:
                monitor = transport.ContinuousGuard(reader, guard, root / "resources.jsonl", server)
                cleanup.callback(monitor.close)
                monitor.start()
            last_sources = [0.]
            def check():
                if monitor:
                    monitor.check()
                for asset in [original_image, copied, *expected["assets"]]:
                    stat = Path(asset["path"]).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (asset["bytes"], asset["mtime_ns"]):
                        raise RuntimeError("Continuous asset guard changed")
                if time.monotonic() - last_sources[0] >= 3:
                    if frozen_sources(project) != source:
                        raise RuntimeError("Continuous source guard changed")
                    last_sources[0] = time.monotonic()
            transport.wait_ready(server, check)
            info = server.request("GET", "/object_info")
            transport.write_json(root / "object-info.json", info)
            graph = loop.build_graph(info, first_frame=filename, chain_id=chain_id, profile=args.profile)
            def environment(payload, name):
                history, _ = transport.execute_graph(server, {"90": {"class_type": "T8ProgressiveEnvironmentAudit", "inputs": {"expected_json": json.dumps(payload)}},
                    "91": {"class_type": "PreviewAny", "inputs": {"source": ["90", 0]}}}, root / name, check)
                return transport.preview_report(history, "91")
            result["registration"] = environment(expected, "registration")
            expected["pilot_graphs"] = {"v2_loop": graph}
            if previous_expected is not None and expected != previous_expected:
                raise RuntimeError("Resume graph/Core/source/assets/output configuration differs")
            transport.write_json(root / "expected.json", expected)
            result["expected_sha256"] = digest(expected)
            try:
                result["graph_validation"] = environment(expected, "graph-validation")
            except RuntimeError as error:
                missing = loop.pending_materials(info, graph)
                if not args.cpu or not missing or not loop.material_validation_failed(root / "graph-validation", error):
                    raise
                result.update(status="live_registration_graph_pending_material_no_inference", pending_material=missing)
            else:
                if args.cpu:
                    result["status"] = "live_Core_graph_validation_pass_no_inference_no_interrupt"
                elif previous is None:
                    watcher = InterruptWatcher(server, root / "generation", output, chain_id)
                    def checked_interrupt():
                        check()
                        watcher.check()
                    try:
                        transport.execute_graph(server, graph, root / "generation", checked_interrupt, timeout=args.timeout)
                    except RuntimeError as error:
                        # execute_graph must keep its real interrupted terminal and
                        # failure receipt. Relabel no crash/timeout/source error.
                        if str(error) != "Graph failed, was cached, or did not execute all instrumented nodes":
                            raise
                        result["transport_native_terminal_error"] = str(error)
                    else:
                        raise RuntimeError("Graph completed without real interruption")
                    state = read_json(chain_root(output, chain_id) / STATE_NAME)
                    validate_state(state, chain_id, "interrupted")
                    validate_trigger(read_json(root / "native-trigger-state.json"), state,
                                     read_json(root / "interrupt-request.json"), chain_id)
                    require_interrupted_history(read_json(root / "generation/history.json"),
                        [json.loads(line) for line in (root / "generation/events.jsonl").read_text(encoding="utf8").splitlines()],
                        read_json(root / "generation/submission.json"), read_json(root / "interrupt-request.json"))
                    frozen = first_evidence(output, chain_id, state, args.profile)
                    transport.write_json(root / "native-interrupted-state.json", state)
                    transport.write_json(root / "first-segment-evidence.json", frozen)
                    result["first_media"] = stage_media(inside(chain_root(output, chain_id), frozen["entry"]["video_path"]), root)
                    result.update(status=INTERRUPTED, interrupt_qualification="authoritative_execution_interrupted_and_native_state",
                        persisted_first_network_forwards=8, previous_partial_second_attempt_forwards="not_measured_do_not_infer_zero")
                    result["evidence_files"] = {name: loop.file_identity(root / name) for name in EVIDENCE_NAMES}
                else:
                    verify_first_unchanged(output, chain_id, frozen)
                    history, timing = transport.execute_graph(server, graph, root / "generation", check, timeout=args.timeout)
                    report = transport.preview_report(history, "50")
                    transport.write_json(root / "loop-report.json", report)
                    result["resume"] = validate_resume(report, frozen, args.profile)
                    validate_final_native(output, chain_id, report)
                    verify_first_unchanged(output, chain_id, frozen)
                    media = inside(output, report["final_video_path"])
                    result["media"] = loop.audit_media(media, root)
                    if result["media"]["sha256"] != report.get("final_video_sha256"):
                        raise RuntimeError("Final media checksum differs")
                    if any(loop.file_identity(previous / name) != identity for name, identity in immutable_previous.items()):
                        raise RuntimeError("Previous terminal/expected evidence changed")
                    result.update(status=RESUMED, timing=timing, interrupted_artifact=str(previous), interrupt_qualification="new_process_native_resume")
            check()
            if frozen_sources(project) != source:
                raise RuntimeError("Final frozen sources changed")
            for asset in [original_image, copied, *expected["assets"]]:
                observed = loop.file_identity(Path(asset["path"]))
                if any(observed[key] != asset[key] for key in ("path", "bytes", "mtime_ns", "sha256")):
                    raise RuntimeError("Final asset SHA changed")
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
    parser.add_argument("--profile", choices=loop.PROFILES, default="trained_vsa_exp")
    parser.add_argument("--reference", type=Path, default=loop.REFERENCE)
    parser.add_argument("--chain-id")
    parser.add_argument("--resume-from", type=Path, help="ONLY this exact verified native interrupted artifact; new process")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    project, core, root = Path(__file__).resolve().parents[1], args.core.resolve(strict=True), args.root.resolve()
    if root.exists() or root == project / "artifacts" or not root.is_relative_to(project / "artifacts") or not 1 <= args.port <= 65535 or args.timeout <= 0:
        raise ValueError("Use a new dedicated artifact directory, valid port and positive timeout")
    root.mkdir(parents=True)
    with (root / "controller.stdout.log").open("x", encoding="utf8") as stdout, (root / "controller.stderr.log").open("x", encoding="utf8") as stderr:
        with redirect_stdout(loop._Tee(sys.stdout, stdout)), redirect_stderr(loop._Tee(sys.stderr, stderr)):
            try:
                run(args, project, core, root)
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
