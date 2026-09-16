"""CPU-only controller oracles; fixture bytes are not GPU/media qualification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def tool():
    path = Path(__file__).resolve().parents[1] / "tools/run_fast_h3_v2_interrupt_probe.py"
    spec = importlib.util.spec_from_file_location("v2_interrupt_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(t, status="interrupted"):
    return {"schema": 1, "format": "minimax_h3_t8_in_node_loop_effects", "chain_id": "own",
            "status": status, "segment_count": 2, "accepted_count": 1,
            "current_segment_index": 1, "manifest_revision": 1,
            "contract_sha256": "a"*64, "final_video_path": ""}


def audit(t, profile="trained_vsa_exp", index=0):
    dual = {"low_reused": False, "high_reused": False,
            "low_context": {"path": "candidates/segment_00000/first/low.context.safetensors"}}
    rungs = [999, 874, 749, 624, 500, 375, 250, 125, 0]
    for name, start, end in (("first_pass", 0, 4), ("second_pass", 4, 8)):
        sparse = profile == "trained_vsa_exp" and name == "second_pass"
        dual[name] = {"nfe": 4, "completed_network_forwards": 4,
            "sigmas": [10*(r/1000)/(1+9*(r/1000)) for r in rungs[start:end+1]],
            "schedule": {"profile": profile, "stage_start": start, "stage_end": end,
                "video_shift": 10., "audio_shift": 3., "rungs": rungs[:-1]},
            "memory_composition": {"kind": "t8_h3_memory", "head_chunks": 4, "ffn_settings": [2, 4096]},
            "backend": {"fasth3_v2": {"profile": profile, "head_chunks": 4,
                "counts": {"vsa": 200 if sparse else 0, "dense": 0 if sparse else 200},
                "actual_vsa_dispatched": sparse, "dense_reasons": {} if sparse else {"below_min_tokens": 200}}},
            "prompt_relay_execution": {"completed_forwards": 4, "routed_attention_calls": 800}
                if profile == "dense_compat_exp" else None}
    value = {"schema": 1, "format": "minimax_h3_t8_in_node_loop_effects_segment",
        "contract_sha256": "a"*64, "segment_index": index, "candidate_id": "first" if index == 0 else "second",
        "sampling_plan": {"dual_model": dual}}
    value["audit_sha256"] = t.digest(value)
    return value


def history():
    return {"status": {"completed": False, "messages": [["execution_interrupted", {"prompt_id": "owned-prompt"}]]}}


def events():
    return [{"type": "execution_interrupted", "data": {"prompt_id": "owned-prompt"}}]


def request():
    return {"method": "POST", "endpoint": "/interrupt", "http_status": 200,
            "prompt_id": "owned-prompt", "owner_pid": 111}


def storage(tmp_path, t, profile="trained_vsa_exp"):
    output = tmp_path / "output"
    native = t.chain_root(output, "own")
    native.mkdir(parents=True)
    names = ["accepted/first.mp4", "accepted/first.context.safetensors",
             "candidates/segment_00000/first/first.mp4", "candidates/segment_00000/first/first.context.safetensors",
             "candidates/segment_00000/first/low.context.safetensors"]
    for name in names:
        path = native / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU-fixture-not-real-media" if name.endswith(".mp4") else b"CPU-fixture-not-real-latent")
    entry = {"index": 0, "candidate_id": "first", "video_path": names[0], "context_path": names[1],
             "video_sha256": t.loop.file_identity(native / names[0])["sha256"],
             "context_sha256": t.loop.file_identity(native / names[1])["sha256"]}
    candidate = {**entry, "schema": 1, "status": "candidate", "chain_id": "own", "frame_count": 124,
        "fps": 24, "width": 512, "height": 768, "is_final_segment": False,
        "video_path": names[2], "context_path": names[3]}
    t.transport.write_json(native / "manifest.json", {"schema": 2, "format": "minimax_h3_t8_accepted_manifest",
        "chain_id": "own", "revision": 1, "segments": [entry]})
    t.transport.write_json(native / t.STATE_NAME, state(t))
    t.transport.write_json(native / "candidates/segment_00000/first/candidate.json", candidate)
    t.transport.write_json(native / "candidates/segment_00000/first/effects_audit.json", audit(t, profile))
    stage_root = native / "dual_stages/segment_00000/base"
    stage_root.mkdir(parents=True)
    for stage in ("low_x0", "high_input", "high_output"):
        tensor = stage_root / (stage + ".safetensors")
        tensor.write_bytes(b"CPU-storage-oracle-only")
        t.transport.write_json(stage_root / (stage + ".json"), {"schema": 1, "stage": stage,
            "contract": {"job": "a"*64}, "tensor_file": tensor.name,
            "tensor_sha256": t.loop.file_identity(tensor)["sha256"]})
    return output, native


@pytest.mark.parametrize("change", [{"accepted_count": 0}, {"accepted_count": 2}, {"status": "complete"},
    {"current_segment_index": 0}, {"chain_id": "foreign"}, {"segment_count": 6}, {"accepted_count": True},
    {"contract_sha256": "label"}, {"final_video_path": "existing.mp4"}])
def test_only_exact_running_or_interrupted_native_state_can_qualify(change):
    t = tool()
    value = {**state(t), **change}
    with pytest.raises(RuntimeError):
        t.validate_state(value, "own", "interrupted")


@pytest.mark.parametrize("path", ["../other", "F:/outside", "", "one\\two", "./one", "one//two"])
def test_unsafe_native_evidence_paths_fail_closed(tmp_path, path):
    t = tool()
    with pytest.raises((ValueError, FileNotFoundError)):
        t.inside(tmp_path, path)


@pytest.mark.parametrize("profile", ["trained_vsa_exp", "dense_compat_exp"])
def test_first_real_signature_storage_identity_and_four_plus_four(tmp_path, profile):
    t = tool()
    output, _ = storage(tmp_path, t, profile)
    frozen = t.first_evidence(output, "own", state(t), profile)
    assert frozen["persisted_first_network_forwards"] == 8
    assert [s["completed_network_forwards"] for s in frozen["dispatch"]] == [4, 4]
    t.verify_first_unchanged(output, "own", frozen)


@pytest.mark.parametrize("change", ["signature", "stage_count", "clock", "memory", "dispatch", "second_cache", "first_bytes", "candidate", "receipt"])
def test_interrupted_evidence_cannot_hide_mismatch_or_persisted_second_counts(tmp_path, change):
    t = tool()
    output, native = storage(tmp_path, t)
    path = native / "candidates/segment_00000/first/effects_audit.json"
    value = t.read_json(path)
    stage = value["sampling_plan"]["dual_model"]["second_pass"]
    if change == "signature":
        value["candidate_id"] = "tampered"
    elif change == "stage_count":
        stage["completed_network_forwards"] = 3
    elif change == "clock":
        stage["sigmas"][0] = 1.
    elif change == "memory":
        stage["memory_composition"]["head_chunks"] = 1
    elif change == "dispatch":
        stage["backend"]["fasth3_v2"]["actual_vsa_dispatched"] = False
    elif change == "second_cache":
        second = native / "dual_stages/segment_00001/base/low_x0.json"
        second.parent.mkdir(parents=True)
        second.write_text("{}", encoding="utf8")
    elif change == "first_bytes":
        (native / "accepted/first.mp4").write_bytes(b"changed")
    elif change == "candidate":
        candidate = native / "candidates/segment_00000/first/candidate.json"
        payload = t.read_json(candidate)
        payload["chain_id"] = "foreign"
        candidate.write_text(json.dumps(payload), encoding="utf8")
    else:
        (native / "dual_stages/segment_00000/base/high_output.safetensors").write_bytes(b"changed")
    if change not in ("signature", "second_cache", "first_bytes", "candidate", "receipt"):
        value.pop("audit_sha256")
        value["audit_sha256"] = t.digest(value)
    path.write_text(json.dumps(value), encoding="utf8")
    with pytest.raises(RuntimeError):
        t.first_evidence(output, "own", state(t), "trained_vsa_exp")


@pytest.mark.parametrize("change", ["no_request", "other_prompt", "success", "history", "completed", "multiple"])
def test_real_targeted_request_and_both_authoritative_interrupt_terminals_required(change):
    t = tool()
    saved_history, saved_events, saved_request = history(), events(), request()
    if change == "no_request":
        saved_request = {}
    elif change == "other_prompt":
        saved_request["prompt_id"] = "foreign"
    elif change == "success":
        saved_events[0]["type"] = "execution_success"
    elif change == "history":
        saved_history["status"]["messages"] = []
    elif change == "completed":
        saved_history["status"]["completed"] = True
    else:
        saved_events.append({"type": "execution_error", "data": {"prompt_id": "owned-prompt"}})
    with pytest.raises(RuntimeError):
        t.require_interrupted_history(saved_history, saved_events, {"prompt_id": "owned-prompt"}, saved_request)
    t.require_interrupted_history(history(), events(), {"prompt_id": "owned-prompt"}, request())


def test_watcher_posts_only_after_native_one_accepted_and_never_stops_a_fake_api(monkeypatch, tmp_path):
    t = tool()
    output, native = storage(tmp_path, t)
    generation = tmp_path / "generation"
    generation.mkdir()
    t.transport.write_json(generation / "submission.json", {"prompt_id": "owned-prompt"})
    seen = []
    monkeypatch.setattr(t, "post_owned_interrupt", lambda server, prompt_id: seen.append(prompt_id) or request())
    watcher = t.InterruptWatcher(object(), generation, output, "own")
    watcher.check()
    assert seen == []  # already interrupted is not a new actual request
    (native / t.STATE_NAME).write_text(json.dumps(state(t, "running")), encoding="utf8")
    watcher.check()
    watcher.check()
    assert seen == ["owned-prompt"] and watcher.sent
    assert t.read_json(tmp_path / "native-trigger-state.json")["status"] == "running"
    t.validate_trigger(t.read_json(tmp_path / "native-trigger-state.json"), state(t),
                       t.read_json(tmp_path / "interrupt-request.json"), "own")


def test_post_uses_verified_own_port_target_and_handles_empty_http200(monkeypatch):
    t = tool()
    seen = []
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, **kwargs):
            assert self.trust_env is False
            seen.append((url, kwargs))
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    import requests
    monkeypatch.setattr(requests, "Session", Session)
    server = SimpleNamespace(url="http://127.0.0.1:8208", process=SimpleNamespace(pid=111),
        assert_port_owner=lambda: seen.append("verified"),
        request=lambda method, endpoint: {"queue_running": [[0, "owned-prompt"]]})
    assert t.post_owned_interrupt(server, "owned-prompt") == request()
    assert seen[0] == "verified" and seen[1][1]["json"] == {"prompt_id": "owned-prompt"}
    server.request = lambda *args: {"queue_running": [[0, "foreign"]]}
    with pytest.raises(RuntimeError, match="sole running"):
        t.post_owned_interrupt(server, "owned-prompt")


@pytest.mark.parametrize("change", [None, "first_audit", "low_reused", "high_reused", "final_cache"])
def test_resume_distinguishes_persisted_first8_and_new_second8(change):
    t = tool()
    first, second = audit(t), audit(t, index=1)
    report = {"status": "complete", "segment_count": 2, "accepted_count": 2, "contract_sha256": "a"*64,
        "segment_audits": [first, second], "resume_action": "generated_or_resumed_then_completed"}
    frozen = {"audit": deepcopy(first)}
    if change == "first_audit":
        report["segment_audits"][0]["extra"] = "changed"
        report["segment_audits"][0].pop("audit_sha256")
        report["segment_audits"][0]["audit_sha256"] = t.digest(report["segment_audits"][0])
    elif change in ("low_reused", "high_reused"):
        second["sampling_plan"]["dual_model"][change] = True
        second.pop("audit_sha256")
        second["audit_sha256"] = t.digest(second)
    elif change == "final_cache":
        report["resume_action"] = "returned_verified_existing_final"
    if change:
        with pytest.raises(RuntimeError):
            t.validate_resume(report, frozen, "trained_vsa_exp")
    else:
        result = t.validate_resume(report, frozen, "trained_vsa_exp")
        assert (result["persisted_first_network_forwards"], result["new_second_network_forwards"],
                result["final_persisted_successful_network_forwards"]) == (8, 8, 16)
        assert result["previous_partial_second_attempt_forwards"] != 0


@pytest.mark.parametrize("status", ["complete", "failed", "mechanical_8s_AV_stage_contract_pass_human_review_pending"])
def test_resume_refuses_completed_failed_or_other_controller_evidence(tmp_path, status):
    t = tool()
    project = tmp_path / "project"
    previous = project / "artifacts/previous"
    previous.mkdir(parents=True)
    t.transport.write_json(previous / "terminal.json", {"schema": t.SCHEMA, "status": status})
    t.transport.write_json(previous / "expected.json", {})
    with pytest.raises(ValueError, match="verified interrupted"):
        t.previous_evidence(previous, project, project / "artifacts/new")


def test_first_accepted_context_and_cache_bytes_must_survive_resume(tmp_path):
    t = tool()
    output, native = storage(tmp_path, t)
    frozen = t.first_evidence(output, "own", state(t), "trained_vsa_exp")
    (native / "accepted/first.context.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes/context"):
        t.verify_first_unchanged(output, "own", frozen)


def previous_fixture(tmp_path, t):
    project = tmp_path / "project"
    previous = project / "artifacts/previous"
    previous.mkdir(parents=True)
    output, _ = storage(previous, t)
    input_scope = previous / "input"
    input_scope.mkdir()
    generation = previous / "generation"
    generation.mkdir()
    trigger = state(t, "running")
    saved = {"generation/history.json": history(), "generation/submission.json": {"prompt_id": "owned-prompt"},
             "interrupt-request.json": {**request(), "trigger_state_sha256": t.digest(trigger)},
             "native-interrupted-state.json": state(t), "native-trigger-state.json": trigger,
             "generation/timing.json": {"terminal": "execution_interrupted", "complete_uncached_graph": False},
             "first-media-probe.json": {"returncode": 0}, "first-media-strict-decode.json": {"returncode": 0}}
    for name, value in saved.items():
        t.transport.write_json(previous / name, value)
    (generation / "events.jsonl").write_text(json.dumps(events()[0]) + "\n", encoding="utf8")
    expected = {"pilot_graphs": {"v2_loop": {"8": {"inputs": {"chain_id": "own"}}}}}
    t.transport.write_json(previous / "expected.json", expected)
    t.transport.write_json(previous / "generation/prompt.json", expected["pilot_graphs"]["v2_loop"])
    frozen = t.first_evidence(output, "own", state(t), "trained_vsa_exp")
    t.transport.write_json(previous / "first-segment-evidence.json", frozen)
    receipt = {"schema": t.SCHEMA, "status": t.INTERRUPTED, "human_review": "pending",
        "cleanup_errors": [], "server_stop": {"owned_children_remaining": []}, "expected_sha256": t.digest(expected),
        "output_directory": str(output), "input_directory": str(input_scope), "profile": "trained_vsa_exp",
        "evidence_files": {name: t.loop.file_identity(previous / name) for name in t.EVIDENCE_NAMES}}
    t.transport.write_json(previous / "terminal.json", receipt)
    return project, previous


@pytest.mark.parametrize("change", [None, "expected", "old_history", "output_escape", "missing_media", "state_complete", "cleanup"])
def test_exact_previous_interrupted_artifact_only_and_immutable_constraints(tmp_path, change):
    t = tool()
    project, previous = previous_fixture(tmp_path, t)
    if change == "expected":
        (previous / "expected.json").write_text("{}", encoding="utf8")
    elif change == "old_history":
        (previous / "generation/history.json").write_text("{}", encoding="utf8")
    elif change in ("output_escape", "cleanup"):
        path = previous / "terminal.json"
        payload = t.read_json(path)
        if change == "output_escape":
            payload["output_directory"] = str(tmp_path)
        else:
            payload["cleanup_errors"] = ["owned child did not exit"]
        path.write_text(json.dumps(payload), encoding="utf8")
    elif change == "missing_media":
        native = t.chain_root(previous / "output", "own")
        (native / "accepted/first.mp4").unlink()
    elif change == "state_complete":
        native = t.chain_root(previous / "output", "own")
        payload = {**state(t), "status": "complete"}
        (native / t.STATE_NAME).write_text(json.dumps(payload), encoding="utf8")
    if change:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            t.previous_evidence(previous, project, project / "artifacts/new")
    else:
        terminal_before = (previous / "terminal.json").read_bytes()
        result = t.previous_evidence(previous, project, project / "artifacts/new")
        assert result[0] == previous and result[5] == "own"
        assert (previous / "terminal.json").read_bytes() == terminal_before


def test_final_native_state_and_manifest_must_match_successful_report(tmp_path):
    t = tool()
    output, native = storage(tmp_path, t)
    report = {**state(t), "status": "complete", "accepted_count": 2, "current_segment_index": None,
        "manifest_revision": 2, "final_video_path": "final.mp4", "final_video_sha256": "b"*64,
        "segment_audits": [audit(t), audit(t, index=1)]}
    (native / t.STATE_NAME).write_text(json.dumps(report), encoding="utf8")
    manifest = t.read_json(native / "manifest.json")
    manifest["revision"] = 2
    manifest["segments"].append({"index": 1, "candidate_id": "second"})
    (native / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    t.validate_final_native(output, "own", report)
    manifest["segments"][1]["candidate_id"] = "foreign"
    (native / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(RuntimeError, match="signed audits"):
        t.validate_final_native(output, "own", report)


def test_cpu_controller_only_queues_registration_validator_never_gpu_or_interrupt(monkeypatch, tmp_path):
    from PIL import Image
    from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2DualModelLongVideoEXPT8
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
    t = tool()
    project, core = tmp_path / "project", tmp_path / "core"
    root = project / "artifacts/new"
    root.mkdir(parents=True)
    core.mkdir()
    reference = tmp_path / "reference.png"
    Image.new("RGB", (20, 30), "red").save(reference)
    info = {name: {"input": {"required": {}}} for name in ("UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "PreviewAny", t.loop.LOWVRAM, t.loop.FFN)}
    for cls in (MiniMaxH3FastH3V2DualModelLongVideoEXPT8, MiniMaxH3PromptRelayPlanT8Advanced):
        info[cls.define_schema().node_id] = cls.GET_NODE_INFO_V1()
    class Server:
        stop_receipt = None
        def __init__(self, *args):
            assert args[2] is True and args[3] == 0
            self.args = args
        def start(self):
            command = t.transport.server_command(*self.args)
            assert command.count("--vram-headroom") == 1
            assert command[command.index("--vram-headroom")+1] == "4"
            assert command[command.index("--input-directory")+1] == str(root / "input")
            assert command[command.index("--output-directory")+1] == str(root / "output")
            assert "--cpu" in command and "--use-pytorch-cross-attention" in command
        def request(self, method, endpoint):
            assert (method, endpoint) == ("GET", "/object_info")
            return info
        def stop(self):
            self.stop_receipt = {"owned_children_remaining": []}
    queued = []
    def execute(server, graph, folder, check, **kwargs):
        assert graph["90"]["class_type"] == "T8ProgressiveEnvironmentAudit"
        frozen = json.loads(graph["90"]["inputs"]["expected_json"])
        queued.append(frozen)
        check()
        return {"outputs": {"91": {"text": [json.dumps({"status": "pass"})]}}}, {}
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU mode must not access GPU or interrupt")
    monkeypatch.setattr(t.transport, "OwnedServer", Server)
    monkeypatch.setattr(t.transport, "wait_ready", lambda *args: None)
    monkeypatch.setattr(t.transport, "execute_graph", execute)
    monkeypatch.setattr(t.loop, "verify_core_source", lambda core: {"core_root": str(core)})
    monkeypatch.setattr(t.loop, "probe_resource_config", lambda *args: {})
    monkeypatch.setattr(t, "frozen_sources", lambda *args: {"unchanged": "sha"})
    monkeypatch.setattr(t.loop, "model_identity", forbidden)
    monkeypatch.setattr(t.loop, "NvmlResourceReader", forbidden)
    monkeypatch.setattr(t.loop, "SerialProbeLease", forbidden)
    monkeypatch.setattr(t, "post_owned_interrupt", forbidden)
    args = SimpleNamespace(cpu=True, gpu=False, resume_from=None, reference=reference,
        chain_id="own", profile="trained_vsa_exp", port=8208, timeout=10)
    t.run(args, project, core, root)
    terminal = t.read_json(root / "terminal.json")
    assert terminal["status"] == "live_Core_graph_validation_pass_no_inference_no_interrupt"
    assert len(queued) == 2 and not queued[0]["pilot_graphs"]
    assert queued[1]["pilot_graphs"]["v2_loop"]["8"]["inputs"]["total_duration_seconds"] == 8.
    assert queued[1]["owned_reference"]["path"].startswith(str(root))
    assert queued[1]["assets"] == []
    assert queued[1]["runtime_options"] == {"reserve_vram_gib": 5, "headroom_gib": 4}
    assert list(core.iterdir()) == []  # no Core reference imports or writes
