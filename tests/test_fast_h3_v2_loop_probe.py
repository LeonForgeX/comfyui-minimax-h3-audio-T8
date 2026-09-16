from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _tool():
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("v2_loop_probe_test", tools / "run_fast_h3_v2_loop_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _info(tool):
    from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2DualModelLongVideoEXPT8
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
    info = {name: {"input": {"required": {}}} for name in (
        "UNETLoader", "CLIPLoader", "VAELoader", "LoadImage", "PreviewAny", tool.LOWVRAM, tool.FFN, tool.LORA)}
    # Real production schemas, not guessed inherited widget/default names.
    for cls in (MiniMaxH3FastH3V2DualModelLongVideoEXPT8, MiniMaxH3PromptRelayPlanT8Advanced):
        info[cls.define_schema().node_id] = cls.GET_NODE_INFO_V1()
    return info


@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
def test_exact_two_window_full_student_branches_and_profile_boundary(profile):
    tool = _tool()
    info = _info(tool)
    original = deepcopy(info)
    graph = tool.build_graph(info, first_frame="portrait.png", chain_id="new_v2_exp", profile=profile)
    assert info == original
    runner = graph["8"]["inputs"]
    assert (runner["low_width"], runner["low_height"], runner["width"], runner["height"]) == (256, 384, 512, 768)
    assert (runner["total_duration_seconds"], runner["render_window_frames"], runner["context_frames"]) == (8., 124, 22)
    assert runner["second_audio_source"] == "auto" and runner["second_audio_strength"] == 0
    assert runner["eav_mode"] == "disabled" and runner["task_type"] == "I2VA"
    from h3_audio_t8_pkg.conditioning import resolve_task_type
    assert resolve_task_type(runner["task_type"], object(), None, False) == "i2va"
    assert runner["first_frame_reuse"] == "segment0_only"
    assert not any("LoRA" in item["class_type"] or "Setup" in item["class_type"] for item in graph.values())
    assert graph["1"]["inputs"]["unet_name"] == graph["2"]["inputs"]["unet_name"] == tool.MODEL
    assert graph["1"] is not graph["2"]
    for key, source in (("21", "1"), ("22", "2")):
        assert graph[key]["inputs"] == {"model": [source, 0], "head_chunks": 4}
    for key, source in (("24", "21"), ("25", "22")):
        assert graph[key]["inputs"] == {"model": [source, 0], "chunks": 2, "seq_threshold": 4096}
    assert graph["50"]["inputs"]["source"] == ["8", 5]
    fields = {**info[tool.NODE]["input"]["required"], **info[tool.NODE]["input"].get("optional", {})}
    assert runner.keys() <= fields.keys()
    assert not ({"coarse_steps", "refine_steps", "first_shift_video", "second_shift_audio"} & runner.keys())
    if profile == "dense_compat_exp":
        assert runner["prompt_relay_mode"] == "apply_exp"
        assert graph["7"]["inputs"]["length"] == 193
        assert graph["7"]["inputs"]["time_ranges"] == "0-43.75\n43.75-87.5\n87.5-100"
        assert "<d>" not in graph["7"]["inputs"]["global_prompt"]
        assert graph["7"]["inputs"]["local_prompts"].count("<d>") == 3
    else:
        assert "7" not in graph and "prompt_relay_plan" not in runner
        assert runner["prompt_relay_mode"] == "disabled"


def test_optional_independent_lora_zero_requires_no_file_and_active_has_own_route():
    tool = _tool()
    graph = tool.build_graph(_info(tool), first_frame="portrait.png", chain_id="lora_test",
        lora1="not_downloaded.safetensors", strength1=0., lora2="style.safetensors", strength2=.5)
    assert "30" not in graph
    assert graph["31"]["inputs"] == {"model": ["2", 0], "lora_name": "style.safetensors", "strength_model": .5}
    assert graph["22"]["inputs"]["model"] == ["31", 0]
    with pytest.raises(ValueError, match="active independent LoRA"):
        tool.build_graph(_info(tool), first_frame="portrait.png", chain_id="bad_lora", strength1=1)


@pytest.mark.parametrize("change", [dict(first_frame="../outside.png"), dict(first_frame="F:/outside.png"),
    dict(first_frame=""), dict(chain_id="../chain"), dict(profile="official_comfy_template_exp"), dict(strength1=float("nan"))])
def test_unknown_or_unsafe_inputs_fail_closed(change):
    tool = _tool()
    kwargs = dict(first_frame="portrait.png", chain_id="safe")
    kwargs.update(change)
    with pytest.raises(ValueError):
        tool.build_graph(_info(tool), **kwargs)


def test_task_reference_import_is_byte_identical_exact_aspect_and_never_overwrites(tmp_path):
    from PIL import Image
    tool = _tool()
    core = tmp_path / "core"
    (core / "input").mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 30), "red").save(source)
    filename, original, copied = tool.import_reference(core, source)
    assert original["sha256"] == copied["sha256"]
    assert (core / "input" / filename).read_bytes() == source.read_bytes()
    assert tool.import_reference(core, source)[2] == copied
    (core / "input" / filename).write_bytes(b"do-not-overwrite")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        tool.import_reference(core, source)
    Image.new("RGB", (20, 20)).save(source)
    with pytest.raises(ValueError, match="aspect ratio"):
        tool.import_reference(core, source)


def _sign(audit):
    unsigned = dict(audit)
    unsigned.pop("audit_sha256", None)
    audit["audit_sha256"] = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _report(profile):
    audits = []
    for index in (0, 1):
        dual = {}
        for name, start, end in (("first_pass", 0, 4), ("second_pass", 4, 8)):
            # The LOW token count can legitimately be below sparse threshold.
            sparse = profile == "trained_vsa_exp" and name == "second_pass"
            dual[name] = {"nfe": 4, "completed_network_forwards": 4,
                "sigmas": [10*(r/1000)/(1+9*(r/1000)) for r in [999, 874, 749, 624, 500, 375, 250, 125, 0][start:end+1]],
                "schedule": {"profile": profile, "stage_start": start, "stage_end": end,
                    "video_shift": 10., "audio_shift": 3., "rungs": [999, 874, 749, 624, 500, 375, 250, 125]},
                "memory_composition": {"kind": "t8_h3_memory", "head_chunks": 4, "ffn_settings": [2, 4096]},
                "backend": {"fasth3_v2": {"profile": profile, "head_chunks": 4,
                    "counts": {"vsa": 50 if sparse else 0, "dense": 0 if sparse else 50},
                    "dense_reasons": {} if sparse else {"below_min_tokens": 50},
                    "actual_vsa_dispatched": sparse}},
                "prompt_relay_execution": {"completed_forwards": 4, "routed_attention_calls": 200}
                    if profile == "dense_compat_exp" else None}
        audit = {"contract_sha256": "job_sha", "segment_index": index, "sampling_plan": {"dual_model": dual}}
        _sign(audit)
        audits.append(audit)
    return {"status": "complete", "segment_count": 2, "accepted_count": 2,
        "contract_sha256": "job_sha", "segment_audits": audits}


@pytest.mark.parametrize("profile", ["dense_compat_exp", "trained_vsa_exp"])
def test_report_checks_actual_four_forwards_each_stage_and_retains_dispatch_reasons(profile):
    tool = _tool()
    result = tool.validate_loop_report(_report(profile), profile)
    assert result["actual_network_forwards"] == 16 and len(result["stages"]) == 4
    assert result["stages"][0]["dispatch"]["dense_reasons"]
    assert "human quality acceptance" in result["scope"]


@pytest.mark.parametrize("field,value", [("nfe", 8), ("completed_network_forwards", 3)])
def test_report_cannot_replace_observed_calls_with_configured_step_count(field, value):
    tool = _tool()
    report = _report("dense_compat_exp")
    report["segment_audits"][1]["sampling_plan"]["dual_model"]["second_pass"][field] = value
    _sign(report["segment_audits"][1])
    with pytest.raises(RuntimeError, match="four completed"):
        tool.validate_loop_report(report, "dense_compat_exp")


@pytest.mark.parametrize("change", ["hash", "profile", "relay", "clock", "memory", "fake_dispatch", "zero_vsa"])
def test_report_detects_wrong_identity_or_silent_backend_change(change):
    tool = _tool()
    report = _report("trained_vsa_exp")
    audit = report["segment_audits"][0]
    stage = audit["sampling_plan"]["dual_model"]["second_pass"]
    if change == "hash":
        audit["candidate_id"] = "tampered"
    elif change == "profile":
        stage["backend"]["fasth3_v2"]["profile"] = "dense_compat_exp"
    elif change == "relay":
        stage["prompt_relay_execution"] = {"routed_attention_calls": 100}
    elif change == "clock":
        stage["sigmas"][0] = 1.
    elif change == "memory":
        stage["memory_composition"]["ffn_settings"] = [1, 4096]
    elif change == "fake_dispatch":
        stage["backend"]["fasth3_v2"]["actual_vsa_dispatched"] = False
    else:
        for signed in report["segment_audits"]:
            for candidate in signed["sampling_plan"]["dual_model"].values():
                candidate["backend"]["fasth3_v2"].update(counts={}, actual_vsa_dispatched=False)
            _sign(signed)
    if change != "hash":
        _sign(audit)
    with pytest.raises(RuntimeError):
        tool.validate_loop_report(report, "trained_vsa_exp")


@pytest.mark.parametrize("change", [None, "frames", "codec", "audio", "duration", "decode"])
def test_independent_media_audit_requires_exact_canvas_frames_fps_audio_and_strict_decode(monkeypatch, tmp_path, change):
    tool = _tool()
    media = tmp_path / "output.mp4"
    media.write_bytes(b"test-only-not-real-media")
    payload = {"streams": [dict(codec_type="video", codec_name="h264", width=512, height=768,
        nb_read_frames="192", avg_frame_rate="24/1", duration="8.0"),
        dict(codec_type="audio", codec_name="aac", duration="8.0")], "format": {"duration": "8.0"}}
    if change == "frames":
        payload["streams"][0]["nb_read_frames"] = "191"
    elif change == "codec":
        payload["streams"][0]["codec_name"] = "hevc"
    elif change == "audio":
        payload["streams"].pop()
    elif change == "duration":
        payload["streams"][1]["duration"] = "7.5"
    def execute(command, **kwargs):
        if command[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        assert "-xerror" in command and "explode" in command
        return SimpleNamespace(returncode=1 if change == "decode" else 0, stdout="", stderr="bad" if change == "decode" else "")
    monkeypatch.setattr(tool.subprocess, "run", execute)
    if change:
        with pytest.raises(RuntimeError):
            tool.audit_media(media, tmp_path)
    else:
        receipt = tool.audit_media(media, tmp_path)
        assert receipt["strict_decode"] is True and receipt["quality_accepted"] is False
        assert len(list(tmp_path.glob("decode-*.json"))) == 3


def test_incomplete_student_is_never_hashed_or_loaded(monkeypatch, tmp_path):
    tool = _tool()
    (tmp_path / "models/diffusion_models").mkdir(parents=True)
    (tmp_path / "models/diffusion_models" / tool.MODEL).write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="incomplete"):
        tool.model_identity(tmp_path)


def test_pending_material_does_not_hide_crash_or_source_identity_error(tmp_path):
    tool = _tool()
    for message in ("Owned server exited", "Probe source identity changed", "Port ownership failed"):
        assert tool.material_validation_failed(tmp_path, RuntimeError(message)) is False
    (tmp_path / "history.json").write_text(json.dumps({"status": {"messages": [["execution_error",
        {"exception_message": "Instrumented pilot graph failed Core validation: missing UNET"}]]}}), encoding="utf-8")
    assert tool.material_validation_failed(tmp_path, RuntimeError("Graph failed")) is True


@pytest.mark.parametrize("pending", [False, True])
def test_cpu_controller_invokes_complete_Core_validator_never_queues_inference_or_reads_gpu(monkeypatch, tmp_path, pending):
    tool = _tool()
    project, core = tmp_path / "project", tmp_path / "core"
    root = project / "artifacts/new"
    root.mkdir(parents=True)
    (core / "input").mkdir(parents=True)
    reference = core / "input/portrait.png"
    reference.write_bytes(b"task-scoped-fixture")
    identity = tool.file_identity(reference)
    info = _info(tool)
    info["UNETLoader"]["input"]["required"] = {"unet_name": [["old.safetensors"] if pending else [tool.MODEL], {}]}
    class Server:
        stop_receipt = None
        def __init__(self, *args):
            assert args[-1] is True
        def start(self):
            pass
        def request(self, method, endpoint):
            assert (method, endpoint) == ("GET", "/object_info")
            return info
        def stop(self):
            self.stop_receipt = {"scope": "owned_only"}
    seen = []
    def execute(server, graph, folder, check, **kwargs):
        # ONLY the probe environment node is queued, never student loaders.
        assert graph["90"]["class_type"] == "T8ProgressiveEnvironmentAudit"
        frozen = json.loads(graph["90"]["inputs"]["expected_json"])
        seen.append(frozen)
        if frozen["pilot_graphs"] and pending:
            raise RuntimeError("Instrumented pilot graph failed Core validation: checkpoint enum missing")
        return {"outputs": {"91": {"text": [json.dumps({"status": "pass"})]}}}, {}
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU validation must never read GPU or checkpoint tensors")
    monkeypatch.setattr(tool.transport, "OwnedServer", Server)
    monkeypatch.setattr(tool.transport, "CORE", core)
    monkeypatch.setattr(tool.transport, "PROJECT", project)
    monkeypatch.setattr(tool.transport, "wait_ready", lambda *args: None)
    monkeypatch.setattr(tool.transport, "execute_graph", execute)
    monkeypatch.setattr(tool, "verify_core_source", lambda core: {"core_root": str(core)})
    monkeypatch.setattr(tool, "frozen_sources", lambda project: {"frozen.py": "test-sha"})
    monkeypatch.setattr(tool, "import_reference", lambda *args: ("portrait.png", identity, identity))
    monkeypatch.setattr(tool, "model_identity", forbidden)
    monkeypatch.setattr(tool, "NvmlResourceReader", forbidden)
    monkeypatch.setattr(tool, "SerialProbeLease", forbidden)
    args = SimpleNamespace(cpu=True, gpu=False, port=8199, resume_from=None, reference=reference,
        chain_id="cpu_schema", profile="dense_compat_exp", lora1=None, lora2=None,
        strength1=0., strength2=0., timeout=10)
    tool.run(args, project, core, root)
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert len(seen) == 2 and seen[0]["pilot_graphs"] == {}
    assert seen[1]["pilot_graphs"]["v2_loop"]["8"]["class_type"] == tool.NODE
    assert terminal["graph_validated"] is not pending
    assert "pending_material" in terminal if pending else "pending_material" not in terminal
    assert terminal["server_stop"] == {"scope": "owned_only"}
