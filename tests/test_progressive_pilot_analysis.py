from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.progressive_pilot_analysis import compare_runs, load_run, validate_timeline, verify_execution_report


def test_legacy_runs_do_not_acquire_fabricated_allocator_measurements(tmp_path):
    from tools.progressive_pilot_analysis import load_allocator_evidence
    assert load_allocator_evidence(tmp_path, {}, {"sources": {"old.py": "identity"}}, {}) is None


def test_new_probe_source_requires_allocator_artifact_not_optional_zero(tmp_path):
    from tools.progressive_pilot_analysis import load_allocator_evidence
    with pytest.raises(FileNotFoundError):
        load_allocator_evidence(tmp_path, {}, {"sources": {"tools/progressive_memory_metrics.py": "identity"}}, {})


@pytest.mark.parametrize("fault", [None, "history", "begin", "terminal", "uuid"])
def test_allocator_artifact_and_actual_history_must_agree(tmp_path, fault):
    import os
    from tools.progressive_memory_metrics import AllocatorMemorySession
    from test_progressive_memory_metrics import Reader
    from tools.progressive_pilot_analysis import load_allocator_evidence
    reader = Reader("cudaMallocAsync")
    probe = AllocatorMemorySession(run_id=tmp_path.name, device_type="cuda", device_index=0, reader=reader)
    begin = probe.begin()
    report = probe.finish()
    terminal = {"resource_guard": {"gpu_uuid": "fixture-only"}, "allocator_interval": {
        "file": "allocator-memory.json", "status": report["status"], "counter_scope": report["counter_scope"]}}
    end = deepcopy(report)
    if fault == "history":
        end["run_id"] = "different"
    elif fault == "begin":
        begin["run_id"] = "different"
    elif fault == "terminal":
        terminal.pop("allocator_interval")
    elif fault == "uuid":
        terminal["resource_guard"]["gpu_uuid"] = "another-device"
    (tmp_path / "allocator-memory.json").write_text(json.dumps(report), encoding="utf-8")
    for action, payload in (("begin", begin), ("finish", end)):
        folder = tmp_path / ("allocator-" + action)
        folder.mkdir()
        (folder / "history.json").write_text(json.dumps({"status": {"completed": True},
            "outputs": {"2": {"text": [json.dumps(payload)]}}}), encoding="utf-8")
    args = (tmp_path, terminal, {"sources": {"tools/progressive_memory_metrics.py": "identity"}}, {"pid": os.getpid()})
    if fault:
        with pytest.raises(ValueError):
            load_allocator_evidence(*args)
    else:
        assert load_allocator_evidence(*args) == report


def media_values():
    probe = {"streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1024, "height": 512,
         "avg_frame_rate": "24/1", "r_frame_rate": "24/1", "time_base": "1/12288"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2,
         "start_time": "0", "duration": "3.041667"}]}
    frames = [{"pts": i*512, "width": 1024, "height": 512} for i in range(73)]
    return probe, frames


def validate(probe, frames):
    return validate_timeline(probe, frames, width=1024, height=512, count=73, fps=24)


def test_exact_video_pts_and_codec_rounded_audio_are_accepted():
    report = validate(*media_values())
    assert report["frames"] == 73 and report["all_video_pts_checked"]


@pytest.mark.parametrize("fault", ["missing_frame", "duplicate_pts", "offset_pts", "changed_dimensions", "no_audio",
                                  "extra_audio", "rate", "short_audio", "audio_offset"])
def test_metadata_and_full_frame_timeline_defects_are_rejected(fault):
    probe, frames = media_values()
    if fault == "missing_frame":
        frames.pop()
    elif fault == "duplicate_pts":
        frames[20]["pts"] = frames[19]["pts"]
    elif fault == "offset_pts":
        for frame in frames:
            frame["pts"] += 512
    elif fault == "changed_dimensions":
        frames[30]["height"] = 508
    elif fault == "no_audio":
        probe["streams"].pop()
    elif fault == "extra_audio":
        probe["streams"].append(deepcopy(probe["streams"][-1]))
    elif fault == "rate":
        probe["streams"][0]["r_frame_rate"] = "30/1"
    elif fault == "short_audio":
        probe["streams"][1]["duration"] = "2.0"
    elif fault == "audio_offset":
        probe["streams"][1]["start_time"] = "0.08"
    with pytest.raises(ValueError):
        validate(probe, frames)


def execution_values(progressive=False):
    graph = {"10": {"inputs": {"seed": 123}}}
    sampler = {"actual_network_forwards": 8, "cfg": 1., "seed": 123}
    if progressive:
        sampler = {"counts": {key: {"low": 6, "high": 2} for key in ("callbacks", "actual_forwards", "apply_model_calls")},
                   "pixel_anchor": False, "highres_tiling": False, "noise": {"low_seed": 123}}
    reports = {"sampler": sampler, "lora": {"status": "applied", "selected_strength": 1.,
        "applied_patch_count": 259, "missed_patch_target_count": 0, "observed_model_patch_targets": 259}}
    return graph, reports


@pytest.mark.parametrize("progressive", [False, True])
def test_actual_network_counts_and_lora_are_checked(progressive):
    graph, reports = execution_values(progressive)
    case = "T2VA_progressive6plus2" if progressive else "T2VA_native8"
    verify_execution_report(case, graph, reports)
    if progressive:
        reports["sampler"]["counts"]["actual_forwards"]["high"] = 1
    else:
        reports["sampler"]["actual_network_forwards"] = 7
    with pytest.raises(ValueError):
        verify_execution_report(case, graph, reports)
    graph, reports = execution_values(progressive)
    reports["lora"]["missed_patch_target_count"] = 51
    with pytest.raises(ValueError, match="EMA"):
        verify_execution_report(case, graph, reports)


def pair_values():
    a = {"root": "baseline", "terminal": {"case": "T2VA_native8", "thermal_scope": "cold",
                                          "resource_guard": {"gpu_uuid": "same"}},
         "assets": [{"sha256": "same"}], "environment": {"core": "same", "sources": "same"},
         "graph": {"10": {"inputs": {"seed": 123}}, "18": {"inputs": {"filename_prefix": "a"}}},
         "live": {"torch_version": "same", "torch_cuda": "same", "device": "cuda:0"},
         "reports": {"conditioning": {"sha": "same"}, "decode": {"timings": [{"stage": "video_vae_decode", "seconds": 10}]}},
         "media": {"timeline": {key: 1 for key in ("width", "height", "frames", "fps", "sample_rate", "channels")}},
         "timing": {"elapsed_seconds": 100}}
    b = deepcopy(a)
    b["root"] = "candidate"
    b["terminal"]["case"] = "T2VA_progressive6plus2"
    b["timing"]["elapsed_seconds"] = 80
    return a, b


def test_pair_comparison_does_not_claim_quality():
    result = compare_runs(*pair_values())
    assert result["saved_fraction"] == pytest.approx(.2)
    assert "unreviewed" in result["status"]


@pytest.mark.parametrize("field", ["assets", "sources", "condition", "seed", "temperature", "uuid", "frames", "task"])
def test_pair_rejects_different_workload_or_runtime(field):
    a, b = pair_values()
    if field == "assets":
        b["assets"] = []
    elif field == "sources":
        b["environment"]["sources"] = "changed"
    elif field == "condition":
        b["reports"]["conditioning"] = "changed"
    elif field == "seed":
        b["graph"]["10"]["inputs"]["seed"] = 456
    elif field == "temperature":
        b["terminal"]["thermal_scope"] = "warm"
    elif field == "uuid":
        b["terminal"]["resource_guard"]["gpu_uuid"] = "other"
    elif field == "frames":
        b["media"]["timeline"]["frames"] = 72
    elif field == "task":
        b["terminal"]["case"] = "I2VA_progressive6plus2"
    with pytest.raises(ValueError):
        compare_runs(a, b)


def test_completed_cpu_smoke_cannot_be_used_as_a_gpu_benchmark(tmp_path):
    (tmp_path / "terminal.json").write_text(json.dumps({"mode": "cpu-smoke", "status": "cpu_live_transport_pass_no_models_or_gpu"}))
    with pytest.raises(ValueError, match="GPU"):
        load_run(tmp_path)


def test_actual_cpu_smoke_receipt_is_rejected_for_performance():
    root = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909/live-controller-cpu-v4"
    if not root.is_dir():
        pytest.skip("Local development receipt is not in a released source checkout")
    with pytest.raises(ValueError, match="GPU"):
        load_run(root)
