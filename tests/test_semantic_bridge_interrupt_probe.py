"""Synthetic storage negatives; no claim of actual interruption from CPU tests."""
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from tools import run_semantic_bridge_interrupt_probe as probe
from tools.audit_semantic_bridge_loop import digest


def resource_row(stamp, uuid="fixed-gpu"):
    mib = 1024**2
    return dict(monotonic=stamp, gpu_uuid=uuid, gpu_total_bytes=16384*mib,
                gpu_free_bytes=14336*mib, gpu_used_bytes=2048*mib,
                ram_total_bytes=65536*mib, ram_available_bytes=32768*mib)


def test_deliberate_stopped_service_gap_is_not_a_live_sample_gap():
    first = probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(1.)))
    assert first.observe(resource_row(2.)) is None
    second = probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(100.)), first)
    assert second is not first and second.gpu_uuid == first.gpu_uuid
    assert second.observe(resource_row(101.)) is None
    assert second.observe(resource_row(105.)) == "resource_telemetry_invalid_or_stale"
    with pytest.raises(RuntimeError, match="Previous phase failed"):
        probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(106.)), second)


def test_phase_restart_rechecks_startup_resources_and_device_identity():
    first = probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(1.)))
    with pytest.raises(RuntimeError, match="identity changed"):
        probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(100., "other-gpu")), first)
    row = resource_row(101.)
    row["gpu_free_bytes"] = 1024**2
    row["gpu_used_bytes"] = row["gpu_total_bytes"] - row["gpu_free_bytes"]
    with pytest.raises(RuntimeError, match="startup resource guard"):
        probe.start_phase_guard(SimpleNamespace(sample=lambda: row), first)


def test_both_phase_receipts_are_exclusive_and_independent(tmp_path):
    result = {"server_phases": []}
    server = SimpleNamespace(stop=lambda: None, stop_receipt={"pid": 1})
    guard = probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(1.)))
    for phase in ("interrupt", "resume"):
        stage = tmp_path / phase
        stage.mkdir()
        probe.finish_phase(stage, phase, server, None, guard, result)
        saved = json.loads((stage / "progress.json").read_text(encoding="utf8"))
        assert saved["server_phases"][-1]["phase"] == phase
    assert len(result["server_phases"]) == 2


@pytest.mark.parametrize("primary", [False, True])
def test_cleanup_stops_server_and_preserves_primary_error(tmp_path, monkeypatch, primary):
    calls = []
    def failed_close():
        raise RuntimeError("monitor cleanup failed")
    def failed_write(*args):
        raise OSError("receipt cannot be written")
    monkeypatch.setattr(probe.transport, "write_json", failed_write)
    server = SimpleNamespace(stop=lambda: calls.append("stopped"), stop_receipt={"pid": 1})
    monitor = SimpleNamespace(close=failed_close)
    guard = probe.start_phase_guard(SimpleNamespace(sample=lambda: resource_row(1.)))
    result = {"server_phases": []}
    with pytest.raises(RuntimeError, match="original generation error" if primary else "Owned phase cleanup"):
        try:
            if primary:
                raise RuntimeError("original generation error")
        finally:
            probe.finish_phase(tmp_path, "interrupt", server, monitor, guard, result)
    assert calls == ["stopped"] and len(result["cleanup_errors"]) == 2


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def stored(tmp_path):
    root = probe.interrupt.chain_root(tmp_path, "test")
    prefix = "candidates/segment_00000/fixture"
    video, context = root / prefix / "video.mp4", root / prefix / "context.latent"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"synthetic-video-not-real-media")
    context.write_bytes(b"synthetic-context-not-real-latent")
    entry = dict(index=0, candidate_id="fixture", video_path=prefix + "/video.mp4",
                 context_path=prefix + "/context.latent",
                 video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                 context_sha256=hashlib.sha256(context.read_bytes()).hexdigest())
    manifest = dict(schema=2, format="minimax_h3_t8_accepted_manifest",
                    chain_id="test", revision=1, segments=[entry])
    write(root / "manifest.json", manifest)
    receipt = dict(schema="t8_semantic_bridge_v1", encoding_source="t8:segment=0:low",
                   input_sha256="a"*64, output_sha256="b"*64)
    receipt["receipt_sha256"] = digest(receipt)
    write(root / prefix / "effects_audit.json", {"bridge": receipt})
    state = dict(schema=1, format="minimax_h3_t8_in_node_loop_effects",
                 chain_id="test", status="interrupted", segment_count=2,
                 accepted_count=1, current_segment_index=1, manifest_revision=1,
                 contract_sha256="c"*64, final_video_path=None)
    return tmp_path, root, manifest, receipt, state


def complete(stored):
    output, root, manifest, receipt, state = stored
    first = probe.first_snapshot(output, "test", state)
    manifest["segments"].append({"index": 1})
    manifest["revision"] = 2
    write(root / "manifest.json", manifest)
    second = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    second["encoding_source"] = "t8:segment=1:low"
    second["receipt_sha256"] = digest(second)
    write(root / "candidates/segment_00001/fixture/effects_audit.json", {"bridge": second})
    report = dict(status="complete", accepted_count=2, segment_count=2, contract_sha256="c"*64)
    return first, report


def test_storage_resume_requires_unchanged_first_and_new_second_receipt(stored):
    first, report = complete(stored)
    result = probe.verify_resume(stored[0], "test", first, report)
    assert result["first_files_unchanged_including_mtime"]
    assert len(result["bridge_receipts"]) == 2


@pytest.mark.parametrize("mutation", ["content", "mtime", "first_entry", "contract", "count"])
def test_changed_first_or_incomplete_resume_rejected(stored, mutation):
    first, report = complete(stored)
    output, root, manifest, _, _ = stored
    path = root / first["entry"]["video_path"]
    if mutation == "content":
        path.write_bytes(b"different")
    elif mutation == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000))
    elif mutation == "first_entry":
        changed = deepcopy(manifest)
        changed["segments"][0]["index"] = 5
        write(root / "manifest.json", changed)
    elif mutation == "contract":
        report["contract_sha256"] = "d"*64
    else:
        report["accepted_count"] = 1
    with pytest.raises(ValueError):
        probe.verify_resume(output, "test", first, report)


def test_invalid_native_interrupted_state_and_bridge_receipt_rejected(stored):
    output, root, _, receipt, state = stored
    with pytest.raises(RuntimeError):
        probe.first_snapshot(output, "test", {**state, "status": "complete"})
    receipt["input_sha256"] = "changed-without-rehash"
    write(root / "candidates/segment_00000/fixture/effects_audit.json", {"bridge": receipt})
    with pytest.raises(ValueError, match="integrity"):
        probe.first_snapshot(output, "test", state)
