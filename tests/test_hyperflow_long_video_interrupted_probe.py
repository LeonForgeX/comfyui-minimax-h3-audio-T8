"""CPU-only synthetic fixture for P7 interrupted-copy staging."""
from __future__ import annotations

import hashlib
import json

import pytest

from tools import hyperflow_long_video_interrupted_resume_probe as probe


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content), encoding="utf-8")


def _stage(folder, name, contract, content):
    key = name + "-" + hashlib.sha256(probe.canonical(contract).encode()).hexdigest()
    tensor_name = key + "-fake.safetensors"
    _write(folder / tensor_name, content)
    receipt = {"stage": name, "contract": contract,
               "tensor_file": tensor_name, "tensor_sha256": hashlib.sha256(content).hexdigest()}
    _write(folder / (key + ".json"), receipt)
    return receipt


def test_interrupted_copy_retains_only_parent_and_reusable_low_high_input(tmp_path):
    source = tmp_path / "source"
    state = {"chain_id": "fake", "contract_sha256": "j" * 64, "status": "complete",
             "accepted_count": 2, "manifest_revision": 2, "current_segment_index": None,
             "final_video_path": str(source / "assembled/final.mp4"), "final_video_sha256": "f" * 64}
    _write(source / "in_node_loop_effects_state.json", state)
    first_movie = b"accepted-first-video"
    first_hash = hashlib.sha256(first_movie).hexdigest()
    parent = {"candidate_id": "c0", "seed": 7, "video_path": "accepted/segment_00000.mp4",
              "video_sha256": first_hash}
    _write(source / parent["video_path"], first_movie)
    _write(source / "manifest.json", {"revision": 2, "segments": [parent, {"index": 1}]})
    _write(source / "manifest.json.bak", {"revision": 1, "segments": [parent]})
    _write(source / "candidates/segment_00000/c0/effects_audit.json", {
        "sampling_plan": {"dual_model": {"low_context": {"sha256": "p" * 64}}}})
    _write(source / "candidates/segment_00001/c1/candidate.mp4", b"old child")
    _write(source / "accepted/segment_00001.mp4", b"old accepted child")
    _write(source / "assembled/final.mp4", b"old final")
    _write(source / "last_execution_report.json", {})
    _write(source / "manifest.lock.v2", b"old lock")
    _write(source / "in_node_loop.lock", b"old lock")
    stage_dir = source / "hyperflow_stages/segment_00001/base-c1"
    low_contract = {
        "schema": probe.RECIPE, "job": "j" * 64, "segment": 1,
        "parent_high": "c0", "parent_revision": 1, "parent_low_sha256": "p" * 64,
        "seed_low": 8, "seed_high": 9, "intervals": [[0, 4], [4, 8]],
        "low_picture_context": {"name": "accepted_picture_low_context_v1",
                                "source_media_sha256": first_hash,
                                "source_frame_interval": [85, 124],
                                "source_segment_index": 0},
    }
    low = _stage(stage_dir, "low_x0", low_contract, b"low tensor")
    high_contract = {**low_contract, "low_tensor_sha256": low["tensor_sha256"]}
    _stage(stage_dir, "high_input", high_contract, b"high input tensor")
    _stage(stage_dir, "high_output", high_contract, b"old high output")

    gate = probe.preflight(source, state)
    assert gate["stage_contract_key_verified"]
    dest = tmp_path / "copy" / "output" / "minimax_h3_t8_long_video" / "fake"
    source_before = probe.source_hashes(source)
    retained = probe.prepare_copy(source, dest, state, gate)
    assert probe.immutable_snapshot(dest) == retained
    assert probe.source_hashes(source) == source_before
    assert json.loads((dest / "manifest.json").read_text())["revision"] == 1
    assert json.loads((dest / probe.STATE).read_text())["accepted_count"] == 1
    assert not (dest / "candidates/segment_00001").exists()
    assert not (dest / "accepted/segment_00001.mp4").exists()
    assert not (dest / "assembled").exists()
    assert not list((dest / "hyperflow_stages/segment_00001/base-c1").glob("high_output-*"))
    assert probe.stage_receipt(dest / "hyperflow_stages/segment_00001/base-c1", "low_x0")
    assert probe.stage_receipt(dest / "hyperflow_stages/segment_00001/base-c1", "high_input")

    low_file, low_receipt = probe.stage_receipt(
        dest / "hyperflow_stages/segment_00001/base-c1", "low_x0")
    (low_file.parent / low_receipt["tensor_file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="failed SHA-256"):
        probe.stage_receipt(low_file.parent, "low_x0")
