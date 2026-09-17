import json
import hashlib

import pytest

from tools.audit_semantic_bridge_loop import audit, digest, receipts


def test_receipt_digest_and_duplicate_report_copies():
    row = {"schema": "t8_semantic_bridge_v1", "input_sha256": "x"}
    row["receipt_sha256"] = digest(row)
    found = {}
    receipts([row, json.dumps(row)], found)
    assert len(found) == 1
    row["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="integrity"):
        receipts(row, {})


def fixture_run(tmp_path, dual=False):
    (tmp_path / "output").mkdir()
    media = tmp_path / "output/fixture.bin"
    media.write_bytes(b"synthetic audit fixture, not real video")
    terminal = {"status": "mechanical_pass_human_pending", "quality_accepted": False,
        "resume": {"report": {"resume_action": "returned_verified_existing_final"},
                   "latent_and_media_files_unchanged": True},
        "media": {"path": str(media), "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                  "geometry": [896, 448, 192, "24/1"], "strict_decode": True}}
    (tmp_path / "terminal.json").write_text(json.dumps(terminal), encoding="utf8")
    rows = []
    for segment in range(2):
        for stage in range(2 if dual else 1):
            row = {"schema": "t8_semantic_bridge_v1", "sha256": f"model-{stage}",
                   "encoding_source": f"native_h3_long:t2va:segment={segment}:context={bool(segment)}",
                   "input_sha256": f"source-{segment}-{stage}", "output_sha256": f"target-{segment}-{stage}"}
            row["receipt_sha256"] = digest(row)
            rows.append(row)
    (tmp_path / "output/fixture.json").write_text(json.dumps(rows), encoding="utf8")
    return terminal, rows


@pytest.mark.parametrize("dual", [False, True])
def test_audit_counts_distinct_segment_stage_receipts(tmp_path, dual):
    fixture_run(tmp_path, dual)
    report = audit(tmp_path, dual)
    assert len(report["actual_bridge_receipts"]) == (4 if dual else 2)


@pytest.mark.parametrize("fault", ["status", "resume", "frames", "media", "missing_receipt"])
def test_audit_rejects_incomplete_or_changed_evidence(tmp_path, fault):
    terminal, rows = fixture_run(tmp_path)
    if fault == "status":
        terminal["status"] = "failed"
    elif fault == "resume":
        terminal["resume"]["report"]["resume_action"] = "generated_again"
    elif fault == "frames":
        terminal["media"]["geometry"][2] = 193
    elif fault == "media":
        (tmp_path / "output/fixture.bin").write_bytes(b"changed")
    else:
        (tmp_path / "output/fixture.json").write_text(json.dumps(rows[:1]), encoding="utf8")
    (tmp_path / "terminal.json").write_text(json.dumps(terminal), encoding="utf8")
    with pytest.raises(ValueError):
        audit(tmp_path)
