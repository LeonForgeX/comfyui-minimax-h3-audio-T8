import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from h3_audio_t8_pkg import video_outpaint_delivery as delivery


def _inputs(tmp_path):
    private, target = tmp_path / "private.mp4", tmp_path / "result.mp4"
    private.write_bytes(b"fake media bytes; publication-layer fixture only")
    report = {"schema": "t8.h3.video_outpaint.final_file/v1", "path": str(target),
              "sha256": hashlib.sha256(private.read_bytes()).hexdigest(), "perceptual_acceptance": False}
    return private, target, report


def test_persisted_report_requires_actual_matching_video(tmp_path):
    private, target, report = _inputs(tmp_path)
    returned = delivery.publish_with_delivery_report(private, target, report)
    assert delivery.read_delivery_report(target) == returned
    assert not list(tmp_path.glob(".*.receipt-*.json"))
    target.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="differs"):
        delivery.read_delivery_report(target)


def test_modified_report_and_missing_video_never_count_as_delivery(tmp_path):
    private, target, report = _inputs(tmp_path)
    delivery.publish_with_delivery_report(private, target, report)
    path = delivery.delivery_report_path(target)
    data = json.loads(path.read_text())
    data["perceptual_acceptance"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity"):
        delivery.read_delivery_report(target)
    target.unlink()
    with pytest.raises(FileNotFoundError):
        delivery.read_delivery_report(target)


def test_report_write_failure_publishes_no_video(tmp_path, monkeypatch):
    private, target, report = _inputs(tmp_path)
    def fail(*args):
        raise OSError("disk fsync failed")
    monkeypatch.setattr(delivery.os, "fsync", fail)
    with pytest.raises(OSError, match="fsync"):
        delivery.publish_with_delivery_report(private, target, report)
    assert not target.exists() and not delivery.delivery_report_path(target).exists()
    assert not list(tmp_path.glob(".*.receipt-*.json"))


def test_cancel_between_report_and_video_removes_only_owned_report(tmp_path):
    private, target, report = _inputs(tmp_path)
    count = 0
    def cancel():
        nonlocal count
        count += 1
        if count == 2:
            assert delivery.delivery_report_path(target).exists() and not target.exists()
            raise RuntimeError("cancel between links")
    with pytest.raises(RuntimeError, match="cancel"):
        delivery.publish_with_delivery_report(private, target, report, interrupt_check=cancel)
    assert not target.exists() and not delivery.delivery_report_path(target).exists()
    assert delivery.publish_with_delivery_report(private, target, report)


def test_racing_user_target_survives_failed_no_replace_link(tmp_path, monkeypatch):
    private, target, report = _inputs(tmp_path)
    link = os.link
    def race(source, destination):
        if Path(destination) == target:
            target.write_bytes(b"user result wins")
        return link(source, destination)
    monkeypatch.setattr(delivery.os, "link", race)
    with pytest.raises(FileExistsError):
        delivery.publish_with_delivery_report(private, target, report)
    assert target.read_bytes() == b"user result wins"
    assert not delivery.delivery_report_path(target).exists()


def test_existing_report_not_overwritten_even_without_video(tmp_path):
    private, target, report = _inputs(tmp_path)
    sidecar = delivery.delivery_report_path(target)
    sidecar.write_bytes(b"existing record")
    with pytest.raises(FileExistsError):
        delivery.publish_with_delivery_report(private, target, report)
    assert sidecar.read_bytes() == b"existing record" and not target.exists()


def test_actual_process_exit_leaves_no_false_success(tmp_path):
    worker = Path(__file__).with_name("outpaint_delivery_exit_worker.py")
    result = subprocess.run([sys.executable, str(worker), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 47, result.stderr
    target = tmp_path / "result.mp4"
    assert delivery.delivery_report_path(target).exists() and not target.exists()
    with pytest.raises(FileNotFoundError):
        delivery.read_delivery_report(target)


def test_metadata_size_limit_prevents_partial_publication(tmp_path, monkeypatch):
    private, target, report = _inputs(tmp_path)
    monkeypatch.setattr(delivery, "MAX_REPORT_BYTES", 1)
    with pytest.raises(ValueError, match="metadata limit"):
        delivery.publish_with_delivery_report(private, target, report)
    assert not target.exists() and not delivery.delivery_report_path(target).exists()
