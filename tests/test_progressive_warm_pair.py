from copy import deepcopy
import json
import subprocess
import sys

import pytest

from tools.run_progressive_warm_pair import RESEARCH, sequence, summarize


def rows():
    return [{**item, "status": "media_validated", "server_pid": 123, "cached": False,
             "conditioning_sha256": "fixture-not-real-GPU", "seconds": seconds,
             "allocator_peak_bytes": 100} for item, seconds in zip(sequence("T2VA"), (999, 888, 100, 60, 80, 120))]


def test_sequence_has_two_unscored_warmups_and_fixed_ABBA_measurements():
    for task in ("T2VA", "I2VA"):
        plan = sequence(task)
        assert [p["measured"] for p in plan] == [False, False, True, True, True, True]
        assert [p["case"].split("_", 1)[1] for p in plan] == [
            "native8", "progressive6plus2", "native8", "progressive6plus2", "progressive6plus2", "native8"]
    with pytest.raises(ValueError):
        sequence("VDN")


def test_summary_excludes_warmups_and_does_not_claim_statistical_significance():
    summary = summarize(rows(), "T2VA")
    assert summary["routes"]["native8"]["median_seconds"] == 110
    assert summary["routes"]["progressive6plus2"]["median_seconds"] == 70
    assert summary["saved_fraction_from_medians"] == pytest.approx(1-70/110)
    assert "human_review_pending" in summary["status"]
    assert "no statistical significance" in summary["scope"]


@pytest.mark.parametrize("fault", ["partial", "order", "cached", "pid", "condition", "failed", "nan", "negative", "cpu"])
def test_bad_warm_evidence_cannot_pass(fault):
    data = deepcopy(rows())
    if fault == "partial":
        data.pop()
    elif fault == "order":
        data[2], data[3] = data[3], data[2]
    elif fault == "cached":
        data[3]["cached"] = True
    elif fault == "pid":
        data[3]["server_pid"] = 456
    elif fault == "condition":
        data[3]["conditioning_sha256"] = "changed"
    elif fault == "failed":
        data[3]["status"] = "failed"
    elif fault == "nan":
        data[3]["seconds"] = float("nan")
    elif fault == "negative":
        data[3]["seconds"] = -1
    else:
        data[3]["status"] = "cpu_transport_only_no_GPU"
    with pytest.raises(ValueError):
        summarize(data, "T2VA")


def test_default_plan_creates_no_files_or_gpu_jobs():
    target = RESEARCH / "warm-plan-must-not-create"
    assert not target.exists()
    run = subprocess.run([sys.executable, "-X", "utf8", "tools/run_progressive_warm_pair.py",
        "--run-root", str(target)], cwd=RESEARCH.parents[1], text=True, capture_output=True, timeout=15)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["mode"] == "plan_only_no_GPU_or_files"
    assert not target.exists()
