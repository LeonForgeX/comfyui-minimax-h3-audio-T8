from copy import deepcopy
import subprocess
import sys

import pytest

from tools.verify_progressive_allocator_gpu import MIB, RESEARCH, verify_payload


def report():
    return {"snapshots": [{"boundary": name, "allocated_current_bytes": used*MIB,
                            "allocated_peak_bytes": peak*MIB} for name, used, peak in (
        ("before_generation", 0, 0), ("first_live", 32, 32),
        ("both_live", 96, 96), ("after_generation_terminal", 0, 96))]}


def test_real_payload_oracle_requires_allocation_growth_and_retained_high_water():
    verify_payload(report())


@pytest.mark.parametrize("fault", ["first", "transient", "peak", "release", "boundary"])
def test_zero_or_stale_allocator_data_cannot_pass(fault):
    value = deepcopy(report())
    rows = value["snapshots"]
    if fault == "first":
        rows[1]["allocated_current_bytes"] = 0
    elif fault == "transient":
        rows[2]["allocated_current_bytes"] = 32*MIB
    elif fault == "peak":
        rows[3]["allocated_peak_bytes"] = 32*MIB
    elif fault == "release":
        rows[3]["allocated_current_bytes"] = 96*MIB
    else:
        rows.pop()
    with pytest.raises(ValueError):
        verify_payload(value)


def test_default_cli_does_not_create_a_run_or_start_gpu():
    destination = RESEARCH / "allocator-default-must-not-start"
    assert not destination.exists()
    run = subprocess.run([sys.executable, "-X", "utf8", "tools/verify_progressive_allocator_gpu.py",
        "--run-root", str(destination)], cwd=RESEARCH.parents[1], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "No GPU action" in run.stdout and not destination.exists()
