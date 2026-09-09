from __future__ import annotations

import os
import json
import copy
import math
from pathlib import Path
import subprocess
import sys

import torch
import pytest

from h3_audio_t8_pkg.speed_advanced import accumulate_spectrum_dataset
from h3_audio_t8_pkg.speed_spectrum_storage import save_spectrum_dataset_file
from h3_audio_t8_pkg.tools.finalize_h3_speed_spectrum_dataset import finalize_file


def _assert_same_fitted_report(actual, expected):
    actual = copy.deepcopy(actual)
    expected = copy.deepcopy(expected)
    # CPU float64 least-squares repeated on identical stored data differed by
    # at most12 ULP locally. Compare only the3 fitted scalars within32 ULP;
    # every provenance hash, policy, count and other value remains exact.
    for key in ("amplitude", "beta", "r_squared"):
        left = actual["profile"]["fit"].pop(key)
        right = expected["profile"]["fit"].pop(key)
        assert type(left) is float and type(right) is float
        assert math.isfinite(left) and math.isfinite(right)
        assert math.isclose(left, right, rel_tol=0.0,
                            abs_tol=32 * max(math.ulp(left), math.ulp(right))), (key, left, right)
    assert actual == expected


@pytest.mark.parametrize("corruption", ["amplitude", "hash", "status", "type", "nonfinite"])
def test_fitted_report_comparison_rejects_non_roundoff_changes(corruption):
    expected = {"profile": {"fit": {"amplitude": 0.75, "beta": 1.9, "r_squared": 0.88},
                            "status": "research_probe_only", "sha256": "a" * 64}}
    actual = copy.deepcopy(expected)
    if corruption == "hash":
        actual["profile"]["sha256"] = "b" * 64
    elif corruption == "status":
        actual["profile"]["status"] = "dataset_profile"
    else:
        actual["profile"]["fit"]["amplitude"] = {
            "amplitude": 0.75 + 1e-10, "type": True, "nonfinite": float("nan")}[corruption]
    with pytest.raises(AssertionError):
        _assert_same_fitted_report(actual, expected)


def test_fitted_report_comparison_accepts_roundoff_without_mutating_reports():
    expected = {"profile": {"fit": {"amplitude": 0.7436082615051828,
                                     "beta": 1.9352623315279447, "r_squared": 0.88540114068362}}}
    actual = copy.deepcopy(expected)
    actual["profile"]["fit"].update(amplitude=0.7436082615051841, beta=1.9352623315279456)
    saved = copy.deepcopy((actual, expected))
    _assert_same_fitted_report(actual, expected)
    assert (actual, expected) == saved


def test_finalize_tool_cli_bootstraps_project_package_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "tools" / "finalize_h3_speed_spectrum_dataset.py"),
            "--help",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Finalize one persisted H3 SPEED spectrum dataset" in completed.stdout


def test_finalize_tool_loads_persisted_dataset_and_keeps_small_probe_research(tmp_path):
    generator = torch.Generator().manual_seed(71)
    latent = torch.randn(2, 24, 4, 16, 24, generator=generator)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(2, 24, 4, 16, 24)
    dataset, _ = accumulate_spectrum_dataset(
        latent,
        batch_id="two-clips",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        max_temporal_samples=4,
    )
    save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="probe",
        overwrite=False,
    )
    result = finalize_file(
        storage_root=tmp_path,
        dataset_name="probe",
        profile_name="probe-profile",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    assert result["storage"]["action"] == "load"
    assert result["profile"]["independent_clip_count"] == 2
    assert result["profile"]["status"] == "research_probe_only"
    assert result["profile"]["validated_for_delta_optimal"] is False

    # Exercise actual standalone file fitting, not only argparse --help.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    environment.pop("PYTHONPATH", None)
    report = tmp_path / "cli-report.json"
    completed = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "tools/finalize_h3_speed_spectrum_dataset.py"),
        "--storage-root", str(tmp_path), "--dataset-name", "probe", "--profile-name", "probe-profile",
        "--minimum-r-squared", "0.0", "--minimum-independent-clips", "100", "--output", str(report),
    ], env=environment, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    _assert_same_fitted_report(json.loads(report.read_text(encoding="utf-8")), result)
