from copy import deepcopy

import pytest

from tools.audit_trt_vae_benchmark import PHASES, summarize_rounds, validate_round


def fixture_round():
    request = {"warmup_complete_decodes": 1}
    result = {
        "warmup_complete_decodes_per_route": 1,
        "events": [
            {"phase": name, "monotonic": index + 1} for index, name in enumerate(PHASES)
        ],
    }
    for kind, elapsed in (("native", 2.0), ("trt", 1.0)):
        result.update(
            {
                kind + "_calls": 60,
                kind + "_warmup_calls": 60,
                kind + "_warmup_seconds": [elapsed + 0.1],
                kind + "_measured_complete_decode_seconds": elapsed,
                kind + "_first_complete_decode_seconds": None,
            }
        )
    audit = {
        "status": "complete_rgb_evidence_verified_not_human_or_speed_qualification",
        "cuda_initialized": False,
        "finite_normalized_all_frames": True,
        "actual_calls_per_route": 60,
        "rgb_psnr_db": 60.0,
    }
    terminal = {
        "status": "worker_complete_result_requires_audit",
        "isolated": {"status": "complete", "active_after_cleanup": 0},
    }
    return request, result, audit, terminal


def test_valid_round_and_three_pair_medians():
    row = validate_round(*fixture_round())
    rows = [dict(row, start=10 * i + 1, end=10 * i + 6) for i in range(3)]
    report = summarize_rounds(rows)
    assert report["median_decode_speed_ratio"] == 2
    assert report["median_decode_time_reduction_percent"] == 50


@pytest.mark.parametrize(
    "failure",
    [
        "no_warmup",
        "tile_missing",
        "phase_missing",
        "bad_clock",
        "bad_time",
        "cleanup",
        "not_audited",
        "mislabelled",
    ],
)
def test_cannot_turn_partial_or_cold_runs_into_warm_benchmark(failure):
    request, result, audit, terminal = deepcopy(fixture_round())
    if failure == "no_warmup":
        request["warmup_complete_decodes"] = 0
    if failure == "tile_missing":
        result["trt_warmup_calls"] = 59
    if failure == "phase_missing":
        result["events"].pop(0)
    if failure == "bad_clock":
        result["events"][1]["monotonic"] = 1
    if failure == "bad_time":
        result["native_measured_complete_decode_seconds"] = float("nan")
    if failure == "cleanup":
        terminal["isolated"]["active_after_cleanup"] = 1
    if failure == "not_audited":
        audit["status"] = "pending"
    if failure == "mislabelled":
        result["native_first_complete_decode_seconds"] = 2.0
    with pytest.raises(ValueError):
        validate_round(request, result, audit, terminal)


def test_rejects_two_rounds_or_overlapping_rounds():
    row = validate_round(*fixture_round())
    with pytest.raises(ValueError):
        summarize_rounds([row, row])
    with pytest.raises(ValueError):
        summarize_rounds([row, row, row])
