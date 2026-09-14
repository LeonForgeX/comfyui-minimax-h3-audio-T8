from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "selflift-controlled-20260914"
    / "prepare_calibrated_eav.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_calibrated_eav_test", MODULE_PATH)
calibrated = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrated)

ACCEPTED_MODULE_PATH = MODULE_PATH.with_name("prepare_accepted_eav_workflows.py")
ACCEPTED_SPEC = importlib.util.spec_from_file_location(
    "prepare_accepted_eav_workflows_test", ACCEPTED_MODULE_PATH
)
accepted = importlib.util.module_from_spec(ACCEPTED_SPEC)
assert ACCEPTED_SPEC.loader is not None
ACCEPTED_SPEC.loader.exec_module(accepted)


def _graph():
    return {
        "8": {
            "inputs": {
                "eav_mode": "apply_exp",
                "eav_tau": 0.2,
                "eav_start_video_progress": 0.0,
                "eav_end_video_progress": 1.0,
                "eav_g_hard_limit": 1.5,
                "chain_id": "calibration",
                "filename_prefix": "calibration",
            }
        }
    }


def _phase(progress, active, *, gains):
    return {
        "aborted": False,
        "model_forward_count": 4,
        "config": {
            "mode": "apply_exp",
            "tau": calibrated.TAU,
            "start_video_progress": 0.15,
            "end_video_progress": 0.90,
            "g_hard_limit": 1.5,
        },
        "forwards": [
            {"progress_video": value, "active": enabled}
            for value, enabled in zip(progress, active)
        ],
        "g_min": gains[0],
        "g_max": gains[1],
        "g_mean": gains[2],
    }


def _sampling():
    return {
        "eav": {
            "low": _phase(
                [0.0, 0.01176, 0.02703, 0.04762],
                [False, False, False, False],
                gains=(None, None, None),
            ),
            "high": _phase(
                [0.07692, 0.12195, 0.20, 0.36842],
                [False, False, True, True],
                gains=(1.01, 1.10, 1.05),
            ),
            "summary": {
                "mode": "apply_exp",
                "full_schedule_nfe": 8,
                "active_forwards": 2,
                "output_gain_above_one_applied": True,
            },
        }
    }


def test_calibrated_variant_changes_only_declared_controls_and_identity():
    source = _graph()
    result = calibrated.variant(source)
    assert source == _graph()
    changed = result["8"]["inputs"]
    assert changed["eav_tau"] == 8.0
    assert changed["eav_start_video_progress"] == 0.15
    assert changed["eav_end_video_progress"] == 0.90
    assert changed["eav_g_hard_limit"] == 1.5
    assert changed["chain_id"].endswith(calibrated.SUFFIX)
    assert changed["filename_prefix"].endswith(calibrated.SUFFIX)
    assert len(changed["chain_id"] + "_real_person_followup") <= 64


def test_calibrated_receipt_requires_zero_low_and_two_high_active_forwards():
    result = calibrated.require_calibrated_eav(_sampling())
    assert result["status"] == "high_tail_nonidentity_gain_executed_quality_unverified"
    assert result["phases"]["low"]["active_forwards"] == 0
    assert result["phases"]["high"]["active_forwards"] == 2

    damaged = deepcopy(_sampling())
    damaged["eav"]["low"]["forwards"][0]["active"] = True
    with pytest.raises(ValueError, match="phase/window"):
        calibrated.require_calibrated_eav(damaged)

    damaged = deepcopy(_sampling())
    damaged["eav"]["high"]["g_max"] = 1.51
    with pytest.raises(ValueError, match="hard gain"):
        calibrated.require_calibrated_eav(damaged)

    damaged = deepcopy(_sampling())
    damaged["eav"]["summary"]["output_gain_above_one_applied"] = False
    with pytest.raises(ValueError, match="nonidentity gain"):
        calibrated.require_calibrated_eav(damaged)


@pytest.mark.parametrize("case", sorted(accepted.CASES))
def test_accepted_workflow_drafts_use_reviewed_recipe_and_explain_higher_tau(case):
    workflow, _ = accepted.convert(case)
    long_node = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3ProgressiveLongVideoEXPT8"
    )
    values = long_node["widgets_values_named"]
    assert values["eav_tau"] == 8.0
    assert values["eav_start_video_progress"] == 0.15
    assert values["eav_end_video_progress"] == 0.90
    assert values["eav_g_hard_limit"] == 1.5
    note = next(node for node in workflow["nodes"] if node["type"] == "MarkdownNote")
    text = note["widgets_values"][0]
    for phrase in (
        "如果希望更高动态",
        "逐步提高eav_tau",
        "不要误调TST自己的tau",
        "身份漂移",
        "局部形变",
        "闪烁",
        "声音变化",
    ):
        assert phrase in text
