import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from h3_audio_t8_pkg.readable_diagnostics import diagnose_node_source, explain_audio_report
from h3_audio_t8_pkg.nodes import comfy_entrypoint
from h3_audio_t8_pkg.nodes_readable_audio import MiniMaxH3NodeSourceDiagnosticT8


def test_actual_module_path_not_version_guess():
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    summary, text = diagnose_node_source("MiniMaxH3AudioConditioningT8", classes)
    report = json.loads(text)
    assert report["status"] == "loaded"
    expected_version = json.loads((Path(__file__).resolve().parents[1] / "meta.json").read_text(encoding="utf8"))["version"]
    assert report["disk_package_version"] == expected_version
    assert report["running_package_version"].startswith("unknown")
    assert report["nodes"][0]["loaded_module_path"] == str(Path(__file__).resolve().parents[1] / "h3_t8/nodes.py")
    assert report["nodes"][0]["memory_bytecode_matches_disk"] == "not_proven_by_file_hash"
    assert report["mutation"] is False and report["blocks_sampling"] is False
    assert "不等于" in summary
    assert asyncio.run(MiniMaxH3NodeSourceDiagnosticT8.execute("MiniMaxH3AudioConditioningT8")) is not None


def test_missing_node_keeps_unknown():
    _, text = diagnose_node_source("NeverInstalled", [])
    assert json.loads(text)["status"] == "not_found"


def test_native_conditioning_text_report_supported_without_prompt_parsing():
    summary, report = explain_audio_report('task=ref2va\naudio_mode=native\nsource_audio_tag=none\nframes=73 (3.042s at 24fps)')
    assert '不是直接交付音轨' in summary
    assert json.loads(report)['input_format'] == 'conditioning_text'
    summary, report = explain_audio_report('audio_mode=lock_source\nframes=73')
    assert '锁定源录音驱动画面' in summary


def test_actual_dual_policy_dict_not_confused_with_delivery():
    summary, report = explain_audio_report(json.dumps({'audio_policy': {
        'effective_source': 'legacy_policy', 'effective_strength': 0,
        'migration': 'partial_first_pass_zero_lock_to_joint_continuation',
        'first_pass_complete_trajectory': False}}))
    assert '原生联合续采' in summary and '不是最终交付音轨证明' in summary
    assert len(json.loads(report)['evidence']) == 4


def test_audio_report_discarded_refine_prediction_not_labeled_joint_delivery():
    original = {"second_pass_audio_policy": "joint_av_preserve_input",
                "final_audio_policy": "return_exact_first_pass_audio_tensor", "cache_hit": True}
    before = json.dumps(original)
    summary, text = explain_audio_report(before)
    report = json.loads(text)
    assert "不交付其音频预测" in summary
    assert "复用" in summary
    assert report["numerical_mutation"] is False and original == json.loads(before)
    assert any(e["path"] == "$.final_audio_policy" for e in report["evidence"])


def test_nested_two_models_audio_and_cache_explanation():
    value = {"audio_mode": "native", "sampling_report": {"dual_model": {
        "low_reused": True, "high_reused": False, "low_context": {"audio_source": "completed_second_pass_output"}}},
        "final_audio_policy": "publish_refined_joint_audio"}
    summary, report = explain_audio_report(json.dumps(value))
    assert "二采联合精修" in summary and "不是粗采" in summary and "不是直接交付" in summary
    assert len(json.loads(report)["evidence"]) == 5


def test_unrecognized_fields_not_invented_and_bad_json_rejected():
    for value in ("", '{"seed":3}', '{"audio_source":"unexpected_vendor"}'):
        summary, text = explain_audio_report(value)
        assert summary and json.loads(text)["status"] in {"unknown", "evidence_found"}
    with pytest.raises(ValueError, match="不是提示词"):
        explain_audio_report("singing")


def test_new_ids_append_without_changing_343_old_schemas():
    root = Path(__file__).resolve().parents[1]
    before = root / "artifacts/five-track-development-20260918/preimage/nodes.py"
    name = "h3_audio_t8_pkg._baseline343_nodes"
    spec = importlib.util.spec_from_file_location(name, before)
    old = importlib.util.module_from_spec(spec)
    sys.modules[name] = old
    try:
        spec.loader.exec_module(old)
        baseline = asyncio.run(old.comfy_entrypoint().get_node_list())
        current = asyncio.run(comfy_entrypoint().get_node_list())
        assert len(baseline) == 343 and len(current) >= 347
        assert [c.__name__ for c in current[:343]] == [c.__name__ for c in baseline]
        assert [c.INPUT_TYPES() for c in current[:343]] == [c.INPUT_TYPES() for c in baseline]
        assert all(c.define_schema() for c in current[343:347])
    finally:
        sys.modules.pop(name, None)
