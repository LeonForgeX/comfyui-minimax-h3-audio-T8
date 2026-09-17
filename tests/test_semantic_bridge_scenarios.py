from copy import deepcopy

import pytest

from tools.run_semantic_bridge_probe import build_graph


@pytest.mark.parametrize("scenario", ["constraints", "i2va_mandarin", "ref2va_korean"])
def test_three_arms_keep_identical_generation_settings(scenario):
    native, _ = build_graph("native", 91703, scenario=scenario)
    for arm in ("original", "bunny"):
        graph, _ = build_graph(arm, 91703, scenario=scenario)
        graph.pop("33")
        graph["9"]["inputs"].pop("semantic_bridge")
        graph["16"]["inputs"]["filename_prefix"] = native["16"]["inputs"]["filename_prefix"]
        assert graph == native


@pytest.mark.parametrize("scenario", ["i2va_mandarin", "ref2va_korean"])
def test_image_scenarios_preserve_user_aspect_and_real_task_path(scenario):
    graph, _ = build_graph("original", 91703, scenario=scenario)
    values = graph["9"]["inputs"]
    assert values["width"] * 3 == values["height"] * 2
    assert values["length"] == 73 and graph["10"]["inputs"]["steps"] == 8
    assert values["task_type"] == "auto"
    if scenario == "i2va_mandarin":
        assert values["first_frame"] == ["41", 0]
        assert "ref_images.ref_image_0" not in values
        assert "你好，很高兴见到你" in values["prompt"]
        assert "fl2va" in graph["1"]["inputs"]["unet_name"]
    else:
        assert values["ref_images.ref_image_0"] == ["41", 0]
        assert "first_frame" not in values
        assert "작은 새가 노래해요" in values["prompt"]
        assert "ref2va" in graph["1"]["inputs"]["unet_name"]


def test_encoder_change_is_separate_from_bridge_algorithm():
    first, _ = build_graph("original", 91703)
    second, _ = build_graph("original", 91703, encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    second = deepcopy(second)
    second["6"] = first["6"]
    assert first == second


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="scenario"):
        build_graph("original", 0, scenario="unknown")
