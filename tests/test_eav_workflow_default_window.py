from tools import build_eav_extended_workflows as extended
from tools import build_eav_long_video_workflow as long_video


EAV_PREFIX = "MiniMaxH3EnhanceAVideo"


def _assert_moderate_window(workflow):
    composers = [
        node
        for node in workflow["nodes"]
        if node["type"].startswith(EAV_PREFIX) and "Audit" not in node["type"]
    ]
    assert len(composers) == 1
    values = composers[0]["widgets_values"]
    offset = 1 if composers[0]["type"].endswith("SageComposerT8Advanced") else 0
    assert values[offset : offset + 4] == ["apply_exp", 4.0, 0.15, 0.90]


def test_extended_eav_builders_use_moderate_window_from_any_checkout_depth():
    workflows = [
        extended.build("T2VA", "stock20")[1],
        extended.build("I2VA", "stock20")[1],
        extended.build("T2VA", "turbo8_alpha8")[1],
        extended.build_strict_sage_workflow()[1],
        extended.build_prompt_relay_workflow()[1],
        extended.build_block_cache_workflow()[1],
        extended.build_stg_workflow()[1],
    ]
    for workflow in workflows:
        _assert_moderate_window(workflow)


def test_long_video_eav_builder_uses_moderate_window():
    _assert_moderate_window(long_video.build())
