from __future__ import annotations

from h3_audio_t8_pkg.director_capabilities import inspect_director_capabilities


def test_d3_capability_inventory_is_explicit_and_truthful():
    report = inspect_director_capabilities(
        {
            "MiniMaxH3SemanticBridgeConfigT8",
            "MiniMaxH3SemanticBridgeApplyT8",
            "MiniMaxH3TopazEnvironmentEXPT8",
        },
        model_names=["minimax_h3_ref2va_int8_convrot.safetensors"],
    )
    rows = {row["id"]: row for row in report["capabilities"]}
    assert rows["semantic_bridge"]["state"] == "ready"
    assert rows["topaz"]["state"] == "missing_entry_nodes"
    assert report["warning"].startswith("ready means registered")
    assert report["model_inventory_context"] == ["minimax_h3_ref2va_int8_convrot.safetensors"]
