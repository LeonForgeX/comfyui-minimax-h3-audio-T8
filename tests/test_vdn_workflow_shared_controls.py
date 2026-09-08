"""Shared workflow controls preserve the previously validated fixed API recipe."""

import copy
import json
from pathlib import Path

import pytest

from tools import export_vdn_two_pass_workflows as exporter
from h3_audio_t8_pkg.timing import make_timing_plan


ROUTES = tuple(f"H3_OpenVDN_{task}_{backend}_TwoPass_EXP"
               for task in ("T2VA", "I2VA") for backend in ("vdn", "native_h3"))


def source(graph, kind):
    matches = [(key, node) for key, node in graph.items() if node["class_type"] == kind]
    assert len(matches) == 1, f"Expected exactly one shared {kind} control"
    return matches[0]


def controls(graph):
    text_id, text = source(graph, "PrimitiveStringMultiline")
    timing_id, timing = source(graph, "MiniMaxH3DurationPlannerT8")
    for node_id in ("6", "61"):
        assert graph[node_id]["inputs"]["prompt"] == [text_id, 0]
        assert graph[node_id]["inputs"]["length"] == [timing_id, 0]
    for node_id in ("12", "71"):
        assert graph[node_id]["inputs"]["duration_seconds"] == [timing_id, 1]
    assert timing["inputs"]["scene_start_seconds"] == 0
    assert timing["inputs"]["warmup_seconds"] == timing["inputs"]["cooldown_seconds"] == 0
    assert timing["inputs"]["ensure_minimum_context"] is False
    return text_id, text, timing_id, timing


def resolve_shared(graph):
    graph = copy.deepcopy(graph)
    text_id, text, timing_id, timing = controls(graph)
    plan = make_timing_plan(**timing["inputs"])
    values = {(text_id, 0): text["inputs"]["value"],
              (timing_id, 0): plan.frame_count, (timing_id, 1): plan.render_duration_seconds}
    graph.pop(text_id)
    graph.pop(timing_id)
    for node in graph.values():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and tuple(value) in values:
                node["inputs"][key] = values[tuple(value)]
    return graph


@pytest.mark.parametrize("route", ROUTES)
def test_shared_control_defaults_preserve_the_entire_previously_roundtripped_api(route):
    graph, _backend = exporter.candidate_graphs()[route]
    # The original files remain historical evidence; never rewrite them to pass this test.
    root = Path(exporter.__file__).resolve().parents[1]
    original = json.loads((root / "tests/fixtures/vdn_two_pass_controls_baseline"
                           / f"{route}.prompt.json").read_text(encoding="utf-8"))
    assert resolve_shared(graph) == original


@pytest.mark.parametrize("route", ROUTES)
def test_one_prompt_and_duration_edit_reaches_both_passes_and_both_trim_nodes(route):
    graph, _backend = exporter.candidate_graphs()[route]
    _text_id, text, _timing_id, timing = controls(graph)
    text["inputs"]["value"] = "A different prompt, changed exactly once."
    timing["inputs"]["scene_duration_seconds"] = 5.0
    resolved = resolve_shared(graph)
    for node_id in ("6", "61"):
        assert resolved[node_id]["inputs"]["prompt"] == "A different prompt, changed exactly once."
        assert resolved[node_id]["inputs"]["length"] == 124
    for node_id in ("12", "71"):
        assert resolved[node_id]["inputs"]["duration_seconds"] == 124 / 24
    assert graph["18"]["class_type"] == graph["73"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
    assert graph["62"]["inputs"]["second_pass_audio_strength"] == 0
