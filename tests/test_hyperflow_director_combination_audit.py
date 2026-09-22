"""Fail-closed Director HyperFlow mode matrix; this does not claim GPU quality."""

from __future__ import annotations

import pytest

from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_hyperflow import apply_hyperflow_graph
from h3_audio_t8_pkg.director_project import ProjectStore, new_project


VARIANTS = ("single8", "continuous4plus4", "upscale8plus4", "upscale4plus4")
TASKS = ("t2va", "i2va", "l2va", "fl2va", "ref2va", "hybrid")
AUDIO_MODES = ("native", "lock_source")
UNQUALIFIED = tuple(
    (task, audio_mode)
    for task in TASKS
    for audio_mode in AUDIO_MODES
    if (task, audio_mode) != ("t2va", "native")
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("task,audio_mode", UNQUALIFIED)
def test_unqualified_task_audio_pairs_reject_before_weight_or_graph_use(
    variant, task, audio_mode
):
    with pytest.raises(ValueError, match="仅验证 T2VA 原生声音"):
        apply_hyperflow_graph(
            {}, {"task_type": task, "audio_mode": audio_mode},
            {"variant": variant}, None, 0, {},
        )


@pytest.mark.parametrize("feature", ("prompt_relay", "fast_h3_v2"))
def test_unqualified_d3_model_or_condition_route_rejects_before_model_load(
    tmp_path, monkeypatch, feature
):
    from h3_audio_t8_pkg import director_generation

    monkeypatch.setattr(director_generation, "resolve_ffmpeg", lambda: "ffmpeg")
    project = new_project()
    shot = project["doc"]["shots"][0]
    shot["simplePrompt"] = "A single steady shot in a bright room."
    project["doc"]["sampling"] = {
        "mode": "hyperflow", "variant": "single8",
        "hyperflow_file": "hyperflow/placeholder.safetensors",
    }
    project["doc"]["d3"][feature] = {"enabled": True}
    store = ProjectStore(tmp_path / "user", tmp_path / "input")
    with pytest.raises(ValueError, match="尚未实现 D3 Relay/FastH3"):
        build_director_generation_prompt(project, shot["id"], store)
