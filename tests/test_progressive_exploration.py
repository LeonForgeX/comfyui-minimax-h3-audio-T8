from copy import deepcopy
import json

import pytest

from tools.run_progressive_pilot import EXPLORATION_CASES, RESEARCH, GAME_PROMPT, exploration_recipe
from tools.run_progressive_exploration import frozen_plan, validate_job
from tools.progressive_pilot_analysis import common_graph


def test_plan_contains_exact_predeclared_ten_jobs_without_repeating_accepted_seed():
    plan = frozen_plan()
    assert len(plan) == 10
    assert len({item["folder"] for item in plan}) == 10
    assert {item["exploration_case"] for item in plan} == set(EXPLORATION_CASES)
    assert "portrait_2609032101" not in EXPLORATION_CASES
    for a, b in zip(plan[::2], plan[1::2]):
        assert a["case"] == "T2VA_native8" and b["case"] == "T2VA_progressive6plus2"
        assert a["exploration_case"] == b["exploration_case"]
        assert common_graph(a["graph"]) == common_graph(b["graph"])
        assert a["graph"]["10"]["inputs"]["seed"] == b["graph"]["10"]["inputs"]["seed"]


@pytest.mark.parametrize("item", frozen_plan(), ids=lambda x: x["folder"])
def test_only_predeclared_prompt_seed_prefix_change_and_original_immutable(item):
    case, exploration = item["case"], item["exploration_case"]
    recipe = json.loads((RESEARCH / "pilot-api-drafts" / (case + ".prompt.json")).read_text(encoding="utf-8"))
    original = deepcopy(recipe)
    result = exploration_recipe(recipe, case, exploration)
    assert recipe == original
    seed_node, seed_key = ("8", "noise_seed") if case.endswith("native8") else ("10", "seed")
    assert result[seed_node]["inputs"][seed_key] == int(exploration.split("_")[1])
    if exploration.startswith("game_"):
        assert result["90"]["inputs"]["value"] == GAME_PROMPT
    else:
        assert result["90"] == original["90"]
    for node in (seed_node, "90", "18"):
        result[node] = original[node]
    assert result == original


@pytest.mark.parametrize("case,exploration", [
    ("I2VA_native8", "game_2609032101"), ("T2VA_native8", "game_1"),
    ("T2VA_progressive6plus2", "portrait_2609032101"), ("T2VA_other", "game_2609032102")])
def test_unplanned_or_reused_cases_cannot_generate(case, exploration):
    with pytest.raises(ValueError, match="predeclared"):
        exploration_recipe({}, case, exploration)


@pytest.mark.parametrize("tamper", ("graph", "exploration_case", "case", "sources", "core"))
def test_postflight_rejects_changed_manifest_or_runtime(tamper):
    item = frozen_plan()[0]
    expected = {"sources": {"a": "b"}, "core": {"commit": "c"}}
    run = {"terminal": {"exploration_case": item["exploration_case"], "case": item["case"]},
        "graph": deepcopy(item["graph"]), "environment": deepcopy(expected)}
    validate_job(run, item, expected)
    if tamper == "graph":
        run["graph"]["10"]["inputs"]["seed"] += 1
    elif tamper in ("sources", "core"):
        run["environment"][tamper] = {}
    else:
        run["terminal"][tamper] = "wrong"
    with pytest.raises(ValueError, match="frozen"):
        validate_job(run, item, expected)


def test_game_prompt_matches_predeclared_document():
    assert "> " + GAME_PROMPT in (RESEARCH / "EXPLORATION_PLAN.md").read_text(encoding="utf-8")
