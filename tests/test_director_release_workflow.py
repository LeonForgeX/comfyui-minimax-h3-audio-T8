import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / "examples"
    / "workflows"
    / "39-director-console"
    / "2026-09-20_H3_Obsidian_Director_Starter.json"
)


def test_director_starter_is_clean_native_entry():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    assert workflow["links"] == []
    assert len(workflow["nodes"]) == 1
    node = workflow["nodes"][0]
    assert node["type"] == "MiniMaxH3DirectorProjectT8"
    project = json.loads(node["widgets_values"][0])
    assert project["schema"] == "t8.minimax_h3.director_project"
    assert project["assets"] == []
    assert project["doc"]["sharedRefs"] == []
    assert project["doc"]["shots"][0]["tray"] == []
    assert node["widgets_values"][1] == project["current"]
    serialized = WORKFLOW.read_text(encoding="utf-8").lower()
    assert "t8_director/" not in serialized
    assert "f:\\" not in serialized and "c:\\users" not in serialized


def test_director_starter_has_user_guide():
    guide = WORKFLOW.with_name("README.md")
    text = guide.read_text(encoding="utf-8")
    assert "T8 导演台" in text
    assert "DIRECTOR_D1.md" in text
    assert "DIRECTOR_D2A.md" in text
