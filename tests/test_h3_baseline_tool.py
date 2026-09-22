"""Controller safety/identity tests do not import torch or initialize CUDA."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools/h3_cold_warm_baseline.py"
spec = importlib.util.spec_from_file_location("h3_baseline_tool_under_test", PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_default_command_only_describes_fixed_recipe():
    result = subprocess.run([sys.executable, str(PATH)], check=True, capture_output=True, text=True)
    config = json.loads(result.stdout)
    assert config == tool.recipe()
    assert (config["length"] - 5) % 17 == 0
    assert config["length"] >= 24 * config["seconds"]
    assert config["width"] * config["height"] == 399360
    assert config["disable_pinned_memory"] is True


@pytest.mark.parametrize("payload", [{}, {"schema": "t8.h3.owned_cold_warm.v1"}])
def test_worker_refuses_missing_explicit_controller_authorization(tmp_path, payload):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit --run"):
        tool.worker(path)


def test_new_evidence_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "report.json"
    tool.write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        tool.write_new(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}


def test_identity_binds_recipe_change_and_ignores_dict_key_order():
    assert tool.json_sha({"a": 1, "b": 2}) == tool.json_sha({"b": 2, "a": 1})
    config = tool.recipe()
    assert tool.json_sha(config) != tool.json_sha({**config, "seed": config["seed"] + 1})


def test_source_identity_does_not_change_when_evidence_directory_moves(tmp_path):
    identities = []
    for folder in ("first-evidence", "second-evidence"):
        root = tmp_path / folder
        source, fallback, core, controller = root / "source", root / "fallback", root / "core", root / "tool.py"
        sources = {str(source / "nodes.py"): "a" * 64, str(fallback / "sol.py"): "b" * 64,
                   str(core / "nodes.py"): "c" * 64, str(controller): "d" * 64}
        identities.append(tool.source_content_identity(sources, source, fallback, core, controller))
    assert identities[0] == identities[1]
    sources[str(source / "nodes.py")] = "e" * 64
    assert tool.source_content_identity(sources, source, fallback, core, controller) != identities[0]
