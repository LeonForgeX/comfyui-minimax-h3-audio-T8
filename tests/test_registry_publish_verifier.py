import json
from pathlib import Path

import pytest

from tools.verify_registry_publish import (
    ACTIVE_STATUS,
    evaluate_registry_state,
    release_identity,
    verify_registry_publish,
)


ROOT = Path(__file__).resolve().parents[1]


def _version(version: str, status: str) -> dict[str, str]:
    return {"version": version, "status": status}


def _node(latest: str) -> dict[str, dict[str, str]]:
    return {"latest_version": {"version": latest}}


def test_release_identity_uses_the_synchronized_project_metadata():
    node_id, version = release_identity(ROOT)
    metadata = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    assert node_id == "minimax-h3-audio-t8"
    assert version == metadata["version"]


def test_release_identity_is_python_310_safe_and_scoped_to_project_table(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.decoy]\nname = "wrong"\nversion = "9.9.9"\n\n'
        '[project]\nname = "demo-node"\nversion = "2.3.4"\n',
        encoding="utf-8",
    )
    assert release_identity(tmp_path) == ("demo-node", "2.3.4")


def test_release_identity_rejects_missing_or_duplicate_project_values(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nname = "duplicate"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than once"):
        release_identity(tmp_path)


def test_active_target_must_also_be_the_public_latest_version():
    waiting = evaluate_registry_state(
        node_id="demo",
        target_version="2.0.0",
        versions=[_version("2.0.0", ACTIVE_STATUS)],
        node=_node("1.9.0"),
    )
    assert waiting["state"] == "waiting"
    assert waiting["reason"] == "public_latest_not_updated"

    active = evaluate_registry_state(
        node_id="demo",
        target_version="2.0.0",
        versions=[_version("2.0.0", ACTIVE_STATUS)],
        node=_node("2.0.0"),
    )
    assert active["state"] == "active"


def test_flagged_or_banned_upload_is_blocked_not_successful():
    for status in ("NodeVersionStatusFlagged", "NodeVersionStatusBanned"):
        result = evaluate_registry_state(
            node_id="demo",
            target_version="2.0.0",
            versions=[_version("2.0.0", status)],
            node=_node("1.9.0"),
        )
        assert result["state"] == "blocked"
        assert result["reason"] == "registry_security_review_required"


def test_missing_version_waits_and_bounded_retry_can_recover():
    responses = iter(
        [
            [],
            _node("1.9.0"),
            [_version("2.0.0", ACTIVE_STATUS)],
            _node("2.0.0"),
        ]
    )
    sleeps: list[float] = []

    exit_code, result = verify_registry_publish(
        node_id="demo",
        target_version="2.0.0",
        attempts=2,
        interval_seconds=0.25,
        fetcher=lambda _url, _timeout: next(responses),
        sleeper=sleeps.append,
    )

    assert exit_code == 0
    assert result["state"] == "active"
    assert sleeps == [0.25]


def test_flagged_version_fails_immediately_without_sleeping():
    responses = iter(
        [
            [_version("2.0.0", "NodeVersionStatusFlagged")],
            _node("1.9.0"),
        ]
    )
    sleeps: list[float] = []
    exit_code, result = verify_registry_publish(
        node_id="demo",
        target_version="2.0.0",
        attempts=20,
        fetcher=lambda _url, _timeout: next(responses),
        sleeper=sleeps.append,
    )
    assert exit_code == 2
    assert result["state"] == "blocked"
    assert sleeps == []


def test_publish_workflow_has_a_post_upload_activation_gate():
    workflow = (ROOT / ".github" / "workflows" / "publish_action.yml").read_text(
        encoding="utf-8"
    )
    publish = workflow.index("Comfy-Org/publish-node-action@main")
    activation = workflow.index("tools/verify_registry_publish.py")
    assert activation > publish
    assert "--attempts 20" in workflow


def test_registry_archive_excludes_unregistered_vretoucher_research():
    ignored = (ROOT / ".comfyignore").read_text(encoding="utf-8")
    assert "h3_t8/skin_finish_vretoucher_adapter.py" in ignored
    assert "h3_t8/skin_finish_vretoucher_runtime.py" in ignored
    assert "h3_t8/skin_finish_vretoucher_pipeline.py" in ignored
    assert "h3_t8/vendor/vretoucher_upstream/" in ignored
