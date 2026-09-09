"""Outpaint cannot rewrite207 established workflows, including released DLSS fixes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _snapshots():
    snapshot = json.loads((ROOT / "tests/fixtures/outpaint_v174_workflow_baseline.json").read_text(encoding="utf-8"))
    updates = json.loads((ROOT / "tests/fixtures/outpaint_published_workflow_updates.json").read_text(encoding="utf-8"))
    return snapshot, updates


def _guard(read_bytes, snapshot, updates):
    paths = snapshot["workflow_paths"]
    assert len(paths) == len(set(paths)) == snapshot["workflow_count"] == 207
    assert updates["baseline_commit"] == snapshot["baseline_commit"]
    assert updates["published_commit"] == "3769d70aa70431793cfbb29cd381ac4b19eee3fa"
    approved = updates["approved_updates"]
    assert len(approved) == 4 and set(approved) <= set(paths)
    assert all(path.startswith("examples/workflows/25-dlss-nr/") for path in approved)
    digest = hashlib.sha256()
    for relative in paths:
        actual = hashlib.sha256(read_bytes(relative).replace(b"\r\n", b"\n")).digest()
        if relative in approved:
            update = approved[relative]
            assert actual.hex() == update["published_sha256"], relative
            # Reconstruct the original aggregate only after validating the exact
            # already-published replacement. No directory exclusions or rebaseline.
            actual = bytes.fromhex(update["old_sha256"])
            assert len(actual) == 32
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(actual)
    assert digest.hexdigest() == snapshot["workflow_sha256"]


def test_every_v174_workflow_retains_exact_content_except_platform_newlines():
    _guard(lambda relative: (ROOT / relative).read_bytes(), *_snapshots())


@pytest.mark.parametrize("which", ["old_workflow", "published_dlss", "approval_old_hash", "extra_approval"])
def test_unapproved_edits_are_still_rejected(which):
    snapshot, updates = _snapshots()
    first = snapshot["workflow_paths"][0]
    dlss = next(iter(updates["approved_updates"]))
    def read(relative):
        data = (ROOT / relative).read_bytes()
        return data + b" " if relative == {"old_workflow": first, "published_dlss": dlss}.get(which) else data
    if which == "approval_old_hash":
        updates["approved_updates"][dlss]["old_sha256"] = "0"*64
    elif which == "extra_approval":
        updates["approved_updates"][first] = updates["approved_updates"][dlss]
    with pytest.raises(AssertionError):
        _guard(read, snapshot, updates)
