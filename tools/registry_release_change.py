"""A documentation/packaging-only push must not republish an unchanged Registry version."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def version_at(root: Path, revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("Expected a Git event commit SHA")
    content = subprocess.check_output(
        ["git", "show", revision + ":pyproject.toml"], cwd=root
    ).decode("utf-8")
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", content)
    if project is None:
        raise ValueError("Missing [project] release metadata")
    found = re.findall(r'^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$', project[1], re.M)
    if len(found) != 1:
        raise ValueError("Expected one project.version")
    return found[0]


def should_publish(root: Path, event: str, before: str, after: str) -> dict:
    if event == "workflow_dispatch":
        return {"publish": True, "reason": "explicit_workflow_dispatch"}
    if event != "push":
        raise ValueError("Unsupported publication event")
    current = version_at(root, after)
    if before and set(before) == {"0"} and len(before) in (40, 64):
        return {"publish": True, "reason": "first_push", "after_version": current}
    previous = version_at(root, before)
    return {
        "publish": current != previous,
        "reason": "version_changed" if current != previous else "same_version_source_only",
        "before_version": previous,
        "after_version": current,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--before", default="")
    parser.add_argument("--after", required=True)
    args = parser.parse_args()
    result = should_publish(ROOT, args.event, args.before, args.after)
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write("publish=" + str(result["publish"]).lower() + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
