#!/usr/bin/env python3
"""Fail a release job unless the uploaded Registry version is installable.

The official publish action reports that the archive upload completed.  That is
not the same as the version becoming Active: automated scanning can leave the
version Flagged/Banned while the node endpoint keeps an older Active release as
``latest_version``.  This tool checks both authoritative Registry endpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_ROOT = "https://api.comfy.org"
ACTIVE_STATUS = "NodeVersionStatusActive"
BLOCKED_STATUSES = {
    "NodeVersionStatusBanned",
    "NodeVersionStatusFlagged",
}

TABLE_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
PROJECT_STRING_RE = re.compile(
    r'^\s*(name|version)\s*=\s*"([^"\r\n]+)"\s*(?:#.*)?$'
)


def _project_identity_from_toml(text: str) -> tuple[str, str]:
    """Read the two simple strings needed by the post-publish gate.

    The Comfy publish action currently leaves Python 3.10 on PATH after it
    runs.  ``tomllib`` is Python 3.11+, and adding a runtime dependency only
    for these two scalar values would make the release gate less portable.
    This intentionally narrow reader accepts only quoted ``name`` and
    ``version`` keys inside the top-level ``[project]`` table.
    """

    in_project = False
    values: dict[str, str] = {}
    for line in str(text).splitlines():
        table = TABLE_HEADER_RE.fullmatch(line)
        if table is not None:
            in_project = table.group(1).strip() == "project"
            continue
        if not in_project:
            continue
        match = PROJECT_STRING_RE.fullmatch(line)
        if match is None:
            continue
        key, value = match.groups()
        if key in values:
            raise ValueError(f"pyproject.toml defines project.{key} more than once")
        values[key] = value.strip()
    node_id = values.get("name", "")
    version = values.get("version", "")
    if not node_id or not version:
        raise ValueError("pyproject.toml must define quoted project.name and project.version")
    return node_id, version


def release_identity(project_root: Path) -> tuple[str, str]:
    text = (Path(project_root) / "pyproject.toml").read_text(encoding="utf-8")
    return _project_identity_from_toml(text)


def fetch_json(url: str, timeout_seconds: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "t8-registry-release-gate/1"},
    )
    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
        return json.load(response)


def evaluate_registry_state(
    *,
    node_id: str,
    target_version: str,
    versions: Any,
    node: Any,
) -> dict[str, Any]:
    if not isinstance(versions, list):
        raise ValueError("Registry versions endpoint did not return a list")
    if not isinstance(node, dict):
        raise ValueError("Registry node endpoint did not return an object")

    target = next(
        (
            item
            for item in versions
            if isinstance(item, dict) and str(item.get("version", "")) == target_version
        ),
        None,
    )
    latest = node.get("latest_version")
    public_latest = (
        str(latest.get("version", "")) if isinstance(latest, dict) else ""
    )
    result: dict[str, Any] = {
        "schema": "t8.registry.publish_activation.v1",
        "node_id": node_id,
        "target_version": target_version,
        "public_latest_version": public_latest or None,
    }
    if target is None:
        return {
            **result,
            "state": "waiting",
            "reason": "target_version_not_visible",
            "target_status": None,
        }

    target_status = str(target.get("status", ""))
    result["target_status"] = target_status or None
    if target_status in BLOCKED_STATUSES:
        return {
            **result,
            "state": "blocked",
            "reason": "registry_security_review_required",
        }
    if target_status != ACTIVE_STATUS:
        return {
            **result,
            "state": "waiting",
            "reason": "target_version_not_active",
        }
    if public_latest != target_version:
        return {
            **result,
            "state": "waiting",
            "reason": "public_latest_not_updated",
        }
    return {**result, "state": "active", "reason": "published_and_installable"}


def verify_registry_publish(
    *,
    node_id: str,
    target_version: str,
    api_root: str = DEFAULT_API_ROOT,
    attempts: int = 20,
    interval_seconds: float = 15.0,
    timeout_seconds: float = 20.0,
    fetcher: Callable[[str, float], Any] = fetch_json,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    if int(attempts) < 1:
        raise ValueError("attempts must be at least 1")
    if float(interval_seconds) < 0 or float(timeout_seconds) <= 0:
        raise ValueError("interval must be non-negative and timeout must be positive")
    root = str(api_root).rstrip("/")
    versions_url = f"{root}/nodes/{node_id}/versions"
    node_url = f"{root}/nodes/{node_id}"
    last: dict[str, Any] = {
        "schema": "t8.registry.publish_activation.v1",
        "node_id": node_id,
        "target_version": target_version,
        "state": "waiting",
        "reason": "not_checked",
    }

    for attempt in range(1, int(attempts) + 1):
        try:
            versions = fetcher(versions_url, float(timeout_seconds))
            node = fetcher(node_url, float(timeout_seconds))
            last = evaluate_registry_state(
                node_id=node_id,
                target_version=target_version,
                versions=versions,
                node=node,
            )
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:
            last = {
                "schema": "t8.registry.publish_activation.v1",
                "node_id": node_id,
                "target_version": target_version,
                "state": "waiting",
                "reason": "registry_query_failed",
                "error_type": type(error).__name__,
            }
        last["attempt"] = attempt
        last["attempts"] = int(attempts)
        print(json.dumps(last, ensure_ascii=False, sort_keys=True), flush=True)

        if last["state"] == "active":
            return 0, last
        if last["state"] == "blocked":
            return 2, last
        if attempt < int(attempts):
            sleeper(float(interval_seconds))
    return 3, last


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require a published Comfy Registry version to be Active and publicly latest."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--node-id", default="")
    parser.add_argument("--version", default="")
    parser.add_argument("--api-root", default=DEFAULT_API_ROOT)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_node, project_version = release_identity(args.project_root)
    exit_code, _result = verify_registry_publish(
        node_id=str(args.node_id).strip() or project_node,
        target_version=str(args.version).strip() or project_version,
        api_root=args.api_root,
        attempts=args.attempts,
        interval_seconds=args.interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
