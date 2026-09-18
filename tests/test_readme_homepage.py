from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("readme_guard", ROOT / "tools/check_readme_homepage.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


def test_formal_bilingual_homepages_are_short_and_navigable():
    assert GUARD.check(ROOT) == []


@pytest.mark.parametrize("filename", ["README.md", "README_EN.md"])
@pytest.mark.parametrize("mutation,diagnostic", [
    (lambda text: text + "x" * 10001, "bytes"),
    (lambda text: text + "\n" * 141, "lines"),
    (lambda text: text + "\n## Another section\n" * 9, "headings"),
    (lambda text: text.replace("](CHANGELOG.md)", "](README.md)"), "CHANGELOG.md"),
    (lambda text: text.replace("https://huggingface.co/t8star/Taeh3-Comfy", "https://example.com/model"), "Taeh3-Comfy"),
    (lambda text: text.replace("https://huggingface.co/t8star/Meridian-Comfy", "https://example.com/model"), "Meridian-Comfy"),
    (lambda text: text + "\n2026-09-19 **v1.86.0 new release**\n", "dated"),
    (lambda text: text + "\n<details><summary>Changelog</summary>history</details>\n", "details"),
    (lambda text: text + "\n[missing](docs/never-created.md)\n", "invalid local link"),
    (lambda text: text + "\n[outside](../README.md)\n", "invalid local link"),
])
def test_guard_rejects_sprawl_and_broken_navigation(filename, mutation, diagnostic):
    text = (ROOT / filename).read_text(encoding="utf-8")
    errors = GUARD.check_text(ROOT, filename, mutation(text))
    assert any(diagnostic in error for error in errors), errors


@pytest.mark.parametrize("filename", ["README.md", "README_EN.md"])
def test_duplicate_version_is_rejected(filename):
    text = (ROOT / filename).read_text(encoding="utf-8")
    marker = "当前版本：**1.85.0**" if filename == "README.md" else "Current version: **1.85.0**"
    assert any("one current-version" in error for error in GUARD.check_text(ROOT, filename, text + "\n" + marker))


def test_unknown_absolute_url_does_not_access_local_files():
    text = (ROOT / "README_EN.md").read_text(encoding="utf-8")
    assert GUARD.check_text(ROOT, "README_EN.md", text + "\n[site](https://example.com/a#fragment)\n") == []
