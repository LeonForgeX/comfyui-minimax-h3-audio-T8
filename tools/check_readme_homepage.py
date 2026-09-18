"""Keep bilingual homepages short, navigable and free of release-log sprawl."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LIMIT_BYTES = 10000
LIMIT_LINES = 140
LIMIT_HEADINGS = 8
MARKERS = {
    "README.md": re.compile(r"当前版本：\*\*(\d+\.\d+\.\d+)\*\*"),
    "README_EN.md": re.compile(r"Current version: \*\*(\d+\.\d+\.\d+)\*\*"),
}
MODEL_URLS = (
    "https://huggingface.co/t8star/Taeh3-Comfy",
    "https://huggingface.co/t8star/Meridian-Comfy",
)
LINK_RE = re.compile(r"\]\(([^)]+)\)")


def check_text(root: Path, filename: str, text: str) -> list[str]:
    errors = []
    if len(text.encode("utf-8")) > LIMIT_BYTES:
        errors.append(f"{filename}: homepage exceeds {LIMIT_BYTES} bytes; move details to docs")
    if len(text.splitlines()) > LIMIT_LINES:
        errors.append(f"{filename}: homepage exceeds {LIMIT_LINES} lines; use CHANGELOG.md")
    if len(re.findall(r"(?m)^#{1,6}\s", text)) > LIMIT_HEADINGS:
        errors.append(f"{filename}: too many headings; keep the homepage navigational")
    if len(MARKERS[filename].findall(text)) != 1:
        errors.append(f"{filename}: expected one current-version marker")
    targets = LINK_RE.findall(text)
    if "CHANGELOG.md" not in targets:
        errors.append(f"{filename}: missing clickable CHANGELOG.md link")
    for required in MODEL_URLS:
        if required not in targets:
            errors.append(f"{filename}: missing clickable model link {required}")
    if re.search(r"(?m)^\s*(?:\*\*)?20\d\d[-/]\d\d[-/]\d\d\b", text):
        errors.append(f"{filename}: dated release entries belong in CHANGELOG.md")
    if re.search(r"(?is)<details\b.*?(?:release|changelog|更新日志|发布记录)", text):
        errors.append(f"{filename}: do not hide release-history sprawl in details")
    for target in targets:
        if re.match(r"^(?:https?://|mailto:|#|codex:)", target):
            continue
        relative = target.split("#", 1)[0].split("?", 1)[0]
        if not relative:
            continue
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f"{filename}: invalid local link {target}")
    return errors


def check(root: Path = ROOT) -> list[str]:
    errors, versions = [], []
    for filename, marker in MARKERS.items():
        try:
            text = (root / filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{filename}: cannot read homepage ({type(exc).__name__})")
            continue
        errors.extend(check_text(root, filename, text))
        versions.extend(marker.findall(text))
    if len(versions) == 2 and versions[0] != versions[1]:
        errors.append("Bilingual current versions disagree")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = check(args.root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("README homepages: length, structure, model links and local navigation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
