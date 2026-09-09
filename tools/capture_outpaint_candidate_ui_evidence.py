"""Verify/save isolated archive UI history and displayed PNG evidence read-only.

Writes only a new report. No queueing, model calls or cache edits. PNG hashes are
computed from the files actually returned by each executed output node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from PIL import Image


ALLOWED = {"LoadVideo", "PreviewAny", "MiniMaxH3VideoOutpaintPlanT8",
    "MiniMaxH3VideoOutpaintLoadPreparedT8", "MiniMaxH3VideoOutpaintLoadCandidateT8",
    "MiniMaxH3VideoOutpaintSelectCandidateT8", "MiniMaxH3VideoOutpaintLoadSelectionT8"}


def capture_history(history, *, fixture, temp_root, copied_id):
    if len(copied_id) != 64 or any(c not in "0123456789abcdef" for c in copied_id):
        raise ValueError("requires the explicitly copied complete selection ID")
    records = {}
    for prompt_id, entry in history.items():
        graph = entry["prompt"][2]
        kinds = {n["class_type"] for n in graph.values()}
        if not kinds <= ALLOWED:
            raise ValueError("history contains non-archive/model execution nodes")
        if sum(n["class_type"] == "LoadVideo" for n in graph.values()) != 1:
            raise ValueError("unexpected source ownership")
        for node in graph.values():
            if node["class_type"] == "LoadVideo" and node["inputs"]["file"] != fixture["source_file"]:
                raise ValueError("history source differs from explicit fixture")
        status = entry["status"]
        errors = [msg[1] for msg in status["messages"] if msg[0] == "execution_error"]
        images, text = [], []
        for output in entry["outputs"].values():
            text.extend(output.get("text", []))
            for image in output.get("images", []):
                if image["type"] != "temp":
                    raise ValueError("unexpected image output type")
                path = (temp_root / image["subfolder"] / image["filename"]).resolve()
                if not path.is_relative_to(temp_root.resolve()):
                    raise ValueError("image escapes the owned temp directory")
                with Image.open(path) as picture:
                    rgb = picture.convert("RGB")
                    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                    if digest != fixture["preview_report"]["rgb8_sha256"]:
                        raise ValueError("displayed image differs from archived candidate RGB")
                    images.append({**image, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rgb8_sha256": digest, "dimensions": list(rgb.size)})
        selectors = [n["inputs"]["confirm_selection"] for n in graph.values()
                     if n["class_type"] == "MiniMaxH3VideoOutpaintSelectCandidateT8"]
        if selectors == [False]:
            if (status["status_str"] != "error" or len(errors) != 1
                    or errors[0]["node_type"] != "MiniMaxH3VideoOutpaintSelectCandidateT8"):
                raise ValueError("unconfirmed execution did not stop at Select")
        elif status["status_str"] != "success" or not status["completed"] or errors:
            raise ValueError("archive UI execution failed")
        if selectors == [True] and text != [copied_id]:
            raise ValueError("copied ID differs from actual PreviewAny output")
        if not images:
            raise ValueError("archive output did not produce an image")
        restores = [n for n in graph.values() if n["class_type"] == "MiniMaxH3VideoOutpaintLoadSelectionT8"]
        if restores and any(n["inputs"]["selection_id"] != copied_id for n in restores):
            raise ValueError("restore did not consume the copied selection ID")
        records[prompt_id] = {"status": status["status_str"], "node_types": sorted(kinds),
            "confirm_selection": selectors, "restored_selection": bool(restores),
            "images": images, "text": text, "error": errors[0]["exception_message"] if errors else None,
            "prompt": graph, "messages": [[kind, {k: v for k, v in value.items() if k in
                {"node", "prompt_id", "timestamp", "nodes"}}] for kind, value in status["messages"]]}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--copied-id", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8192")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_bytes())
    with urllib.request.urlopen(args.server.rstrip("/") + "/history", timeout=30) as response:
        history = json.load(response)
    if not history:
        raise ValueError("no executed history")
    records = capture_history(history, fixture=fixture, temp_root=args.temp_root, copied_id=args.copied_id)
    selection = Path(fixture["cache_root"]) / f"selection-{args.copied_id}.json"
    saved = json.loads(selection.read_bytes())
    if saved["candidate_archive_id"] != fixture["candidate_id"] or saved["sha256"] != args.copied_id:
        raise ValueError("confirmation archive differs from the copied ID or candidate")
    window_manifest = Path(fixture["cache_root"]) / "outpaint_windows.json"
    window_sha = hashlib.sha256(window_manifest.read_bytes()).hexdigest()
    if window_sha != fixture["preview_report"]["window_manifest_sha256"]:
        raise ValueError("archive UI changed the original sampled window manifest")
    evidence = {"scope": "CPU_archive_UI_execution_not_learned_generation_acceptance",
        "candidate_id": fixture["candidate_id"], "selection_id": args.copied_id, "jobs": records,
        "selection_file_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "window_manifest_sha256": window_sha, "sampled_prefix_unchanged": True,
        "all_displayed_rgb_equal_archived_preview": True, "model_or_sampler_nodes_in_graphs": False}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"jobs": {key: value["status"] for key, value in records.items()},
        "all_displayed_rgb_equal_archived_preview": True, "report": str(args.report)}))


if __name__ == "__main__":
    main()
