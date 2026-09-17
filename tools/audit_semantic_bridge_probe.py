"""Read-only complete-video audit, including recovery from controller report errors."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_semantic_bridge_probe import collect_reports, build_graph  # noqa: E402
from run_h3_memory_node_probe import audit_media  # noqa: E402
from progressive_probe_control import file_identity  # noqa: E402
import run_progressive_pilot as transport  # noqa: E402


def audit(root):
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf8"))
    history = json.loads((root / "generation/history.json").read_text(encoding="utf8"))
    if history.get("status", {}).get("status_str") != "success":
        raise RuntimeError("Comfy generation did not complete successfully")
    expected = json.loads((root / "expected.json").read_text(encoding="utf8"))
    source_now = transport.source_snapshot()
    for name, sha in source_now.items():
        if expected["sources"].get(name) != sha:
            raise RuntimeError("Runtime generation source changed: " + name)
    assets = json.loads((root / "assets.json").read_text(encoding="utf8"))
    for asset in assets:
        if file_identity(Path(asset["path"]))["sha256"] != asset["sha256"]:
            raise RuntimeError("Probe asset changed: " + asset["path"])
    _, names = build_graph(terminal["variant"], terminal["seed"], encoder=terminal["encoder"])
    if "35" not in history["outputs"]:
        names.pop("latent_capture")  # First historical baseline predates lossless capture.
    reports = collect_reports(history, names)
    active = reports["builder"]["semantic_bridge"]
    if (active is not None) != (terminal["variant"] != "native"):
        raise RuntimeError("Requested Bridge branch was not actually used")
    if active is not None and (not active["applied"] or len(active["items"]) != 1):
        raise RuntimeError("Unexpected actual Bridge application count")
    media = audit_media(root)
    if (media["width"], media["height"], media["frames"], media["fps"]) != (832, 480, 73, "24/1"):
        raise RuntimeError("Wrong video dimensions/length/fps")
    if abs(float(media["audio_seconds"]) - 73/24) > .1:
        raise RuntimeError("Audio duration mismatch")
    return {"status": "mechanical_pass_human_pending", "original_controller_status": terminal["status"],
            "controller_error": terminal.get("error"), "actual_generation_success": True,
            "source_and_assets_unchanged": True, "reports": reports, "media": media,
            "quality_accepted": False, "no_regeneration": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    transport.write_json(args.root / "independent-audit.json", result)
    print(json.dumps({"status": result["status"], "media": result["media"]}), flush=True)
