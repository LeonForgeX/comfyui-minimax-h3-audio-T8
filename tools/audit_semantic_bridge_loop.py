"""Read-only loop receipt audit: segment-local Bridge identities and exact resume."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def receipts(value, result):
    if isinstance(value, dict):
        if value.get("schema") == "t8_semantic_bridge_v1" and "receipt_sha256" in value:
            payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
            if digest(payload) != value["receipt_sha256"]:
                raise ValueError("Bridge receipt integrity mismatch")
            result[value["receipt_sha256"]] = value
        for item in value.values():
            receipts(item, result)
    elif isinstance(value, list):
        for item in value:
            receipts(item, result)
    elif isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return
        receipts(parsed, result)


def audit(root, dual=False):
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf8"))
    if terminal["status"] != "mechanical_pass_human_pending" or terminal["quality_accepted"]:
        raise ValueError("Expected completed mechanical-only loop run")
    resumed = terminal["resume"]
    if (resumed["report"].get("resume_action") != "returned_verified_existing_final"
            or not resumed.get("latent_and_media_files_unchanged")):
        raise ValueError("No verified cache-only final return")
    rows = {}
    for path in (root / "output").rglob("*.json"):
        receipts(json.loads(path.read_text(encoding="utf8")), rows)
    expected = 4 if dual else 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} distinct per-encoding Bridge receipts, found {len(rows)}")
    models = {row["sha256"] for row in rows.values()}
    if len(models) != (2 if dual else 1):
        raise ValueError("LOW/HIGH model identities did not match the independent recipe")
    for segment in range(2):
        found = [row for row in rows.values() if f":segment={segment}:" in row["encoding_source"]]
        if len(found) != (2 if dual else 1):
            raise ValueError("Bridge receipt missing for a segment or stage")
        if not all(row["input_sha256"] != row["output_sha256"] for row in found):
            raise ValueError("Active Bridge left sampled conditioning unchanged")
    if terminal["media"]["geometry"] != [896, 448, 192, "24/1"] or not terminal["media"]["strict_decode"]:
        raise ValueError("Media contract differs")
    media_path = Path(terminal["media"]["path"]).resolve(strict=True)
    if not media_path.is_relative_to((root / "output").resolve()):
        raise ValueError("Media path escaped owned output")
    if hashlib.sha256(media_path.read_bytes()).hexdigest() != terminal["media"]["sha256"]:
        raise ValueError("Completed media content changed")
    return {"status": "mechanical_receipts_pass_human_pending", "dual": dual,
        "actual_bridge_receipts": list(rows.values()), "verified_existing_final_resume": True,
        "latent_and_media_files_unchanged": True, "media": terminal["media"],
        "scope": "Content-bound encoding receipts and complete-final cache reuse; not quality or interrupted-stage recovery"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.dual)
    with args.report.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "receipts": len(report["actual_bridge_receipts"])}))


if __name__ == "__main__":
    main()
