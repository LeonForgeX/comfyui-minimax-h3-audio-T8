"""CPU-only HR1 audit: load pure modules, never initialize the package/Core.

This is a fault-injection probe, not a GPU or visual-quality acceptance test.
Its fixtures live only in an owned temporary directory.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types


def main():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("h3_audio_t8_pkg")
    package.__path__ = [str(root / "h3_t8")]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("batch_probe_fixtures", root / "tests/test_director_batch.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    from h3_audio_t8_pkg.director_batch import batch_status
    from h3_audio_t8_pkg.director_project import atomic_json

    observations = []
    with tempfile.TemporaryDirectory(prefix="t8-hr1-contract-audit-") as temporary:
        store, batch_id, project, output = fixtures._setup(Path(temporary))
        batch = fixtures._create(store, batch_id, project, 123)
        receipt_path, _ = fixtures._receipt(store, batch, 0)
        video = output / "decoded-av.mp4"
        fixtures._write_av(video)
        done = {"state": "success", "outputs": {"save": {"videos": [
            {"type": "output", "filename": video.name, "subfolder": ""}
        ]}}}
        first = batch_status(store, batch_id, lambda _id: done, output)
        original = json.loads(receipt_path.read_text(encoding="utf-8"))
        observations.append({"case": "valid_complete_container", "state": first["items"][0]["state"]})
        # Simulate history disappearing while the persisted receipt remains.
        restored = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output, core_epoch="new-core")
        observations.append({"case": "receipt_after_history_loss", "next_index": restored["next_index"]})
        for field, invalid in [("frames", 0), ("width", -1), ("audio_stream", False), ("file", "wrong-file.mp4")]:
            corrupt = json.loads(json.dumps(original))
            corrupt["media_evidence"][0][field] = invalid
            atomic_json(receipt_path, corrupt)
            result = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output, core_epoch="new-core")
            observations.append({"case": "corrupt_evidence_" + field, "state": result["items"][0]["state"],
                                 "next_index": result["next_index"]})
        # Core queue deletion is confirmed by the existing queue-contract test
        # to leave state=unknown with no history. The cancel route currently
        # does not record that acknowledged deletion in the request receipt.
        cancelled = {key: value for key, value in original.items()
                     if key not in {"terminal", "media_evidence", "media_evidence_sha256"}}
        cancelled["core_epoch"] = "same-core"
        atomic_json(receipt_path, cancelled)
        stopped = batch_status(store, batch_id, lambda _id: {"state": "unknown"}, output, core_epoch="same-core")
        observations.append({"case": "confirmed_queue_delete_without_terminal_receipt",
                             "state": stopped["items"][0]["state"],
                             "retry_available": stopped["items"][0]["retry_available"],
                             "next_index": stopped["next_index"]})
    assert "torch" not in sys.modules, "probe must not import Torch or initialize CUDA"
    print(json.dumps({"torch_imported": False, "observations": observations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
