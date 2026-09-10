"""Actual abrupt-exit test of the publication boundary, not an H3 process kill."""
import hashlib
import importlib
import os
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("outpaint_delivery_test_package")
package.__path__ = [str(root / "h3_t8"), str(root)]
sys.modules[package.__name__] = package
publish_with_delivery_report = importlib.import_module(
    "outpaint_delivery_test_package.video_outpaint_delivery").publish_with_delivery_report

directory = Path(sys.argv[1])
private, target = directory / "private.mp4", directory / "result.mp4"
private.write_bytes(b"fake media fixture; no codec/model validation")
report = {"schema": "t8.h3.video_outpaint.final_file/v1", "path": str(target),
          "sha256": hashlib.sha256(private.read_bytes()).hexdigest()}
calls = 0


def interrupt():
    global calls
    calls += 1
    if calls == 2:
        os._exit(47)


publish_with_delivery_report(private, target, report, interrupt_check=interrupt)
raise AssertionError("worker should have exited between the two links")
