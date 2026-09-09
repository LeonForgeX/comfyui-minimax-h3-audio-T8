"""Explicit read-only CPU probe of an existing real preparation cache."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan-document", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    package = types.ModuleType("t8_prepared_reload_probe")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    media = importlib.import_module(package.__name__ + ".video_outpaint_media")
    reload_module = importlib.import_module(package.__name__ + ".video_outpaint_prepared_reload")
    from comfy_api.input_impl import VideoFromFile
    document = json.loads(args.plan_document.read_text(encoding="utf-8"))
    plan = document.get("plan", document)
    started = time.monotonic()
    report = {"status": "running", "source": str(args.source.resolve()),
              "cache_root": str(args.cache_root.resolve()), "perceptual_acceptance": False}
    try:
        inspection = media.inspect_outpaint_source(VideoFromFile(str(args.source.resolve())))
        prepared, evidence = reload_module.load_prepared_outpaint(
            {"inspection": inspection, "plan": plan}, args.cache_root)
        report.update(status="prepared_cache_reload_pass_not_generation_acceptance", evidence=evidence,
            source_frames=plan["source"]["frames"], source_sha256=inspection["sha256"],
            source_has_audio=prepared["audio"].store.has_audio,
            source_audio_chunks=len(prepared["audio"].store.snapshot()["chunks"]),
            saved_conditioning_shots=len(prepared["conditioning"].data["shots"]))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic()-started, 3)
        with (args.run_root / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
