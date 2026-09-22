"""Summarize existing, homogeneous timing receipts; no inference or downloads.

Usage: python tools/summarize_h3_measurements.py cold-1.json cold-2.json cold-3.json
Run separately for cold/warm groups. Redirect stdout to a chosen report path.
The producer must use Measurement or explicit SynchronizedCudaMeasurement around
real stages (model_loading, sampling, video_vae, audio_vae, delivery).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--minimum-runs", type=int, default=3)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "h3_t8" / "acceleration_measurement.py"
    spec = importlib.util.spec_from_file_location("t8_measurement", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    print(json.dumps(module.summarize_repeated_measurements(reports, minimum_runs=args.minimum_runs),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
