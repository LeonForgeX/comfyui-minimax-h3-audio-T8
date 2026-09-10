"""Audit three separately warmed A/B decode pairs; never starts GPU work."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from trt_vae_build import digest_file, write_new_json  # noqa: E402


PHASES = [
    "native_complete_warmup",
    "native_complete_decode",
    "trt_loading_after_native_unload",
    "trt_complete_warmup",
    "trt_complete_decode",
    "complete_rgb_saved",
]


def validate_round(request, result, audit, terminal):
    if (
        type(request.get("warmup_complete_decodes")) is not int
        or request["warmup_complete_decodes"] != 1
    ):
        raise ValueError("Benchmark requires one actual complete warmup per route")
    if (
        terminal["status"] != "worker_complete_result_requires_audit"
        or terminal["isolated"]["status"] != "complete"
        or terminal["isolated"]["active_after_cleanup"] != 0
        or audit["status"]
        != "complete_rgb_evidence_verified_not_human_or_speed_qualification"
        or audit["cuda_initialized"]
        or not audit["finite_normalized_all_frames"]
    ):
        raise ValueError("Execution, cleanup or independent quality audit incomplete")
    calls = audit["actual_calls_per_route"]
    if (
        type(calls) is not int
        or calls <= 0
        or result["warmup_complete_decodes_per_route"] != 1
        or any(
            result[name] != calls
            for name in (
                "native_calls",
                "trt_calls",
                "native_warmup_calls",
                "trt_warmup_calls",
            )
        )
    ):
        raise ValueError(
            "Actual warmup/measured tile calls do not cover the complete video"
        )
    events = result["events"]
    if [row["phase"] for row in events] != PHASES:
        raise ValueError("Missing or reordered complete warmup/measured phases")
    stamps = [row["monotonic"] for row in events]
    if any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
        for x in stamps
    ):
        raise ValueError("Invalid phase clock")
    if any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Phase clock must increase strictly")
    times = {
        kind: result[kind + "_measured_complete_decode_seconds"]
        for kind in ("native", "trt")
    }
    if any(
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not math.isfinite(x)
        or x <= 0
        for x in times.values()
    ):
        raise ValueError("Invalid measured duration")
    for kind in ("native", "trt"):
        warm = result[kind + "_warmup_seconds"]
        if (
            len(warm) != 1
            or not isinstance(warm[0], (int, float))
            or not math.isfinite(warm[0])
            or warm[0] <= 0
        ):
            raise ValueError("Missing real warmup duration")
        if result[kind + "_first_complete_decode_seconds"] is not None:
            raise ValueError("Warm timing incorrectly labelled first decode")
    return {
        **times,
        "start": stamps[0],
        "end": stamps[-1],
        "rgb_psnr_db": audit["rgb_psnr_db"],
    }


def summarize_rounds(rows):
    if len(rows) < 3 or any(a["end"] >= b["start"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Need at least three sequential non-overlapping A/B rounds")
    native = [row["native"] for row in rows]
    trt = [row["trt"] for row in rows]
    a, b = statistics.median(native), statistics.median(trt)
    return {
        "native_seconds": native,
        "trt_seconds": trt,
        "native_median_seconds": a,
        "trt_median_seconds": b,
        "native_range_seconds": [min(native), max(native)],
        "trt_range_seconds": [min(trt), max(trt)],
        "median_decode_speed_ratio": a / b,
        "median_decode_time_reduction_percent": (1 - b / a) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in args.runs]
    if len(set(roots)) != 3:
        raise ValueError(
            "Repeating the same experiment does not count as three measurements"
        )
    rows, identities, evidence = [], [], []
    for root in roots:

        def read(name):
            return json.loads((root / name).read_text())

        request, result = read("request.json"), read("result.json")
        audit, terminal = read("independent-video-audit.json"), read("terminal.json")
        for name, digest in audit["files"].items():
            if Path(name).name != name or digest_file(root / name) != digest:
                raise ValueError("Audited benchmark data changed")
        if read("resource-summary.json")["status"] != "observations_within_policy":
            raise ValueError("Resource guard not passed")
        row = validate_round(request, result, audit, terminal)
        identities.append(
            {
                "sources": request["sources"],
                "normalized_input": result["files"]["input-latent.safetensors"],
                "output_shape": result["output_shape"],
                "timing_scope": result["timing_scope"],
            }
        )
        rows.append(row)
        evidence.append(
            {
                "run": str(root),
                "audit_sha256": digest_file(root / "independent-video-audit.json"),
                "request_sha256": digest_file(root / "request.json"),
                "result_sha256": digest_file(root / "result.json"),
            }
        )
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(
            "The three rounds changed source/model/runtime/code, input, geometry or timing scope"
        )
    report = {
        "status": "three_warmed_alternating_decode_pairs_verified_not_end_to_end_or_human_qualification",
        **summarize_rounds(rows),
        "rounds": rows,
        "evidence": evidence,
        "identity": identities[0],
        "auditor_sha256": digest_file(__file__),
        "scope": "One complete warmup then one measured decode per backend in each A/B process; CPU half latent to CPU float RGB, transfers included; weights reloaded between rounds.",
        "limits": "Not full H3 generation speed, not cold disk, not exact process peak memory; loading/compilation/file encoding excluded from decode time. Quality requires human review.",
    }
    write_new_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("identity", "evidence")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
