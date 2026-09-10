"""Independent CPU evidence gate for a full 0.5MP latent from a completed H3 run."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))
from trt_latent_capture import tensor_identity  # noqa: E402
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from progressive_probe_control import summarize_execution_events  # noqa: E402
from progressive_pilot_analysis import verify_execution_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(strict=True)
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    def load(name):
        return json.loads((root / name).read_text(encoding="utf8"))
    terminal, expected = load("terminal.json"), load("environment-expected.json")
    graph, history = load("generation/prompt.json"), load("generation/history.json")
    reports, capture, media = load("reports.json"), load("trt-latent-capture.json"), load("media-audit.json")
    if (terminal["status"] != "generation_media_validated_unreviewed" or terminal["server_stop"]["owned_children_remaining"]
            or terminal.get("resource_monitor_failure") or terminal["resource_guard"].get("reason")):
        raise ValueError("Generation/process/resource evidence incomplete")
    if not history["status"]["completed"]:
        raise ValueError("Core generation history is not successful")
    if graph != expected["pilot_graphs"][terminal["case"]]:
        raise ValueError("Actual generation graph changed")
    from trt_vdn_probe import is_vdn, report_nodes
    final_sampler = ["67", 0] if is_vdn(graph) else ["10", 0]
    if (graph["11"]["inputs"]["av_latent"] != ["105", 0]
            or graph["105"]["class_type"] != "T8TRTLatentCapture"
            or graph["105"]["inputs"]["av_latent"] != final_sampler):
        raise ValueError("Capture is not the unchanged final sampler-to-decoder link")
    for name, digest in expected["sources"].items():
        if digest_file(PROJECT / name) != digest:
            raise ValueError("Generation source changed before audit")
    verify_execution_report(terminal["case"], graph, reports)
    for name, node in report_nodes(graph).items():
        if reports[name] != json.loads(history["outputs"][node]["text"][0]):
            raise ValueError("Saved report differs from actual history: " + name)
    events = [json.loads(line) for line in (root / "generation/events.jsonl").read_text(encoding="utf8").splitlines()]
    timing = load("generation/timing.json")
    receipt = summarize_execution_events(events, load("generation/submission.json")["prompt_id"], graph, timing["elapsed_seconds"])
    if not receipt["complete_uncached_graph"]:
        raise ValueError("Actual graph did not run every node")
    if json.loads(history["outputs"]["106"]["text"][0]) != capture:
        raise ValueError("Capture report not bound to successful Core execution")
    for kind, key in (("video", "latent_tensor"), ("audio", "audio_latent")):
        item = capture["files"][kind]
        path = Path(item["path"]).resolve(strict=True)
        if not path.is_relative_to(root / "output") or digest_file(path) != item["sha256"]:
            raise ValueError("Capture file identity/path changed")
        with safe_open(path, framework="pt", device="cpu") as file:
            if kind == "video" and "latent_format_version_0" not in file.keys():
                raise ValueError("Video serialization scale is unknown")
            value = file.get_tensor(key)
            if tensor_identity(value) != item["tensor"] or not bool(torch.isfinite(value).all()):
                raise ValueError("Actual captured tensor differs from sampler record")
    if capture["files"]["video"]["tensor"]["shape"] != [1, 24, 22, 32, 64]:
        raise ValueError("This gate requires full1024x512x73 latent, not another sample size")
    if media["status"] != "mechanical_media_pass_human_quality_unverified" or not media["strict_av_decode"]:
        raise ValueError("Native source media unverified")
    source = Path(media["file"]["path"]).resolve(strict=True)
    if not source.is_relative_to(root / "output") or digest_file(source) != media["file"]["sha256"]:
        raise ValueError("Native source media changed")
    snapshot = root / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for relative, digest in expected["sources"].items():
        original = (PROJECT / relative).resolve(strict=True)
        saved = (snapshot / relative).resolve()
        if not original.is_relative_to(PROJECT) or not saved.is_relative_to(snapshot):
            raise ValueError("Source snapshot path escaped")
        saved.parent.mkdir(parents=True, exist_ok=True)
        if not saved.exists():
            with saved.open("xb") as file:
                file.write(original.read_bytes())
        if digest_file(saved) != digest:
            raise ValueError("Frozen source identity mismatch")
    report = {"status": "actual_0p5mp_sampler_latent_and_native_media_bound", "case": terminal["case"],
              "capture": capture, "native_media": media["file"], "generation": receipt,
              "cuda_initialized": torch.cuda.is_initialized(), "audit_source_sha256": digest_file(__file__),
              "evidence": {name: digest_file(root / name) for name in
                           ("terminal.json", "environment-expected.json", "generation/prompt.json", "generation/history.json",
                            "generation/events.jsonl", "trt-latent-capture.json", "media-audit.json", "reports.json")},
              "limits": "New actual sampling for reusable latent, not decoder speed/human qualification. No repeated seed search."}
    if report["cuda_initialized"]:
        raise RuntimeError("CPU capture audit initialized CUDA")
    report["route"] = "VDN8_learned2x_VDN4_audio_locked" if is_vdn(graph) else terminal["case"]
    report["sources_frozen"] = len(expected["sources"])
    write_new_json(root / "independent-capture-audit.json", report)
    print(json.dumps({"status": report["status"], "case": terminal["case"]}))


if __name__ == "__main__":
    main()
