"""Independent CPU postflight for saved-encoder fixed-sampling A/B evidence."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_pilot_analysis import load_run  # noqa: E402
from trt_encoder_condition_probe import PROJECT, RESEARCH, digest, evidence_identity  # noqa: E402


def verify_reference_conditioning(report, expected_latent, reference_kind="image"):
    value = deepcopy(report)
    encoded = json.dumps(value["conditioning"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != value.pop("conditioning_sha256"):
        raise ValueError("Conditioning digest mismatch")
    blocks = value["conditioning"]
    if len(blocks) != 1 or len(blocks[0]) != 2:
        raise ValueError("Unexpected conditioning block layout")
    if reference_kind == "video":
        metadata = blocks[0][1]
        refs = metadata.get("minimax_refs", [])
        if (metadata.get("minimax_keyframes") or len(refs) != 1
                or refs[0].get("kind") != "video" or refs[0].get("latent_t") != 22
                or refs[0].get("latent_h") != 32 or refs[0].get("latent_w") != 64
                or refs[0].get("ref_audio_t") != 0 or refs[0].get("audio_latent") is not None
                or refs[0].get("latent") != expected_latent):
            raise ValueError("Actual complete video reference is not the saved encoder output")
        refs[0]["latent"] = "ONLY_PERMITTED_ENCODER_VARIABLE"
        return value
    if reference_kind != "image":
        raise ValueError("Unknown reference kind")
    frames = blocks[0][1]["minimax_keyframes"]
    if len(frames) != 1 or frames[0]["resolved_frame_index"] != 0 or frames[0]["latent"] != expected_latent:
        raise ValueError("Actual first-frame condition is not the saved encoder output")
    frames[0]["latent"] = "ONLY_PERMITTED_ENCODER_VARIABLE"
    return value


def audit_pair(native_root, trt_root):
    from safetensors import safe_open
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("Pair audit must be CPU-only")
    native, trt = (load_run(path) for path in (native_root, trt_root))
    reference_kind = native["terminal"].get("encoder_reference_kind", "image")
    if reference_kind not in ("image", "video") or trt["terminal"].get("encoder_reference_kind", "image") != reference_kind:
        raise ValueError("Reference kind mismatch")
    normalized_graphs, normalized_conditions, ranges, summaries = [], [], [], []
    for run, backend in ((native, "native"), (trt, "trt")):
        root = Path(run["root"])
        if not root.is_relative_to(RESEARCH) or run["terminal"].get("encoder_backend") != backend:
            raise ValueError("Unexpected encoder run or location")
        expected = run["environment"]["encoder_evidence"]
        if evidence_identity(expected["root"]) != expected:
            raise ValueError("Original encoder evidence changed")
        report = json.loads((root / "encoder-reference-consumed.json").read_text())
        history = json.loads((root / "generation/history.json").read_text())
        if (report != json.loads(history["outputs"]["109"]["text"][0])
                or report.get("status") != "audited_reference_encoding_consumed"
                or report.get("backend") != backend or report.get("actual_encode_calls") != 1
                or report.get("identity") != expected):
            raise ValueError("Encoder consumption differs from actual history")
        graph = deepcopy(run["graph"])
        reference_input = "first_frame" if reference_kind == "image" else "ref_videos.ref_video_0"
        if (graph["107"]["inputs"]["backend"] != backend or graph["11"]["inputs"]["video_vae"] != ["1", 0]
                or graph["6"]["inputs"].get(reference_input) != ["107", 1]
                or graph["6"]["inputs"]["video_vae"] != ["107", 0]):
            raise ValueError("Reference/decode wiring changed")
        if reference_kind == "video" and (
                graph["107"]["inputs"].get("reference_kind") != "video"
                or "first_frame" in graph["6"]["inputs"]
                or report.get("reference_kind") != "video"
                or report.get("reference_frames") != 73 or report.get("reference_latent_tokens") != 22):
            raise ValueError("Video replay was truncated or relabelled as an image")
        graph["107"]["inputs"]["backend"] = "ENCODER_VARIABLE"
        graph["18"]["inputs"].pop("filename_prefix")
        normalized_graphs.append(graph)
        with safe_open(Path(expected["root"]) / (backend + "-latents.safetensors"), framework="pt", device="cpu") as saved:
            latent = saved.get_tensor(reference_kind).contiguous()
        identity = {"tensor_shape": list(latent.shape), "dtype": str(latent.dtype),
                    "sha256": hashlib.sha256(latent.view(torch.uint8).numpy().tobytes()).hexdigest()}
        normalized_conditions.append(verify_reference_conditioning(run["reports"]["conditioning"], identity, reference_kind))
        samples = [json.loads(row)["monotonic"] for row in (root / "resources.jsonl").read_text().splitlines()]
        if not samples or samples != sorted(samples):
            raise ValueError("Missing or unordered resource timestamps")
        ranges.append((samples[0], samples[-1]))
        sources = run["environment"]["sources"]
        for relative, sha in sources.items():
            path = (PROJECT / relative).resolve(strict=True)
            if not path.is_relative_to(PROJECT) or digest(path) != sha:
                raise ValueError("Source changed before snapshot: " + relative)
        target = root / "source-snapshot"
        target.mkdir(exist_ok=True)
        for relative, sha in sources.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if digest(path) != sha:
                    raise ValueError("Existing source snapshot mismatch")
            else:
                with path.open("xb") as output:
                    output.write((PROJECT / relative).read_bytes())
            if digest(path) != sha:
                raise ValueError("Copied source snapshot mismatch")
        summaries.append({"backend": backend, "root": str(root), "media": run["media"],
                          "latent_identity": identity, "sources_frozen": len(sources)})
    if normalized_graphs[0] != normalized_graphs[1] or normalized_conditions[0] != normalized_conditions[1]:
        raise ValueError("More than the reference encoder changed between the two runs")
    for key in ("core", "sources", "encoder_evidence"):
        if native["environment"][key] != trt["environment"][key]:
            raise ValueError("Core/source/evidence mismatch")
    if native["assets"] != trt["assets"]:
        raise ValueError("Model/input asset mismatch")
    if ranges[0][1] >= ranges[1][0]:
        raise ValueError("The GPU runs were not sequential in the same monotonic clock")
    for key in ("torch_version", "torch_cuda", "device"):
        if native["live"][key] != trt["live"][key]:
            raise ValueError("Runtime mismatch")
    if native["terminal"]["resource_guard"]["gpu_uuid"] != trt["terminal"]["resource_guard"]["gpu_uuid"]:
        raise ValueError("GPU mismatch")
    return {"status": "encoder_fixed_sampler_pair_evidence_pass_human_review_pending", "runs": summaries,
            "reference_kind": reference_kind,
            "same_source_pixels_and_text_conditioning": True, "only_reference_latent_changed": True,
            "gpu_execution_serial": True, "cuda_initialized": torch.cuda.is_initialized(),
            "scope": ("Complete0.5MP short " + ("73-frame video reference" if reference_kind == "video" else "I2VA")
                      + "; independent sampling means generated audio need not be identical. No speech/lip-sync claim, encoder-speed or integrated TRT-runtime benchmark, or long-video qualification.")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--trt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(RESEARCH) or args.output.exists():
        raise ValueError("Use a new audit report in research")
    result = audit_pair(args.native, args.trt)
    with args.output.open("x", encoding="utf8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": result["status"], "report": str(args.output)}))


if __name__ == "__main__":
    main()
