"""Pure input-case validation for serial outpaint probes; no Torch/GPU imports."""
import hashlib
import importlib.util
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def pixel_receipt_module():
    # Import only the two pure contracts, never the node package's __init__.
    name = "_t8_outpaint_probe_contracts"
    if name not in sys.modules:
        package = types.ModuleType(name)
        # Artifact roots may be relocated by harness tests; code identity may not.
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "h3_t8"), str(Path(__file__).resolve().parents[1])]
        sys.modules[name] = package
    return importlib.import_module(name + ".video_outpaint_pixel_receipt")


def candidate_finish_inputs(compose_inputs):
    """Carry an archived Compose policy explicitly into a new probe candidate.

    A historical graph without this field used exact-source preservation; a
    current node default must not silently reinterpret that old experiment.
    """
    mode = pixel_receipt_module().validate_source_mode(
        compose_inputs.get("source_mode", "preserve_source"))
    geometry = compose_inputs.get("geometry_align", False)
    if not isinstance(geometry, bool):
        raise ValueError("geometry_align must be boolean")
    return {"source_mode": mode, "geometry_align": geometry}


def plan_module():
    spec = importlib.util.spec_from_file_location("outpaint_probe_plan", ROOT / 'h3_t8/video_outpaint_plan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_audited_kj_source(path):
    """Use the exact runtime source policy without loading Torch or a model."""
    pixel_receipt_module()
    contract = importlib.import_module("_t8_outpaint_probe_contracts.video_outpaint_kj_contract")
    try:
        contract.kj_source_contract(Path(path).read_bytes())
    except (OSError, ValueError):
        return False
    return True


def load_case(path, ffprobe, sha):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    if (set(data) != {"schema", "source_path", "source_sha256", "plan_request", "prompt", "shot_prompts", "color_match"}
            or data["schema"] != "t8.outpaint.probe_case/v1"):
        raise ValueError("invalid outpaint probe case fields/schema")
    source = (path.parent / data["source_path"]).resolve(strict=True)
    source.relative_to(ROOT / "artifacts")
    digest = sha(source)
    if digest != data["source_sha256"].upper():
        raise ValueError("probe case source SHA mismatch")
    plan = plan_module().build_outpaint_plan(**data["plan_request"])
    if plan["source"]["sha256"].upper() != digest:
        raise ValueError("probe case plan is bound to another source")
    if not isinstance(data["prompt"], str) or not isinstance(data["color_match"], bool):
        raise ValueError("invalid case prompt/color flag")
    prompts = data["shot_prompts"]
    if not isinstance(prompts, list) or (prompts and (len(prompts) != len(plan["shots"]) or any(not isinstance(p, str) for p in prompts))):
        raise ValueError("case needs one string prompt per shot or an empty list")
    result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(source)],
                            capture_output=True, text=True, check=True, timeout=180)
    streams = json.loads(result.stdout)["streams"]
    videos = [s for s in streams if s["codec_type"] == "video"]
    expected = plan["source"]
    if len(videos) != 1:
        raise ValueError("probe source needs exactly one video stream")
    video = videos[0]
    if (int(video["width"]), int(video["height"]), int(video["nb_read_frames"]), video["avg_frame_rate"]) != (
            expected["width"], expected["height"], expected["frames"], "24/1"):
        raise ValueError("actual probe source geometry/frame count/fps differs from case")
    if sha(source) != digest:
        raise ValueError("probe source changed during inspection")
    return {**data, "source_path": source, "source_sha256": digest, "plan": plan,
            "audio_tracks": sum(s["codec_type"] == "audio" for s in streams),
            "case_path": str(path), "case_sha256": sha(path)}


def strict_media(path, ffmpeg, ffprobe, audio_tracks, sha):
    result = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True, timeout=180)
    streams = json.loads(result.stdout)["streams"]
    if sum(s["codec_type"] == "audio" for s in streams) != audio_tracks:
        raise ValueError("output audio stream count differs from original source")
    mappings = [("video", ["-map", "0:v:0"]), ("joint", [])]
    if audio_tracks:
        mappings.append(("audio", ["-map", "0:a"]))
    exits = {}
    for label, mapping in mappings:
        checked = subprocess.run([ffmpeg, "-v", "error", "-threads", "4", "-xerror", "-err_detect", "explode",
                                  "-nostdin", "-i", str(path),
                                  *mapping, "-f", "null", "-"], capture_output=True, timeout=300)
        if checked.returncode or checked.stderr.strip():
            raise RuntimeError(f"strict {label} decode failed: {checked.stderr[-1000:]!r}")
        exits[label] = checked.returncode
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path),
            "streams": streams, "strict_decode_exit": exits, "decoder_threads": 4,
            "decoder_error_policy": "xerror+err_detect_explode"}


def verify_delivery_report(delivery, output_sha, source_sha, expected_plan_sha=None, *,
                           expected_plan=None, expected_source_mode=None):
    """Check a sealed report and its distinct pixel contract without loading models.

    New material runs supply the complete independent plan. Historical callers
    supply its digest and separately inspect output geometry; their receipt
    dimensions are checked for internal consistency, not independently proven.
    """
    sealed = dict(delivery)
    digest = sealed.pop("delivery_report_sha256", None)
    encoded = json.dumps(sealed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (digest != hashlib.sha256(encoded).hexdigest() or delivery.get("sha256", "").upper() != output_sha.upper()
            or delivery.get("source_sha256", "").upper() != source_sha.upper()
            or delivery.get("pre_encode_pixel_evidence_verified") is not True
            or (expected_plan_sha and delivery.get("plan_sha256") != expected_plan_sha)):
        raise ValueError("persisted delivery report does not bind this source, plan and output")
    media = delivery["media"]
    for flag in ("audio_packet_payload_exact", "audio_packet_timeline_exact", "audio_decoded_pcm_exact",
                 "audio_decoded_timeline_exact", "strict_ffmpeg_decode"):
        if media.get(flag) is not True:
            raise ValueError(f"delivery validation did not pass: {flag}")
    contract = pixel_receipt_module()
    mode = contract.validate_source_mode(delivery.get("source_mode", "preserve_source"))
    if expected_source_mode is not None and mode != contract.validate_source_mode(expected_source_mode):
        raise ValueError("delivery source mode differs from the requested mode")
    receipt = delivery.get("pixel_receipt")
    if not isinstance(receipt, dict) or delivery.get("pixel_receipt_sha256") != receipt.get("receipt_sha256"):
        raise ValueError("delivery pixel receipt is missing or its digest differs")
    if expected_plan is not None:
        plan = plan_module().validate_outpaint_plan(expected_plan)
        if (plan["plan_sha256"] != delivery["plan_sha256"]
                or plan["source"]["sha256"].upper() != source_sha.upper()):
            raise ValueError("delivery differs from the independently supplied plan")
    else:
        plan = {"plan_sha256": delivery["plan_sha256"], "output": {
            "frames": media.get("decoded_video_frames"), "width": receipt.get("width"),
            "height": receipt.get("height")}}
    contract.validate_pixel_receipt(receipt, plan=plan, source_sha256=source_sha.lower(),
        candidate_sha256=delivery.get("candidate_sha256"), source_mode=mode)
    exact = mode == "preserve_source"
    # Legacy exact-source reports predate the top-level mode fields.
    if (delivery.get("source_exact_before_encoding", exact) is not exact
            or delivery.get("source_reconstructed", False) is not (not exact)
            or media.get("decoded_video_frames") != receipt["frame_count"]):
        raise ValueError("delivery source preservation/reconstruction or frame claims conflict")
    return delivery


def verify_delivery(sidecar, output_sha, source_sha, expected_plan_sha=None, *,
                    expected_plan=None, expected_source_mode=None):
    delivery = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    return verify_delivery_report(delivery, output_sha, source_sha, expected_plan_sha,
        expected_plan=expected_plan, expected_source_mode=expected_source_mode)
