import hashlib
import json
import shutil

import pytest

from tools import outpaint_probe_cases as cases
from tools import run_video_outpaint_t8_probe as probe
from h3_audio_t8_pkg.video_outpaint_plan import canonical
from test_video_outpaint_media import _clip
from test_video_outpaint_pixel_receipt import fixture, seal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _case(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "ROOT", tmp_path)
    # Use the actual module while relocating only the allowed artifact root.
    from h3_audio_t8_pkg import video_outpaint_plan
    monkeypatch.setattr(cases, "plan_module", lambda: video_outpaint_plan)
    folder = tmp_path / "artifacts"
    folder.mkdir()
    source = folder / "source.mp4"
    _clip(source, 32, 32, frames=80, sound=True)
    payload = {"schema": "t8.outpaint.probe_case/v1", "source_path": "source.mp4", "source_sha256": sha(source),
        "plan_request": {"source_sha256": sha(source).lower(), "width": 32, "height": 32, "frame_count": 80,
                         "aspect": "custom", "top": 32, "bottom": 32, "window_frames": 56},
        "prompt": "scene", "shot_prompts": [], "color_match": True}
    path = folder / "case.json"
    path.write_text(json.dumps(payload))
    return path, payload, source


def test_case_drives_actual_audio_geometry_and_two_window_graph(tmp_path, monkeypatch):
    path, _, source = _case(tmp_path, monkeypatch)
    case = cases.load_case(path, shutil.which("ffprobe"), sha)
    assert case["audio_tracks"] == 1 and len(case["plan"]["shots"][0]["windows"]) == 2
    graph = probe.build_prompt(native_noise=True, case=case)
    assert graph["10"]["inputs"]["window_frames"] == "56"
    assert graph["13"]["inputs"]["color_match"] is True
    assert graph["11"]["inputs"]["prompt"] == "scene"
    media = cases.strict_media(source, shutil.which("ffmpeg"), shutil.which("ffprobe"), 1, sha)
    assert media["strict_decode_exit"]["audio"] == 0
    assert media["decoder_threads"] == 4
    assert media["decoder_error_policy"] == "xerror+err_detect_explode"


@pytest.mark.parametrize("mutation", ["file", "geometry", "plan_sha", "prompts"])
def test_mismatched_case_is_rejected_before_gpu(tmp_path, monkeypatch, mutation):
    path, payload, source = _case(tmp_path, monkeypatch)
    if mutation == "file":
        source.write_bytes(b"changed")
    elif mutation == "geometry":
        payload["plan_request"]["frame_count"] = 81
    elif mutation == "plan_sha":
        payload["plan_request"]["source_sha256"] = "a"*64
    else:
        payload["shot_prompts"] = ["extra", "extra"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        cases.load_case(path, shutil.which("ffprobe"), sha)


def _delivery(mode):
    plan, receipt = fixture(mode)
    report = {"source_sha256": "a"*64, "sha256": "b"*64, "pre_encode_pixel_evidence_verified": True,
              "candidate_sha256": "b"*64, "plan_sha256": plan["plan_sha256"],
              "pixel_receipt": receipt, "pixel_receipt_sha256": receipt["receipt_sha256"],
              "source_mode": mode, "source_exact_before_encoding": mode == "preserve_source",
              "source_reconstructed": mode == "joint_decode",
              "media": {flag: True for flag in ("audio_packet_payload_exact", "audio_packet_timeline_exact",
                  "audio_decoded_pcm_exact", "audio_decoded_timeline_exact", "strict_ffmpeg_decode")}}
    report["media"]["decoded_video_frames"] = plan["output"]["frames"]
    return plan, report


def _sealed(report):
    return {**report, "delivery_report_sha256": hashlib.sha256(canonical(report).encode()).hexdigest()}


@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode"])
def test_probe_requires_passed_audio_evidence_in_matching_receipt(tmp_path, mode):
    plan, report = _delivery(mode)
    path = tmp_path / "report.json"
    def write():
        path.write_text(canonical(_sealed(report)))
    write()
    assert cases.verify_delivery(path, "b"*64, "a"*64, expected_plan=plan, expected_source_mode=mode)
    assert cases.verify_delivery(path, "b"*64, "a"*64, plan["plan_sha256"])
    report["media"]["audio_decoded_pcm_exact"] = False
    write()
    with pytest.raises(ValueError, match="validation did not pass"):
        cases.verify_delivery(path, "b"*64, "a"*64)


@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode"])
@pytest.mark.parametrize("change", ["other_mode", "missing", "digest", "source_exact", "top_claim", "geometry", "frames", "evidence_bool"])
def test_resealed_false_or_mixed_pixel_evidence_is_rejected(mode, change):
    plan, report = _delivery(mode)
    receipt = report["pixel_receipt"]
    if change == "other_mode":
        report["source_mode"] = "joint_decode" if mode == "preserve_source" else "preserve_source"
    elif change == "missing":
        report.pop("pixel_receipt")
    elif change == "digest":
        report["pixel_receipt_sha256"] = "0"*64
    elif change == "source_exact":
        receipt["source_exact_before_encoding"] = not receipt["source_exact_before_encoding"]
    elif change == "top_claim":
        report["source_reconstructed"] = not report["source_reconstructed"]
    elif change == "geometry":
        receipt["width"] += 1
    elif change == "frames":
        report["media"]["decoded_video_frames"] += 1
    else:
        report["pre_encode_pixel_evidence_verified"] = 1
    if change in ("source_exact", "geometry"):
        report["pixel_receipt_sha256"] = seal(receipt)["receipt_sha256"]
    with pytest.raises(ValueError):
        cases.verify_delivery_report(_sealed(report), "b"*64, "a"*64,
            expected_plan=plan, expected_source_mode=mode)


def test_old_exact_source_report_can_omit_new_top_level_mode_fields():
    plan, report = _delivery("preserve_source")
    for key in ("source_mode", "source_exact_before_encoding", "source_reconstructed"):
        report.pop(key)
    assert cases.verify_delivery_report(_sealed(report), "b"*64, "a"*64, expected_plan=plan)


def test_contract_import_does_not_load_torch_in_a_fresh_process():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c",
        "from tools import outpaint_probe_cases as c; from pathlib import Path; import sys; "
        "c.ROOT = Path('nonexistent-artifact-root'); "
        "c.pixel_receipt_module(); assert 'torch' not in sys.modules"],
        cwd=cases.ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
