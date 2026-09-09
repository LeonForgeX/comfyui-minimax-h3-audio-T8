from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import build_video_outpaint_material_review as review
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan, canonical
from test_video_outpaint_pixel_receipt import seal


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def test_material_review_rejects_nonterminal_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mechanically complete"):
        review.build_review(report, tmp_path / "missing", tmp_path / "output",
                            title="test", focus="test")


@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode"])
def test_material_review_builds_hash_bound_page(tmp_path, monkeypatch, mode):
    source_payload, output_payload = b"source", b"output"
    source, candidate = tmp_path / "source.mp4", tmp_path / "output.mp4"
    source.write_bytes(source_payload)
    candidate.write_bytes(output_payload)
    windows = {"status": "sampled", "committed": [{"shot": 0, "window": 0}]}
    window_path = tmp_path / "windows.json"
    window_path.write_text(json.dumps(windows), encoding="utf-8")
    plan = build_outpaint_plan(source_sha256=_sha(source_payload), width=64, height=64,
        frame_count=39, aspect="custom", left=16, right=16, window_frames=39)
    exact = mode == "preserve_source"
    receipt = {
        "schema": "t8.h3.video_outpaint.pre_encode_pixels/v1" if exact else "t8.h3.video_outpaint.reconstructed_pixels/v1",
        "plan_sha256": plan["plan_sha256"], "source_sha256": _sha(source_payload),
        "candidate_sha256": "c"*64, "frame_count": 39, "width": 96, "height": 64,
        "source_exact_before_encoding": exact, "lossy_encoded_equality_claimed": False,
        "source_rgb_sha256": "b"*64, "window_manifest_sha256": review._sha256(window_path),
    }
    if exact:
        receipt["pasted_rgb_sha256"] = "b"*64
    else:
        receipt.update(source_mode=mode, source_reconstructed=True, reconstructed_source_rgb_sha256="d"*64)
    seal(receipt)
    delivery = {"source_sha256": _sha(source_payload), "sha256": _sha(output_payload),
        "candidate_sha256": "c"*64, "plan_sha256": plan["plan_sha256"],
        "source_mode": mode, "source_exact_before_encoding": exact, "source_reconstructed": not exact,
        "pre_encode_pixel_evidence_verified": True, "pixel_receipt": receipt,
        "pixel_receipt_sha256": receipt["receipt_sha256"], "color_match_report": {"shot_resets": 1},
        "media": {"decoded_video_frames": 39, **{flag: True for flag in (
            "audio_packet_payload_exact", "audio_packet_timeline_exact", "audio_decoded_pcm_exact",
            "audio_decoded_timeline_exact", "strict_ffmpeg_decode")}}}
    delivery["delivery_report_sha256"] = hashlib.sha256(canonical(delivery).encode()).hexdigest()
    report = {
        "status": "media_pass_human_review_pending", "source_sha256": _sha(source_payload),
        "case": {"source_path": str(source), "plan": plan},
        "media": {"path": str(candidate), "sha256": _sha(output_payload)},
        "delivery": {"report": delivery},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(review, "_video_screen", lambda *_args, **_kwargs: {
        "decoded_frame_count": 39, "black_regression_frames": [], "freeze_regression_frames": []})
    built = review.build_review(report_path, window_path, tmp_path / "review",
                                title="游戏审片", focus="看字幕。")
    page = Path(built["page"]).read_text(encoding="utf-8")
    assert "同步播放" in page and "看字幕" in page
    assert '<video id="source" controls muted' in page
    assert '<video id="outpaint" controls preload=' in page
    assert "只播放右侧扩画成片的音频" in page
    player = Path(built["page"]).with_name("player.js")
    script = player.read_text(encoding="utf-8")
    assert "JSON.stringify(out, null, 2)" in script
    assert "crypto.subtle.digest" in script and "URL.createObjectURL" in script
    assert 'src="source.mp4"' not in page and 'src="outpaint.mp4"' not in page
    assert 'id="crops"' in page and 'id="scrub"' in page
    assert built["manifest"]["player_sha256"] == review._sha256(player)
    assert built["manifest"]["committed_windows"] == 1
    assert built["manifest"]["source_exact_before_encoding"] is exact
    assert built["manifest"]["source_reconstructed"] is not exact
    assert built["manifest"]["source_mode"] == mode
    assert ("原片区域也经过 VAE 重建" in page) is (not exact)
    delivery["pixel_receipt"]["source_exact_before_encoding"] = not exact
    # A corrupt receipt must fail before screening/copying or publishing a page.
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        review.build_review(report_path, window_path, tmp_path / "bad-review", title="test", focus="test")
    assert not (tmp_path / "bad-review").exists()
