from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import build_video_outpaint_hardcut_review as review


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def test_hardcut_review_rejects_nonterminal_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mechanically complete"):
        review.build_review(report, tmp_path / "missing.json", tmp_path / "review")


def test_hardcut_review_builds_bound_synced_page(tmp_path, monkeypatch):
    source_payload, candidate_payload = b"source", b"candidate"
    source, candidate = tmp_path / "source.mp4", tmp_path / "candidate.mp4"
    source.write_bytes(source_payload)
    candidate.write_bytes(candidate_payload)
    windows = {"status": "sampled", "committed": [
        {"shot": 0, "window": 0, "sha256": "a" * 64},
        {"shot": 1, "window": 0, "sha256": "b" * 64},
    ]}
    window_path = tmp_path / "windows.json"
    window_path.write_text(json.dumps(windows), encoding="utf-8")
    color_sha = review._sha256(Path(review.__file__).resolve().parents[1] / "h3_t8/video_outpaint_color.py")
    plan = {
        "plan_sha256": "c" * 64,
        "request": {"cut_frames": [39]},
        "shots": [
            {"start": 0, "windows": [{"context_video_latents": 0, "context_audio_latents": 0}]},
            {"start": 39, "windows": [{"context_video_latents": 0, "context_audio_latents": 0}]},
        ],
    }
    report = {
        "status": "media_pass_human_review_pending",
        "source_sha256": _sha(source_payload),
        "case": {"source_path": str(source), "plan": plan},
        "media": {"path": str(candidate), "sha256": _sha(candidate_payload)},
        "delivery": {"report": {
            "color_match_enabled": True,
            "pixel_receipt": {"window_manifest_sha256": review._sha256(window_path)},
            "composition_implementation_sha256": {"video_outpaint_color.py": color_sha},
        }},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(review, "_video_screen", lambda *_args, **_kwargs: {
        "hard_cut": {
            "source_has_mechanical_hard_cut": True,
            "candidate_preserves_cut_transition": True,
            "post_cut_closer_to_current_source_than_previous_source": True,
        }
    })
    built = review.build_review(report_path, window_path, tmp_path / "review")
    page = Path(built["page"]).read_text(encoding="utf-8")
    assert "同步播放" in page and "第 39 帧" in page
    assert '<video id="source" controls muted' in page
    assert '<video id="outpaint" controls preload=' in page
    assert "只播放右侧扩画成片的音频" in page
    assert "JSON.stringify(out,null,2)+'\\n'" in page
    assert built["manifest"]["two_independent_shots"]
    assert built["manifest"]["color_reset_evidence"]["future_reports_emit_explicit_shot_reset_count"]
    assert (tmp_path / "review/public/source.mp4").read_bytes() == source_payload
    assert (tmp_path / "review/public/outpaint.mp4").read_bytes() == candidate_payload
