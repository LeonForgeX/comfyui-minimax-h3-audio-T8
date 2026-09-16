from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def test_review_template_never_seeks_to_exact_eof_and_recovers_visible_frame():
    html = (PROJECT / "tools" / "candidate_combined_review.html").read_text(
        encoding="utf-8"
    )

    assert "function safeEnd(v)" in html
    assert "duration-Number(seek.step)" in html
    assert "v.currentTime=safeEnd(v)" in html
    assert "回到开头" in html
    assert "v.preload='auto'" in html
    assert 'v.style.aspectRatio=`${clip.width}/${clip.height}`' in html
    assert "先看24秒事件时间线" not in html
    assert "常规Topaz已完成" not in html
