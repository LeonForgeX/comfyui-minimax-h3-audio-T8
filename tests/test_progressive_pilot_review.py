import pytest

from tools.build_progressive_pilot_review import render_template


TEMPLATE = '''
H3 渐进首采：第一组 A/B 审看
H3 渐进首采 · 第一组 A/B 审看
同模型、同新版 EMA B、同提示词和 seed
这是第一组短片，不是32秒验收。两条独立生成，人物和声音不要求逐像素/逐采样相同；请听古典音乐和“你在哪里”，看脸、嘴形、动作与闪烁。只播放所选一路声音，避免回声。
看完请回复：画面 A 更好 / B 更好 / 差不多 / 都不行；并分别说两条的音乐、人声、音量和口型是否正常。允许画面通过但声音不通过，不会自动替你判定。
路线暂不显示
此页已在独立 Chrome 中验证两路完整播放。
const hashes={A:'old',B:'old'},blobUrls=[];
crypto.subtle.digest
new Blob([bytes],{type:'video/mp4'})
两条同步播放中，仅所选一路发声
readyState>=2
'''


def test_i2va_review_preserves_original_media_player_and_corrects_task_instructions():
    html = render_template(TEMPLATE, {"A": "a" * 64, "B": "b" * 64}, "I2VA")
    assert '"A": "' + "a" * 64 + '"' in html
    assert '"B": "' + "b" * 64 + '"' in html
    assert "同一首帧" in html and "本组无对白，不验收口型" in html
    assert "古典音乐" not in html and "你在哪里" not in html
    assert "crypto.subtle.digest" in html and "new Blob([bytes],{type:'video/mp4'})" in html
    assert "两条同步播放中，仅所选一路发声" in html
    assert "此页已在独立 Chrome 中验证两路完整播放" not in html
    assert "readyState>=2" in html


def test_t2va_template_keeps_speech_review_and_does_not_infer_winner():
    html = render_template(TEMPLATE, {"A": "a" * 64, "B": "b" * 64}, "T2VA")
    assert "古典音乐" in html and "你在哪里" in html
    assert "路线暂不显示" in html and "不会自动替你判定" in html


@pytest.mark.parametrize("fault", ["missing_hash", "duplicate_hash", "missing_task_text", "unknown_task"])
def test_incompatible_templates_require_review_instead_of_silent_wrong_instructions(fault):
    html = TEMPLATE
    task = "I2VA"
    if fault == "missing_hash":
        html = html.replace("const hashes=", "const oldHashes=")
    elif fault == "duplicate_hash":
        html += "const hashes={},blobUrls=[];"
    elif fault == "missing_task_text":
        html = html.replace("H3 渐进首采：第一组 A/B 审看", "changed")
    else:
        task = "FL2VA"
    with pytest.raises(ValueError):
        render_template(html, {"A": "a" * 64, "B": "b" * 64}, task)


@pytest.mark.parametrize("hashes", [{"A": "a"*64}, {"A": "a"*64, "B": "bad"},
                                    {"A": "a"*64, "B": "b"*64, "C": "c"*64},
                                    {"A": "a"*64, "B": "</script>"}])
def test_invalid_or_injectable_media_identities_are_rejected(hashes):
    with pytest.raises(ValueError, match="SHA-256"):
        render_template(TEMPLATE, hashes, "T2VA")
