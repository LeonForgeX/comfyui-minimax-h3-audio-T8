from pathlib import Path

import pytest

from tools import build_trt_vae_review as review


def test_seven_inline_pairs_no_implicit_acceptance_or_long_claim():
    sections = [{'id':f'pair-{i}','title':f'case{i}','description':'<script>not executable</script>','accepted':False} for i in range(1,8)]
    page = review.render('TRT<review>','identity123',sections)
    assert page.count('<video ') == 14 and page.count('<canvas ') == 14
    assert page.count('data-review="true"') == 7
    assert page.count('data-answer="lipsync"') == 5
    assert '<script>not executable</script>' not in page
    assert 'TRT&lt;review&gt;' in page
    assert '长片测试按你的要求取消' in page
    assert '没有明显收益' in page
    assert 'TRT未安装' not in page and '这组已收到你的整体通过' not in page
    assert 'autoplay' not in page and 'window.open' not in page
    assert 't8.trt-vae.combined-human.v1' in page
    assert 'trt_vae_combined_human_review.json' in page
    assert 'identity123' in page and 'REVIEW_ID_PLACEHOLDER' not in page


def test_review_refuses_tampered_or_out_of_scope_media(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'RESEARCH',tmp_path/'scope')
    scope = tmp_path/'scope'
    scope.mkdir()
    media = scope/'video.mp4'
    media.write_bytes(b'media fixture')
    identity = {'path':str(media),'sha256':review.digest_file(media)}
    assert Path(review.verify_identity(identity)['path']) == media
    media.write_bytes(b'changed')
    with pytest.raises(ValueError):
        review.verify_identity(identity)
    outside = tmp_path/'outside.mp4'
    outside.write_bytes(b'outside')
    with pytest.raises(ValueError):
        review.verify_identity({'path':str(outside),'sha256':review.digest_file(outside)})
