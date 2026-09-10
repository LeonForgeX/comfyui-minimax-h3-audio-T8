from pathlib import Path

import pytest
import torch

from tools.trt_vae_public_text_worker import chart
from tools.audit_trt_vae_public_text import metrics


def test_text_fixture_layout_and_repeatability():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if not font.is_file():
        pytest.skip('Fixture uses an explicit installed Microsoft YaHei font, not redistributed')
    a, rows = chart(font)
    b, other = chart(font)
    assert a.size == (1024,512) and a.mode == 'RGB'
    assert a.tobytes() == b.tobytes() and rows == other and len(rows) == 14
    assert any('中文' in r['text'] for r in rows)
    for row in rows:
        x0,y0,x1,y1 = row['bbox']
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 512


def test_text_metrics_never_label_pixel_difference_as_readability():
    a = torch.zeros(1,4,4,3)
    assert metrics(a,a)['bit_exact']
    report = metrics(a,a+.1)
    assert report['psnr_db'] == pytest.approx(20)
    assert not report['bit_exact'] and 'quality_pass' not in report
