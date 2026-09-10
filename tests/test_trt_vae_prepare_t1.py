import hashlib
import os
from pathlib import Path

import pytest

from h3_audio_t8_pkg import trt_vae_prepare_t1 as prepare


def test_trace_input_is_finite_without_fp16_linspace_overflow():
    import torch
    from h3_audio_t8_pkg.trt_vae_prepare_t1_worker import trace_pixels
    pixels = trace_pixels()
    assert pixels.device.type == 'cpu' and pixels.dtype == torch.float16
    assert pixels.shape == (1,3,1,256,256) and torch.isfinite(pixels).all()
    assert pixels.min() == -1 and pixels.max() == 1


@pytest.mark.skipif(os.name != 'nt',reason='Windows no-replace rename')
def test_t1_publication_is_exact_and_never_overwrites_user_file(tmp_path,monkeypatch):
    source = tmp_path/'source.onnx'
    source.write_bytes(b'CPU synthetic graph')
    monkeypatch.setattr(prepare,'ENCODER_T1_MODEL_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    dest = tmp_path/'output.onnx'
    prepare.publish_graph(source,dest)
    assert source.read_bytes() == dest.read_bytes()
    dest.write_bytes(b'user file')
    with pytest.raises(ValueError,match='preserving'):
        prepare.publish_graph(source,dest)
    assert dest.read_bytes() == b'user file'
    assert not list(tmp_path.glob('*.partial'))


@pytest.mark.skipif(os.name != 'nt',reason='Windows entry')
def test_prepared_graph_reuses_matching_bytes_without_gpu_or_reexport(tmp_path,monkeypatch):
    destination = tmp_path/'t1.onnx'
    destination.write_bytes(b'CPU graph')
    monkeypatch.setattr(prepare,'ENCODER_T1_MODEL_SHA',hashlib.sha256(destination.read_bytes()).hexdigest())
    report = prepare.main(['--native-vae',str(tmp_path/'not-opened'),'--core-directory',str(tmp_path),
                           '--output',str(destination),'--run-dir',str(tmp_path/'not-created'),'--execute'])
    assert report['status'] == 'existing_pinned_t1_graph_reused_no_export'
    assert not (tmp_path/'not-created').exists()
    destination.write_bytes(b'changed')
    with pytest.raises(ValueError,match='not overwriting'):
        prepare.main(['--native-vae','not-opened','--core-directory',str(tmp_path),
                      '--output',str(destination),'--run-dir',str(tmp_path/'not-created')])


def test_public_t1_preparation_has_no_research_reference_or_tools_import():
    root = Path(__file__).resolve().parents[1]
    for name in ('trt_vae_prepare_t1.py','trt_vae_prepare_t1_worker.py'):
        text = (root/'h3_t8'/name).read_text(encoding='utf8')
        assert 'from tools' not in text and 'source_rgb8' not in text and 'acceleration-research' not in text
