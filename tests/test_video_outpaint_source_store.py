from __future__ import annotations

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_source_store import OutpaintSourceStore
from h3_audio_t8_pkg.video_outpaint_prepare import iter_encode_outpaint_source
from test_video_outpaint_prepare import PublicVAE, _plan


def _store(root, plan=None, vae="b"*64):
    return OutpaintSourceStore(root, plan or _plan(90), video_vae_sha256=vae)


def test_resume_source_preparation_and_window_reads_share_identical_overlaps(tmp_path):
    plan = _plan(90)
    pixels = torch.rand((90, 32, 32, 3))
    store = _store(tmp_path, plan)
    all_chunks = list(iter_encode_outpaint_source(PublicVAE(), lambda a, b: pixels[a:b], plan))
    for start, chunk, report in all_chunks[:2]:
        store.append(0, start, chunk)
    store = _store(tmp_path, plan)
    assert store.position() == (0, 10, 15)
    for start, chunk, report in all_chunks[2:]:
        store.append(0, start, chunk)
    assert store.position() is None
    whole = torch.cat([chunk for _, chunk, _ in all_chunks], dim=2)
    first, second = store.read_range(0, 0, 22), store.read_range(0, 5, 27)
    assert torch.equal(first, whole[:, :, :22])
    assert torch.equal(second, whole[:, :, 5:27])
    assert torch.equal(first[:, :, 5:], second[:, :, :17])


def test_wrong_vae_source_or_geometry_cannot_reuse_store(tmp_path):
    _store(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        _store(tmp_path, vae="c" * 64)
    with pytest.raises(ValueError, match="mismatch"):
        _store(tmp_path, _plan(89))


def test_gap_duplicate_and_unprepared_reads_rejected(tmp_path):
    store = _store(tmp_path)
    tensor = torch.ones((1, 24, 5, 2, 6))
    with pytest.raises(ValueError, match="serial"):
        store.append(0, 5, tensor)
    store.append(0, 0, tensor)
    with pytest.raises(ValueError, match="serial"):
        store.append(0, 0, tensor)
    with pytest.raises(ValueError, match="completely prepared"):
        store.read_range(0, 0, 7)


def test_corrupted_asset_or_manifest_is_not_silently_rebuilt(tmp_path):
    store = _store(tmp_path)
    store.append(0, 0, torch.ones((1, 24, 5, 2, 6)))
    asset = next(tmp_path.glob("video-*.safetensors"))
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="truncated"):
        store.read_range(0, 0, 5)
    store.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        _store(tmp_path)


def test_failure_between_asset_and_manifest_can_retry_without_advancing(tmp_path, monkeypatch):
    store = _store(tmp_path)
    original = store._write
    tensor = torch.ones((1, 24, 5, 2, 6))

    def fail(_):
        raise OSError("simulated write failure")

    monkeypatch.setattr(store, "_write", fail)
    with pytest.raises(OSError, match="simulated"):
        store.append(0, 0, tensor)
    assert store.position() == (0, 0, 5)
    monkeypatch.setattr(store, "_write", original)
    store.append(0, 0, tensor)
    assert torch.equal(store.read_range(0, 0, 5), tensor)
    assert len(list(tmp_path.glob("video-*.safetensors"))) == 1
