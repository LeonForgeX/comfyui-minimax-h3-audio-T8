"""Captured sampler tensors are lossless and do not change the decode graph inputs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from tools.trt_latent_capture import capture_parts, capture_recipe, tensor_identity


def test_full_av_parts_saved_exactly_and_originals_not_mutated(tmp_path):
    root = tmp_path / "run" / "output"
    root.mkdir(parents=True)
    video, audio = torch.rand(1, 24, 7, 16, 16), torch.rand(1, 32, 2, 37)
    before = video.clone(), audio.clone()
    report = capture_parts(video, audio, root, allowed_root=tmp_path)
    assert torch.equal(video, before[0]) and torch.equal(audio, before[1])
    assert report["status"] == "actual_sampler_output_captured_bit_exact"
    for kind, key, value in (("video", "latent_tensor", video), ("audio", "audio_latent", audio)):
        item = report["files"][kind]
        assert torch.equal(load_file(item["path"])[key], value)
        assert item["tensor"] == tensor_identity(value)
    assert "latent_format_version_0" in load_file(report["files"]["video"]["path"])
    with pytest.raises(FileExistsError):
        capture_parts(video, audio, root, allowed_root=tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_sample_never_creates_evidence(tmp_path, bad):
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(ValueError, match="finite"):
        capture_parts(torch.full((1, 24, 2, 2, 2), bad), torch.zeros(1, 32, 2, 4), root, allowed_root=tmp_path)
    assert not (root / "trt-latent-evidence").exists()


def test_cannot_write_in_user_output_directory(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(ValueError, match="research"):
        capture_parts(torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 4), root)


@pytest.mark.parametrize("name", ["T2VA_native8", "T2VA_progressive6plus2", "I2VA_native8", "I2VA_progressive6plus2"])
def test_capture_preserves_every_original_node_except_decode_link(name):
    from tools.run_progressive_pilot import instrument_recipe
    path = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909/pilot-api-drafts" / (name + ".prompt.json")
    original = instrument_recipe(json.loads(path.read_text(encoding="utf8")))
    before = deepcopy(original)
    captured = capture_recipe(original)
    assert original == before
    assert captured["105"]["inputs"]["av_latent"] == before["11"]["inputs"]["av_latent"]
    assert captured["11"]["inputs"]["av_latent"] == ["105", 0]
    restored = deepcopy(captured)
    restored["11"]["inputs"]["av_latent"] = restored.pop("105")["inputs"]["av_latent"]
    restored.pop("106")
    assert restored == original
