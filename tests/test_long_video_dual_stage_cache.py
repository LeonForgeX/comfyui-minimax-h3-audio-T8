import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg.long_video_dual_stage_cache import AVStageCache
from test_long_video_dual_model_stages import latent


def test_roundtrip_exact_samples_masks_and_metadata(tmp_path):
    cache = AVStageCache(tmp_path)
    source = latent(2.)
    source["noise_mask"] = NestedTensor([torch.ones_like(x) for x in source["samples"].unbind()])
    source["batch_index"] = [0]
    contract = {"segment": 1, "first_model_content": "test fixture", "seed": 17}
    assert cache.load("low_x0", contract) is None
    cache.save("low_x0", contract, source, {"stage": "passed"})
    output, receipt = cache.load("low_x0", contract)
    for key in ("samples", "noise_mask"):
        for expected, actual in zip(source[key].unbind(), output[key].unbind()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert output["batch_index"] == [0]
    assert receipt["report"] == {"stage": "passed"}
    with pytest.raises(FileExistsError):
        cache.save("low_x0", contract, latent(3.), {})


@pytest.mark.parametrize("changed", ["model", "lora", "seed", "context", "backend", "high_resolution"])
def test_changed_execution_contract_does_not_reuse_prior_stage(tmp_path, changed):
    cache = AVStageCache(tmp_path)
    contract = {key: "a" for key in ("model", "lora", "seed", "context", "backend", "high_resolution")}
    cache.save("low_x0", contract, latent(2.), {})
    assert cache.load("low_x0", {**contract, changed: "b"}) is None
    assert cache.load("high_output", contract) is None
    assert cache.load("low_x0", contract) is not None


def test_orphan_tensor_is_not_a_completed_stage(tmp_path):
    cache = AVStageCache(tmp_path)
    cache.save("low_x0", {}, latent(2.), {})
    next(tmp_path.glob("*.json")).unlink()
    assert list(tmp_path.glob("*.safetensors"))
    assert cache.load("low_x0", {}) is None


@pytest.mark.parametrize("tamper", ["bytes", "path", "contract"])
def test_corruption_is_not_silently_resumed(tmp_path, tamper):
    cache = AVStageCache(tmp_path)
    cache.save("high_output", {}, latent(2.), {})
    path = next(tmp_path.glob("*.json"))
    receipt = json.loads(path.read_text())
    if tamper == "bytes":
        tensor = next(tmp_path.glob("*.safetensors"))
        tensor.write_bytes(tensor.read_bytes() + b"corrupt")
    elif tamper == "path":
        receipt["tensor_file"] = "../foreign.safetensors"
    else:
        receipt["contract"] = {"seed": 100}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        cache.load("high_output", {})
