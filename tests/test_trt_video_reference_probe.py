"""Short-video replay must reach production reference conditioning, not a keyframe."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from tools.trt_encoder_condition_probe import SavedReferenceEncoder, condition_recipe
from tools.run_progressive_pilot import RESEARCH, instrument_recipe
from tools.audit_trt_encoder_condition_pair import verify_reference_conditioning


def test_video_recipe_has_one_full_reference_and_native_decode():
    graph = instrument_recipe(json.loads((RESEARCH / "pilot-api-drafts/I2VA_native8.prompt.json").read_text()))
    before = deepcopy(graph)
    a, b = (condition_recipe(graph, "evidence", backend, "a" * 64, "video") for backend in ("native", "trt"))
    assert graph == before
    assert "first_frame" not in a["6"]["inputs"]
    assert a["6"]["inputs"]["ref_videos.ref_video_0"] == ["107", 1]
    assert a["107"]["inputs"]["reference_kind"] == "video"
    assert a["11"] == graph["11"] and a["10"] == graph["10"]
    b["107"]["inputs"]["backend"] = "native"
    b["18"] = a["18"]
    assert a == b


def test_complete_video_passes_actual_conditioning_without_resize_or_truncation():
    from h3_audio_t8_pkg.conditioning import build_conditioning
    from helpers import FakeAudioVAE, FakeClip
    # Saved real evidence is RGB8/255. Core's Lanczos roundtrip quantizes
    # arbitrary float gradients, even when the dimensions do not change.
    pixels = torch.arange(73, dtype=torch.float32).div(255).view(73, 1, 1, 1).expand(73, 512, 1024, 3)
    latent = torch.zeros(1, 24, 22, 32, 64)
    delegate = SavedReferenceEncoder(object(), pixels, latent, {}, "trt", "video")
    clip = FakeClip()
    conditioning, *_ = build_conditioning(
        clip=clip, video_vae=delegate, audio_vae=FakeAudioVAE(),
        prompt="Use <Video 1> as reference.", width=1024, height=512, length=73,
        task_type="auto", audio_mode="native", ref_videos={"ref_video_0": pixels})
    metadata = conditioning[0][1]
    assert not metadata.get("minimax_keyframes")
    ref = metadata["minimax_refs"][0]
    assert ref["kind"] == "video" and ref["latent_t"] == 22
    assert torch.equal(ref["latent"], latent)
    assert delegate.report()["reference_frames"] == 73 and delegate.calls == 1
    items = clip.tokenize_calls[0][1]["minimax_ref_items"]
    assert items[0]["type"] == "video" and len(items[0]["timestamps"]) == 7
    assert items[0]["timestamps"][-1] == 3.0
    with pytest.raises(ValueError, match="exactly one"):
        delegate.encode(pixels)


@pytest.mark.parametrize("mutation", ["latent_t", "kind", "keyframe", "latent"])
def test_video_auditor_rejects_wrong_reference(mutation):
    identity = {"sha256": "abc", "tensor_shape": [1, 24, 22, 32, 64], "dtype": "torch.float32"}
    block = {"minimax_refs": [{"kind": "video", "latent_t": 22, "latent_h": 32, "latent_w": 64,
                               "ref_audio_t": 0, "audio_latent": None, "latent": identity}]}
    def report(value):
        c = [[{}, value]]
        return {"conditioning": c, "conditioning_sha256": hashlib.sha256(
            json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    assert verify_reference_conditioning(report(block), identity, "video")
    if mutation == "keyframe":
        block["minimax_keyframes"] = [{"resolved_frame_index": 0}]
    else:
        block["minimax_refs"][0][mutation] = "invalid"
    with pytest.raises(ValueError, match="complete video"):
        verify_reference_conditioning(report(block), identity, "video")
