"""Boundary tests use a double; actual learned-weight execution is separate."""
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import h3_ltx_latent_contract as contract


def source(frames=73, prefix=0):
    count = ((frames - 5) // 17) * 5 + 2
    video = torch.ones(1, 24, count + prefix, 4, 6)
    audio = torch.ones(1, 32, 2, 122)
    return {"samples": NestedTensor((video, audio)), "noise_mask": "H3-only", "metadata": {"ref": 1}}


class Adapter:
    def __init__(self):
        self.calls = []
    def convert(self, value, **kwargs):
        self.calls.append((value.clone(), kwargs))
        value.zero_()  # Even an in-place backend must not alter source.
        count = ((kwargs["pixel_frames"] - 1 + 7) // 8) + 1
        return torch.ones(value.shape[0], 128, count, value.shape[-2] // 2, value.shape[-1] // 2)


def test_joint_av_only_converts_video_and_preserves_source_and_ratio():
    original, adapter = source(), Adapter()
    result, preserved, report = contract.convert_video_latent(original, adapter=adapter, source_frames=73)
    assert preserved is original and set(result) == {"samples"}
    assert torch.all(original["samples"].unbind()[0] == 1)
    assert result["samples"].shape == (1, 128, 10, 2, 3)
    assert (report["width"], report["height"]) == (96, 64)
    assert report["output_frames"] == 73 and not report["audio_duration_adjustment_required"]
    assert adapter.calls[0][1]["input_normalization"] == "normalized"


@pytest.mark.parametrize("policy,frames,t", [("pad_to_ltx_grid", 129, 17), ("crop_to_ltx_grid", 121, 16)])
def test_124_never_reports_false_124_for_121_frame_latent(policy, frames, t):
    result, _, report = contract.convert_video_latent(source(124), adapter=Adapter(), source_frames=124, frame_policy=policy)
    assert report["source_frames"] == 124 and report["output_frames"] == frames
    assert result["samples"].shape[2] == t and report["audio_duration_adjustment_required"]


def test_exact_mode_never_silently_crops():
    with pytest.raises(ValueError, match="explicitly choose"):
        contract.convert_video_latent(source(124), adapter=Adapter(), source_frames=124)


def test_explicit_prefix_and_raw_normalization():
    value, adapter = source(prefix=2), Adapter()
    with pytest.raises(ValueError, match="exact reference prefix"):
        contract.convert_video_latent(value, adapter=adapter, source_frames=73)
    contract.convert_video_latent(value, adapter=adapter, source_frames=73,
                                 reference_prefix_latents=2, normalization="raw_h3")
    assert adapter.calls[0][0].shape[2] == 22
    assert adapter.calls[0][1]["input_normalization"] == "raw"


@pytest.mark.parametrize("frames,fps,policy", [(True,24,"exact"),(72,24,"exact"),(73,30,"exact"),
    (73,float("nan"),"exact"),(73,True,"exact"),(73,24,"unknown")])
def test_invalid_time_contracts(frames, fps, policy):
    with pytest.raises(ValueError):
        contract.timeline(frames, fps, policy)


@pytest.mark.parametrize("case", ["nonfinite", "channels", "spatial", "audio_first", "audio_batch"])
def test_invalid_input_rejected_before_adapter(case):
    value, adapter = source(), Adapter()
    video, audio = value["samples"].unbind()
    if case == "nonfinite":
        video[0,0,0,0,0] = float("nan")
    elif case == "channels":
        video = video[:, :23]
    elif case == "spatial":
        video = video[..., :3, :]
    elif case == "audio_first":
        video, audio = audio, video
    else:
        audio = audio.expand(2, -1, -1, -1)
    value["samples"] = NestedTensor((video, audio))
    with pytest.raises(ValueError):
        contract.convert_video_latent(value, adapter=adapter, source_frames=73)
    assert not adapter.calls


def test_bad_output_and_cancel_do_not_return_partial_latent():
    adapter = SimpleNamespace(convert=lambda *a, **k: torch.zeros(1, 128, 9, 2, 3))
    with pytest.raises(ValueError, match="output violates"):
        contract.convert_video_latent(source(), adapter=adapter, source_frames=73)
    steps = []
    def cancel():
        steps.append(1)
        if len(steps) == 3:
            raise RuntimeError("cancelled after adapter")
    with pytest.raises(RuntimeError, match="cancelled"):
        contract.convert_video_latent(source(), adapter=Adapter(), source_frames=73, check_cancel=cancel)


def test_no_arbitrary_frame_ceiling():
    geometry = contract.timeline(1705, 24, "pad_to_ltx_grid")
    assert geometry["source_frames"] == 1705
