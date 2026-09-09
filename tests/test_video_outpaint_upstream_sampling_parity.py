"""Pinned upstream sampling-boundary oracle; fake VAE/DiT, actual native noise.

These tests prove input/mask/window equivalence, not learned generation quality.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
import comfy.model_management
import comfy.nested_tensor
import comfy.sample

from h3_audio_t8_pkg.video_outpaint_noise import native_outpaint_window_noise
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_sampling import sample_outpaint_window


@pytest.fixture
def upstream(monkeypatch):
    path = Path(__file__).parents[1] / "artifacts/upstream-h3-video-outpaint-27df0ff/nodes.py"
    if not path.exists():
        pytest.skip("optional pinned upstream fixture is not installed")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "9b07f825fa96749abb640b5effeed76a904d3099ae825cd32b98c5f8ee0ff070"
    # ``comfy_extras.nodes_minimax_h3`` only reads this constant while the
    # pinned oracle imports.  Prevent the plugin root's own ``nodes.py`` from
    # being mistaken for ComfyUI's top-level module when pytest runs here.
    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.MAX_RESOLUTION = 16384
    monkeypatch.setitem(sys.modules, "nodes", comfy_nodes)
    spec = importlib.util.spec_from_file_location("outpaint_sampling_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("frames,window_frames,side", [(90,90,32), (90,73,32), (124,73,32),
                                                     (80,73,32), (1,73,32), (56,56,96), (90,73,96)])
def test_entire_sampling_boundary_matches_pinned_reference(upstream, monkeypatch, frames, window_frames, side):
    plan = build_outpaint_plan(source_sha256="a"*64, width=side, height=side, frame_count=frames,
        aspect="custom", top=32, bottom=32, left=32, right=32, window_frames=window_frames)
    shot = plan["shots"][0]
    class VAE:
        def encode(self, pixels):
            return torch.full((1, 24, (len(pixels)-5)//17*5+2, side//16, side//16), 0.5)
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    spatial = upstream._spatial_generation_mask((side+64)//16, (side+64)//16, side, side, 32, 32, "cpu")
    latent, shapes, specs = upstream._assemble_global_latent(
        torch.zeros(shot["aligned_frames"], side, side, 3, dtype=torch.uint8), frames, VAE(),
        32, 32, 32, 32, spatial, None, min(window_frames, shot["aligned_frames"]))
    source_video, _ = latent["samples"].unbind()
    source_video = source_video.clone()
    captures = []
    def backend(*args, **kwargs):
        noise = [t.clone() for t in args[1].unbind()]
        samples = [t.clone() for t in args[8].unbind()]
        masks = [t.clone() for t in kwargs["noise_mask"].unbind()]
        captures.append((noise, samples, masks, args[6]))
        # Deterministic fake denoiser responds to inputs but preserves locked cells.
        return comfy.nested_tensor.NestedTensor([
            torch.where(mask.bool(), sample*0.75+eps*0.25, sample)
            for sample, eps, mask in zip(samples, noise, masks)])
    monkeypatch.setattr(comfy.sample, "sample", backend)
    conditioning = [[torch.zeros(1, 1, 8), {"minimax_token_tags": torch.ones(1, 1, dtype=torch.long)}]]
    with torch.random.fork_rng(devices=[]):
        expected = upstream._sample_sliding_latent(object(), conditioning, conditioning, latent, shapes, specs,
                                                  20260808, 20, "res_multistep", "simple")
    expected_captures = captures[:]
    captures.clear()
    context = None
    generated_video = source_video.clone()
    for index, window in enumerate(shot["windows"]):
        vt = (window["render_frames"]-5)//17*5+2
        at = round(window["render_frames"]/24*40)
        video_noise, audio_noise = native_outpaint_window_noise(plan, 0, index, 20260808)
        result, context, _ = sample_outpaint_window(
            model=object(), conditioning=conditioning, plan=plan, shot_index=0, window_index=index,
            video_latent=source_video[:, :, window["video_start"]:window["video_start"]+vt],
            audio_latent=torch.zeros(1, 32, 2, at), audio_noise_mask=torch.ones(1, 1, 2, at),
            video_noise=video_noise, audio_noise=audio_noise, context=context, sample_function=backend,
            source_edge_mode="strict")
        begin = window["video_start"]+window["context_video_latents"]
        generated_video[:, :, begin:window["video_start"]+vt] = result["samples"].unbind()[0][:, :, window["context_video_latents"]:]
    assert len(captures) == len(expected_captures)
    for actual, reference in zip(captures, expected_captures):
        for label, ours, theirs in zip(("noise", "latents", "masks"), actual[:3], reference[:3]):
            assert all(torch.equal(a, b) for a, b in zip(ours, theirs)), label
        ours, theirs = actual[3][0][1], reference[3][0][1]
        assert torch.equal(ours["minimax_token_tags"], theirs["minimax_token_tags"])
        # Absence and [] currently have the same native PackedLayout meaning.
        our_keys, their_keys = ours.get("minimax_keyframes", []), theirs.get("minimax_keyframes", [])
        assert len(our_keys) == len(their_keys)
        for a, b in zip(our_keys, their_keys):
            assert a["resolved_frame_index"] == b["resolved_frame_index"]
            assert torch.equal(a["latent"], b["latent"])
            assert torch.equal(a["audio_latent"], b["audio_latent"])
    assert torch.equal(generated_video, expected["samples"].unbind()[0])
