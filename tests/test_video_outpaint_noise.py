import pytest
import torch
from comfy.sample import prepare_noise_inner

from h3_audio_t8_pkg import video_outpaint_noise as noise
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan


@pytest.mark.parametrize("frames,cuts", [(1, ()), (90, ()), (768, ()), (129, (45, 88))])
@pytest.mark.parametrize("seed", [0, 20260808, 2**64-1])
def test_native_stream_exact_against_whole_comfy_noise(frames, cuts, seed):
    plan = build_outpaint_plan(source_sha256="a"*64, width=32, height=32, frame_count=frames,
        aspect="custom", left=32, right=32, top=32, bottom=32, window_frames=73, cut_frames=cuts)
    before = torch.random.get_rng_state().clone()
    for shot_index, shot in enumerate(plan["shots"]):
        vt = (shot["aligned_frames"]-5)//17*5+2
        at = round(shot["aligned_frames"]/24*40)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Exact two calls/order used by Comfy prepare_noise(NestedTensor).
        reference_video = prepare_noise_inner(torch.empty(1, 24, vt, 6, 6), gen)
        reference_audio = prepare_noise_inner(torch.empty(1, 32, 2, at), gen)
        for index, window in enumerate(shot["windows"]):
            video, audio = noise.native_outpaint_window_noise(plan, shot_index, index, seed)
            start_v, start_a = window["video_start"], window["audio_start"]
            assert torch.equal(video, reference_video[:, :, start_v:start_v+video.shape[2]])
            assert torch.equal(audio, reference_audio[..., start_a:start_a+audio.shape[-1]])
    assert torch.equal(before, torch.random.get_rng_state())


def test_actual_probe_geometry_crosses_blocks_with_exact_native_values(monkeypatch):
    plan = build_outpaint_plan(source_sha256="a"*64, width=512, height=288, frame_count=90,
        aspect="custom", top=192, bottom=192, window_frames=90, generation_megapixels=0.5)
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    reference = (prepare_noise_inner(torch.empty(1, 24, 27, 42, 32), generator),
                 prepare_noise_inner(torch.empty(1, 32, 2, 150), generator))
    sizes = []
    randn = torch.randn
    def observed(count, **kwargs):
        sizes.append(count)
        return randn(count, **kwargs)
    monkeypatch.setattr(noise.torch, "randn", observed)
    actual = noise.native_outpaint_window_noise(plan, 0, 0, 20260808)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    assert max(sizes) <= noise._BLOCK_FLOATS and len(sizes) > 2


def test_noise_replay_is_cancellable_without_mutating_rng():
    plan = build_outpaint_plan(source_sha256="a"*64, width=512, height=288, frame_count=768,
        aspect="custom", top=192, bottom=192, window_frames=90)
    before = torch.random.get_rng_state().clone()
    def cancel():
        raise RuntimeError("cancel noise replay")
    with pytest.raises(RuntimeError, match="cancel noise"):
        noise.native_outpaint_window_noise(plan, 0, 0, 1, interrupt_check=cancel)
    assert torch.equal(before, torch.random.get_rng_state())


def test_long_actual_canvas_nonzero_window_offsets_match_whole_noise():
    plan = build_outpaint_plan(source_sha256="a"*64, width=512, height=288, frame_count=768,
        aspect="custom", top=192, bottom=192, window_frames=90, generation_megapixels=0.5)
    shot = plan["shots"][0]
    vt = (shot["aligned_frames"]-5)//17*5+2
    at = round(shot["aligned_frames"]/24*40)
    gen = torch.Generator(device="cpu").manual_seed(20260808)
    reference_video = prepare_noise_inner(torch.empty(1, 24, vt, 42, 32), gen)
    reference_audio = prepare_noise_inner(torch.empty(1, 32, 2, at), gen)
    for index in (0, 1, len(shot["windows"])-1):
        window = shot["windows"][index]
        video, audio = noise.native_outpaint_window_noise(plan, 0, index, 20260808)
        assert torch.equal(video, reference_video[:, :, window["video_start"]:window["video_start"]+video.shape[2]])
        assert torch.equal(audio, reference_audio[..., window["audio_start"]:window["audio_start"]+audio.shape[-1]])
