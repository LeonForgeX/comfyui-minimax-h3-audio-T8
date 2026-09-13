from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.dance_motion import DanceMotionSource, prepare_dance_motion
from h3_audio_t8_pkg.nodes_dance_motion import MiniMaxH3DanceMotionSourceEXPT8
from h3_audio_t8_pkg.nodes_long_video_in_node_loop_effects_advanced import MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced
from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8


def _frames(count=192):
    return (torch.arange(count, dtype=torch.float32) / count)[:, None, None, None].expand(-1, 2, 2, 3).contiguous()


def _plan(start=124, new=68, context=22, render=124, final=True):
    return SimpleNamespace(timeline_start_seconds=start / 24, final_frame_count=new,
        context_frames=context, render_frames=render, is_final_segment=final)


def test_two_segment_eight_seconds_uses_source_motion_not_repeated_head():
    source = DanceMotionSource(_frames())
    first = _plan(start=0, new=124, context=0, final=False)
    indices0, audit0 = source.window_indices(first)
    indices1, audit1 = source.window_indices(_plan())
    assert indices0 == list(range(124))
    assert indices1 == list(range(102, 192)) + [191] * 34
    assert indices1[:22] == indices0[-22:]
    assert audit1["hidden_tail_hold_frames"] == 34
    assert audit0["delivered_end_frame_exclusive"] == 124
    assert audit1["delivered_end_frame_exclusive"] == 192


@pytest.mark.parametrize("numerator,denominator", [(24, 1), (30, 1), (60, 1), (30000, 1001), (24000, 1001)])
def test_rational_fps_has_no_accumulated_time_drift(numerator, denominator):
    from fractions import Fraction
    import math
    source = DanceMotionSource(_frames(600), numerator, denominator)
    indices, report = source.window_indices(_plan())
    assert indices == [math.floor(Fraction(i * numerator, 24 * denominator)) for i in range(102, 226)]
    assert report["hidden_tail_hold_frames"] == 0


def test_source_offset_and_existing_video_soundtrack_ordinals_not_replaced():
    source = DanceMotionSource(_frames(260), start_seconds=1.)
    existing = {"ref_video_0": _frames(56), "ref_video_2": _frames(56)}
    refs, report = source.references(_plan(), existing)
    assert refs["ref_video_0"] is existing["ref_video_0"]
    assert refs["ref_video_2"] is existing["ref_video_2"]
    assert set(existing) == {"ref_video_0", "ref_video_2"}
    assert report["input_ordinal"] == 3 and report["prompt_video_number"] == 3
    assert torch.equal(refs["ref_video_3"], source.frames[126:250])


def test_missing_delivered_choreography_does_not_silently_loop():
    with pytest.raises(ValueError, match="ends before delivered"):
        DanceMotionSource(_frames(170)).window_indices(_plan())
    with pytest.raises(ValueError, match="ends before delivered"):
        DanceMotionSource(_frames()).window_indices(_plan(final=False, new=102))


def test_all_windows_checked_before_sampling_and_orphan_audio_cannot_attach():
    segments = [SimpleNamespace(plan=_plan(start=0, new=124, context=0, final=False)), SimpleNamespace(plan=_plan())]
    with pytest.raises(ValueError, match="ends before delivered"):
        prepare_dance_motion(DanceMotionSource(_frames(170)), segments, "official_2_to_15s", None, None)
    with pytest.raises(ValueError, match="own existing video"):
        prepare_dance_motion(DanceMotionSource(_frames()), segments, "official_2_to_15s", None,
            {"ref_video_audio_0": {"waveform": torch.zeros(1, 2, 10), "sample_rate": 24}})


def test_content_identity_covers_interior_pixels_and_fps_offset():
    original = DanceMotionSource(_frames())
    changed = original.frames.clone()
    changed[73, 0, 0, 0] += .01
    assert original.contract() != DanceMotionSource(changed).contract()
    assert original.contract() != DanceMotionSource(original.frames, 30).contract()
    assert original.contract() != DanceMotionSource(original.frames, start_seconds=.1).contract()


@pytest.mark.parametrize("kwargs", [{"fps_denominator": 0}, {"fps_numerator": 0},
    {"start_seconds": -1.}, {"start_seconds": float("nan")}])
def test_invalid_time_metadata_rejected(kwargs):
    with pytest.raises(ValueError):
        DanceMotionSource(_frames(), **kwargs)


def test_motion_node_and_both_consumers_have_matching_optional_port():
    schema = MiniMaxH3DanceMotionSourceEXPT8.define_schema()
    assert schema.node_id == "MiniMaxH3DanceMotionSourceEXPT8"
    for cls in [MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced, MiniMaxH3DualModelLongVideoEXPT8]:
        port = next(item for item in cls.define_schema().inputs if item.id == "source_motion")
        assert port.optional


def test_reference_alignment_report_does_not_claim_all_frames_encoded():
    source = DanceMotionSource(_frames())
    indices, report = source.window_indices(_plan(start=0, new=96, context=0, render=96))
    assert len(indices) == 96 and report["reference_aligned_frames"] == 90


def test_real_conditioning_consumes_source_rgb_separate_from_clean_context_and_music():
    from helpers import FakeVideoVAE, FakeAudioVAE, FakeClip, make_audio
    from h3_audio_t8_pkg.long_video import build_long_video_conditioning
    import json
    source = DanceMotionSource(_frames(226))
    refs, _ = source.references(_plan(new=102))
    context = {"schema": 1, "empty": False,
        "video_tail": torch.ones(1, 24, 12, 8, 8),
        "audio_tail": torch.ones(1, 32, 2, 65),
        "metadata": {"source_segment_index": 0, "target_segment_index": 1,
                     "max_context_frames": 39, "audio_overhang": 1 / 3}}
    before = context["video_tail"].clone()
    music = make_audio(124 / 24)
    video_vae = FakeVideoVAE()
    result = build_long_video_conditioning(
        clip=FakeClip(), video_vae=video_vae, audio_vae=FakeAudioVAE(), context=context,
        segment_index=1, context_frames=22, context_audio="video_and_audio",
        prompt="<Picture 1> follows the dance in <Video 1>.", width=128, height=128, length=124,
        ref_images={"ref_image_0": torch.full((1, 128, 128, 3), .8)},
        ref_videos=refs, final_audio=music)
    positive, _latent, mux, _prompt, _media, report = result
    encoded_motion = next(frames for frames in video_vae.encode_calls if len(frames) == 124)
    # Existing Core Lanczos uses uint8 PIL images; verify temporal values after
    # that known spatial conversion, not fictitious float-perfect RGB passage.
    from h3_audio_t8_pkg.core import resize_image
    expected = resize_image(source.frames[102:226], encoded_motion.shape[2], encoded_motion.shape[1])
    assert torch.equal(encoded_motion, expected)
    assert encoded_motion[0, 0, 0, 0] > .4  # Not a repeated source frame zero.
    assert torch.equal(context["video_tail"], before)
    assert mux is music
    kinds = [ref["kind"] for ref in positive[0][1]["minimax_refs"]]
    assert "video" in kinds and "image" in kinds
    assert json.loads(report)["motion_keyframes"] == 7


@pytest.mark.parametrize("rate", [24000, 32000, 44100, 48000])
def test_source_audio_offset_preserves_exact_pcm_and_segment_timeline(rate):
    from h3_audio_t8_pkg.dance_motion import dance_audio_at_source_start
    from h3_audio_t8_pkg.long_video_in_node_loop_advanced import _window_segment_audio
    start = .375
    audio = {"waveform": torch.linspace(-1, 1, 10 * rate).reshape(1, 1, -1), "sample_rate": rate}
    original = audio["waveform"].clone()
    aligned, report = dance_audio_at_source_start(audio, start)
    assert report["start_sample"] == round(start * rate)
    assert torch.equal(aligned["waveform"], original[..., round(start * rate):])
    segment = _window_segment_audio(aligned, _plan(), name="final_audio")
    # Global second render starts at frame102, including22 overlap frames.
    index = round(start * rate) + round(102 / 24 * rate)
    assert torch.equal(segment["waveform"], original[..., index:index + round(124 / 24 * rate)])
    assert torch.equal(original, audio["waveform"])
    assert dance_audio_at_source_start(audio, 0.)[0] is audio


def test_source_audio_exhausted_offset_and_invalid_rate_fail_without_silent_music():
    from h3_audio_t8_pkg.dance_motion import dance_audio_at_source_start
    for rate, offset in [(0, 0.), (24, 1.)]:
        with pytest.raises(ValueError):
            dance_audio_at_source_start({"waveform": torch.zeros(1, 1, 24), "sample_rate": rate}, offset)
    assert dance_audio_at_source_start(None, 0.)[0] is None


def test_connected_fps_overrides_manual_fields_and_music_is_output():
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    output = MiniMaxH3DanceMotionSourceEXPT8.execute(_frames(), source_fps=30000 / 1001,
        source_audio=audio, source_start_seconds=.125)
    assert output[0].fps_numerator == 30000 and output[0].fps_denominator == 1001
    assert output[2]["waveform"].shape[-1] == 42000


def test_delivered_template_keeps_motion_music_character_and_independent_models():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "examples/workflows/04-long-video"
    rejected = [
        "2026-09-12_H3_Dance_Native_Stock20_EXP.json",
        "2026-09-12_H3_Dance_Dual_4plus4_EXP.json",
    ]
    assert all(not (root / name).exists() for name in rejected)
    workflow = json.loads(
        (root / "2026-09-13_H3_Dance_4plus4_Accepted_Picture_KJ.json").read_text(encoding="utf8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    runner = next(node for node in nodes.values() if node["type"] == "MiniMaxH3DualModelLongVideoEXPT8")
    edges = {edge[0]: edge for edge in workflow["links"]}
    inputs = {pin["name"]: pin for pin in runner["inputs"]}
    motion_edge = edges[inputs["source_motion"]["link"]]
    audio_edge = edges[inputs["final_audio"]["link"]]
    assert motion_edge[1] == audio_edge[1] and motion_edge[2] == 0 and audio_edge[2] == 2
    assert nodes[motion_edge[1]]["type"] == "MiniMaxH3DanceMotionSourceEXPT8"
    character = edges[inputs["ref_images.ref_image_0"]["link"]]
    assert nodes[character[1]]["type"] == "LoadImage"
    first = edges[inputs["model_pass1"]["link"]][1]
    second = edges[inputs["model_pass2"]["link"]][1]
    assert first != second
    assert workflow["extra"]["accepted_video_sha256"] == (
        "58a8a8fb616a741be4c76f75e5857dff098e0b649959470ac0ba9b8ff7877039"
    )
