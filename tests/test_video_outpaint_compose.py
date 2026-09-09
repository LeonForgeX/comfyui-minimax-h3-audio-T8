from __future__ import annotations

import hashlib
import json

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_media import inspect_outpaint_source
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_source_runtime import prepare_outpaint_source_cache
from h3_audio_t8_pkg.video_outpaint_window_store import OutpaintWindowStore
from h3_audio_t8_pkg.video_outpaint_sampling_runtime import sample_prepared_outpaint_windows
from h3_audio_t8_pkg.video_outpaint_compose import (
    OUTPAINT_VIDEO_ENCODER_POLICY,
    OUTPAINT_X264_PARAMS,
    _outpaint_encoder_output_args,
    compose_sampled_outpaint,
)
from h3_audio_t8_pkg.video_outpaint_delivery import read_delivery_report
from test_video_outpaint_media import _clip
from test_video_outpaint_source_runtime import StatefulOracle, StatefulVAE
from test_video_outpaint_decode import DecoderOracle
from test_video_outpaint_sampling import _backend


class CombinedOracle(StatefulOracle, DecoderOracle):
    pass


class CombinedVAE(StatefulVAE):
    device = torch.device("cpu")
    vae_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.first_stage_model = CombinedOracle()
        self.decode_shapes = []

    def throw_exception_if_invalid(self):
        pass

    def prepare_decode(self, shape):
        self.decode_shapes.append(shape)


def _prepared(tmp_path, *, odd=False, frames=39, cut_frames=None):
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=frames, sound=True)
    inspection = inspect_outpaint_source(source)
    plan = build_outpaint_plan(
        source_sha256=inspection["sha256"], width=32, height=32, frame_count=frames,
        aspect="custom", left=17 if odd else 32, right=16 if odd else 32,
        cut_frames=[] if cut_frames is None else cut_frames, window_frames=39,
    )
    vae = CombinedVAE()
    source_store, _ = prepare_outpaint_source_cache(vae, inspection, plan, tmp_path / "source-cache")
    identity = {"model_sha256": "a"*64, "conditioning_sha256": "b"*64, "audio_source_sha256": "c"*64,
                "source_cache_sha256": hashlib.sha256(source_store.path.read_bytes()).hexdigest(),
                "seed": 42, "steps": 20, "sampler_name": "res_multistep", "scheduler": "simple",
                "noise_algorithm": "t8.outpaint.coordinate_noise/v1"}
    window_store = OutpaintWindowStore(tmp_path / "windows", plan, execution_identity=identity)

    def audio(shot, window):
        frames = plan["shots"][shot]["windows"][window]["render_frames"]
        return torch.zeros((1, 32, 2, round(frames/24*40)))

    # Zero fake conditioning audio is only a CPU fixture. Final file must still preserve real AAC.
    sample_prepared_outpaint_windows(model=object(), conditioning_for_window=lambda s, w: [[torch.zeros(1), {}]],
                                     audio_for_window=audio, source_store=source_store, window_store=window_store,
                                     verify_execution=lambda: window_store.identity, sample_function=_backend)
    return dict(vae=vae, inspection=inspection, source_store=source_store, window_store=window_store)


@pytest.mark.parametrize(
    "pixel_format,profile", [("yuv420p", "baseline"), ("yuv444p", "high444")]
)
def test_outpaint_encoder_uses_decoder_safe_all_intra_contract(pixel_format, profile):
    args, selected_profile = _outpaint_encoder_output_args(
        pixel_format=pixel_format, crf=18, frame_count=39
    )
    assert selected_profile == profile
    assert args[args.index("-profile:v") + 1] == profile
    assert args[args.index("-frames:v") + 1] == "39"
    assert args[args.index("-threads") + 1] == "1"
    assert args[args.index("-x264-params") + 1] == OUTPAINT_X264_PARAMS
    for required in ("threads=1", "ref=1", "bframes=0", "keyint=1", "cabac=0"):
        assert required in OUTPAINT_X264_PARAMS


def test_outpaint_encoder_rejects_unsupported_pixel_format():
    with pytest.raises(ValueError, match="pixel format"):
        _outpaint_encoder_output_args(pixel_format="yuv422p", crf=18, frame_count=39)


@pytest.mark.parametrize("odd", [False, True])
def test_real_file_compose_binds_exact_source_rgb_receipt_and_original_audio(tmp_path, odd):
    inputs = _prepared(tmp_path, odd=odd)
    raw_path = tmp_path / "encoder.rgb"
    video, report = compose_sampled_outpaint(**inputs, output_path=tmp_path / "result.mp4", capture_rgb_path=raw_path)
    raw = raw_path.read_bytes()
    metadata = json.loads(raw_path.with_suffix(".rgb.json").read_text())
    width = 65 if odd else 96
    assert len(raw) == 39 * width * 32 * 3
    assert metadata["complete_rgb_stream"] and metadata["error"] is None
    assert metadata["rgb_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["provenance"]["plan_sha256"] == inputs["window_store"].plan["plan_sha256"]
    frames = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(39, 32, width, 3)
    x1, y1, x2, y2 = inputs["window_store"].plan["output"]["source_rect"]
    assert hashlib.sha256(frames[:, y1:y2, x1:x2].contiguous().numpy().tobytes()).hexdigest() == report["pixel_receipt"]["source_rgb_sha256"]
    assert video.get_frame_count() == 39
    assert video.get_dimensions() == ((65 if odd else 96), 32)
    assert report["pre_encode_pixel_evidence_verified"]
    assert report["pixel_receipt"]["source_rgb_sha256"] == report["pixel_receipt"]["pasted_rgb_sha256"]
    assert report["media"]["audio_packet_payload_exact"]
    assert report["media"]["audio_packet_timeline_exact"]
    assert report["media"]["audio_decoded_pcm_exact"]
    assert report["media"]["audio_decoded_timeline_exact"]
    assert not report["pixel_receipt"]["lossy_encoded_equality_claimed"]
    assert report["odd_size_player_review_required"] == odd
    assert report["encoder_policy"] == OUTPAINT_VIDEO_ENCODER_POLICY
    assert report["encoder_x264_params"] == OUTPAINT_X264_PARAMS
    assert report["encoder_profile"] == ("high444" if odd else "baseline")
    assert report["media"]["bounded_multithread_decode_passed"]
    assert report["media"]["bounded_multithread_decode_threads"] == 4
    assert report["media"]["bounded_multithread_decode_attempts"] == 3
    persisted = read_delivery_report(tmp_path / "result.mp4")
    assert persisted == report
    assert persisted["execution_identity"] == inputs["window_store"].identity
    assert persisted["plan"] == inputs["source_store"].plan
    assert persisted["composition_implementation_sha256"]
    assert persisted["color_match_report"]["shot_resets"] == 1
    assert persisted["color_match_report"]["source_exact_before_encoding"]
    assert all(shape[2] == 7 for shape in inputs["vae"].decode_shapes)
    assert not list(tmp_path.glob(".*.video-only-*.mp4"))


def test_composition_records_one_color_state_reset_per_shot(tmp_path):
    inputs = _prepared(tmp_path, frames=78, cut_frames=[39])
    _, report = compose_sampled_outpaint(
        **inputs, output_path=tmp_path / "two-shots.mp4", color_match=True
    )
    assert report["color_match_report"]["color_match_applied"]
    assert report["color_match_report"]["shot_resets"] == 2
    assert report["color_match_report"]["chunks"] >= 2


@pytest.mark.parametrize("odd", [False, True])
def test_composition_cancellation_keeps_checkpoints_and_publishes_nothing(tmp_path, odd):
    inputs = _prepared(tmp_path, odd=odd)
    before = inputs["window_store"].path.read_bytes()

    def cancel(_):
        raise RuntimeError("cancelled compose")

    with pytest.raises(RuntimeError, match="cancelled compose"):
        compose_sampled_outpaint(**inputs, output_path=tmp_path / "result.mp4", progress=cancel)
    assert not (tmp_path / "result.mp4").exists()
    assert not list(tmp_path.glob(".*.video-only-*.mp4"))
    assert inputs["window_store"].path.read_bytes() == before
    _, report = compose_sampled_outpaint(**inputs, output_path=tmp_path / "result.mp4", color_match=False)
    assert not report["color_match_enabled"]


def test_changed_decode_vae_cannot_publish_from_another_cache(tmp_path):
    inputs = _prepared(tmp_path)
    inputs["vae"].first_stage_model.weight += 1
    with pytest.raises(ValueError, match="decode VAE"):
        compose_sampled_outpaint(**inputs, output_path=tmp_path / "result.mp4")
    assert not (tmp_path / "result.mp4").exists()
