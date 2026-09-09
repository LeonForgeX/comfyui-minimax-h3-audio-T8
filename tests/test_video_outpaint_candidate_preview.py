from __future__ import annotations

import hashlib

import pytest
import torch

import h3_audio_t8_pkg.video_outpaint_candidate_preview as preview
from h3_audio_t8_pkg.video_outpaint_candidates import candidate_receipt
from h3_audio_t8_pkg.video_outpaint_compose import compose_sampled_outpaint
from test_video_outpaint_compose import _prepared


@pytest.mark.parametrize(
    "odd,color,geometry",
    [
        (False, True, False),
        (True, True, False),
        (False, False, False),
        (False, True, True),
        (True, False, True),
    ],
)
def test_candidate_first_image_equals_real_compositor_encoder_input(
    tmp_path, odd, color, geometry
):
    inputs = _prepared(tmp_path, odd=odd)
    candidate = candidate_receipt(inputs["window_store"])
    before = inputs["window_store"].path.read_bytes()
    inputs["vae"].decode_shapes.clear()
    image, report = preview.render_candidate_first_frame(
        **inputs, candidate=candidate, color_match=color, geometry_align=geometry
    )
    assert report["geometry_settings"]["geometry_align"] is geometry
    assert (
        len(inputs["vae"].decode_shapes) == 1 and inputs["vae"].decode_shapes[0][2] == 7
    )
    assert image.shape[0] == 1 and report["generated_prefix_used"]
    assert (
        not report["sampling_called_by_preview"] and not report["perceptual_acceptance"]
    )
    assert inputs["window_store"].path.read_bytes() == before
    raw_path = tmp_path / "encoded.rgb"
    compose_sampled_outpaint(
        **inputs,
        output_path=tmp_path / "result.mp4",
        color_match=color,
        capture_rgb_path=raw_path,
        geometry_align=geometry,
        expected_first_frame_sha256=report["rgb8_sha256"],
    )
    with raw_path.open("rb") as handle:
        first_rgb = handle.read(report["width"] * report["height"] * 3)
    assert hashlib.sha256(first_rgb).hexdigest() == report["rgb8_sha256"]
    assert (
        first_rgb
        == (image * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
    )


def test_wrong_decode_vae_is_rejected_before_model_call(tmp_path):
    inputs = _prepared(tmp_path)
    candidate = candidate_receipt(inputs["window_store"])
    inputs["vae"].first_stage_model.weight += 1
    inputs["vae"].decode_shapes.clear()
    with pytest.raises(ValueError, match="decode VAE"):
        preview.render_candidate_first_frame(**inputs, candidate=candidate)
    assert not inputs["vae"].decode_shapes


def test_decode_mutation_is_rejected_after_first_chunk(tmp_path, monkeypatch):
    inputs = _prepared(tmp_path)
    candidate = candidate_receipt(inputs["window_store"])
    original = inputs["vae"].first_stage_model._adaptive_decode

    def mutate(z):
        pixels = original(z)
        inputs["vae"].first_stage_model.weight += 1
        return pixels

    monkeypatch.setattr(inputs["vae"].first_stage_model, "_adaptive_decode", mutate)
    with pytest.raises(ValueError, match="VAE changed"):
        preview.render_candidate_first_frame(**inputs, candidate=candidate)


def test_cancelled_preview_does_not_advance_sampling_and_releases_lease(tmp_path):
    inputs = _prepared(tmp_path)
    store = inputs["window_store"]
    candidate = candidate_receipt(store)
    before = store.path.read_bytes()

    def cancel():
        raise RuntimeError("cancel preview")

    with pytest.raises(RuntimeError, match="cancel preview"):
        preview.render_candidate_first_frame(
            **inputs, candidate=candidate, interrupt_check=cancel
        )
    assert store.path.read_bytes() == before
    assert (
        preview.render_candidate_first_frame(**inputs, candidate=candidate)[0].shape[0]
        == 1
    )


def test_invalid_finish_settings_rejected(tmp_path):
    inputs = _prepared(tmp_path)
    candidate = candidate_receipt(inputs["window_store"])
    for kwargs in ({"color_match": "yes"}, {"color_settings": {"unknown": 1}}):
        with pytest.raises(ValueError, match="color"):
            preview.render_candidate_first_frame(
                **inputs, candidate=candidate, **kwargs
            )


@pytest.mark.parametrize("odd", [False, True])
def test_joint_decode_preview_matches_encoder_and_reports_reconstructed_source(
    tmp_path, odd
):
    inputs = _prepared(tmp_path, odd=odd)
    candidate = candidate_receipt(inputs["window_store"])
    image, shown = preview.render_candidate_first_frame(
        **inputs,
        candidate=candidate,
        source_mode="joint_decode",
        color_match=True,
        geometry_align=True,
    )
    assert shown["source_mode"] == "joint_decode" and shown["source_reconstructed"]
    assert shown["source_exact_before_encoding"] is False
    raw = tmp_path / "joint.rgb"
    _, report = compose_sampled_outpaint(
        **inputs,
        output_path=tmp_path / "joint.mp4",
        source_mode="joint_decode",
        color_match=True,
        geometry_align=True,
        capture_rgb_path=raw,
        expected_first_frame_sha256=shown["rgb8_sha256"],
    )
    with raw.open("rb") as stream:
        first = stream.read(shown["width"] * shown["height"] * 3)
    assert first == (image * 255).round().to(torch.uint8).numpy().tobytes()
    assert report["source_mode"] == "joint_decode" and report["source_reconstructed"]
    assert report["source_exact_before_encoding"] is False
    assert (
        report["pixel_receipt"]["schema"]
        == "t8.h3.video_outpaint.reconstructed_pixels/v1"
    )
    assert "pasted_rgb_sha256" not in report["pixel_receipt"]
    assert report["media"]["audio_decoded_pcm_exact"]
