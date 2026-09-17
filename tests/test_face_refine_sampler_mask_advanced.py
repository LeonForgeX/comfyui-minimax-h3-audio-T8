from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor
import comfy.utils

from h3_audio_t8_pkg.face_refine_sampler_mask_advanced import (
    PATCH_SCHEMA,
    UPSTREAM_COMMIT,
    apply_face_refine_sampler_mask_patch,
    tensor_sha256,
)
from h3_audio_t8_pkg.nodes_face_refine_sampler_mask_advanced import (
    FACE_REFINE_SAMPLER_MASK_ADVANCED_NODE_CLASSES,
)


class _Sampling:
    shift = 12.0
    audio_shift = 3.0

    @staticmethod
    def noise_scaling(sigma, noise, clean):
        return clean + sigma * noise * 10.0


class _Base:
    def __init__(self, *, audio_scale: float = 1.0):
        self.model_sampling = _Sampling()
        self._audio_scale = audio_scale
        self.latent_shapes = None

    @staticmethod
    def _denoise_mask_conds(_denoise_mask, _latent_shapes):
        return {
            "denoise_mask": "video-condition",
            "audio_denoise_mask": "audio-condition",
            "other": "preserved",
        }

    @staticmethod
    def scale_latent_inpaint(**_kwargs):
        return torch.tensor([-999.0])

    def audio_scale(self):
        return self._audio_scale


class _Patcher:
    def __init__(self, base=None, object_patches=None):
        self.model = base or _Base()
        self.object_patches = dict(object_patches or {})

    def clone(self):
        return _Patcher(self.model, self.object_patches)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name))


def _fixture(*, audio_scale: float = 1.0):
    video = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2, 1)
    audio = torch.full((1, 2, 1, 2), 4.0)
    video_mask = torch.full_like(video, 0.5)
    audio_mask = torch.zeros_like(audio)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
    }
    report = {
        "schema": "h3_t8_face_refine_per_frame_denoise/v1",
        "status": "parity_per_frame_video_mask_applied",
        "plan_sha256": "a" * 64,
        "latent_time": 2,
        "video_mask_mode": "replace_video_parity",
        "require_locked_audio": True,
        "audio_mask_all_zero": True,
        "audio_samples_modified": False,
        "video_mask_sha256": tensor_sha256(video_mask),
        "audio_mask_sha256": tensor_sha256(audio_mask),
    }
    return _Patcher(_Base(audio_scale=audio_scale)), latent, json.dumps(report)


def test_patch_removes_only_video_model_condition_and_preserves_source_model():
    model, latent, report_json = _fixture()
    patched, returned_latent, patch_report_json = apply_face_refine_sampler_mask_patch(
        model, latent, report_json, enabled=True
    )

    assert returned_latent is latent
    assert model.object_patches == {}
    assert set(patched.object_patches) == {
        "_denoise_mask_conds",
        "scale_latent_inpaint",
    }
    conds = patched.object_patches["_denoise_mask_conds"](object(), [object(), object()])
    assert conds == {
        "audio_denoise_mask": "audio-condition",
        "other": "preserved",
    }

    patch_report = json.loads(patch_report_json)
    assert patch_report["schema"] == PATCH_SCHEMA
    assert patch_report["upstream_commit"] == UPSTREAM_COMMIT
    assert patch_report["video_mask_model_condition_removed"] is True
    assert patch_report["audio_mask_condition_preserved"] is True
    assert patch_report["source_model_mutated"] is False


def test_current_sigma_renoise_matches_sampling_clock_and_keeps_audio_at_scale_one():
    model, latent, report_json = _fixture()
    patched, _, _ = apply_face_refine_sampler_mask_patch(model, latent, report_json, enabled=True)
    video, audio = latent["samples"].unbind()
    base = model.model
    packed_clean, shapes = comfy.utils.pack_latents([video, audio])
    video_noise = torch.ones_like(video)
    audio_noise = torch.full_like(audio, 3.0)
    packed_noise, _ = comfy.utils.pack_latents([video_noise, audio_noise])
    base.latent_shapes = shapes

    output = patched.object_patches["scale_latent_inpaint"](
        sigma=torch.tensor([0.25]),
        noise=packed_noise,
        latent_image=packed_clean,
        x=torch.full_like(packed_clean, 123.0),
        denoise_mask=torch.zeros_like(packed_clean),
    )
    output_video, output_audio = comfy.utils.unpack_latents(output, shapes)

    assert torch.equal(output_video, video + 2.5 * video_noise)
    assert torch.equal(output_audio, audio)
    assert not torch.any(output == 123.0)


def test_audio_scale_uses_native_minimax_time_shift(monkeypatch):
    model, latent, report_json = _fixture(audio_scale=4.0)
    patched, _, _ = apply_face_refine_sampler_mask_patch(model, latent, report_json, enabled=True)
    video, audio = latent["samples"].unbind()
    packed_clean, shapes = comfy.utils.pack_latents([video, audio])
    packed_noise, _ = comfy.utils.pack_latents(
        [torch.zeros_like(video), torch.zeros_like(audio)]
    )
    model.model.latent_shapes = shapes
    seen = {}

    def _time_shift(sigma, shift, audio_shift):
        seen.update(sigma=sigma.clone(), shift=shift, audio_shift=audio_shift)
        return sigma * 0.5

    monkeypatch.setattr(
        "h3_audio_t8_pkg.face_refine_sampler_mask_advanced.minimax_model.time_shift_sigma",
        _time_shift,
    )
    output = patched.object_patches["scale_latent_inpaint"](
        sigma=torch.tensor([0.4]),
        noise=packed_noise,
        latent_image=packed_clean,
    )
    _output_video, output_audio = comfy.utils.unpack_latents(output, shapes)

    assert seen["shift"] == 12.0
    assert seen["audio_shift"] == 3.0
    assert torch.equal(output_audio, audio * 0.5)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report, _latent, _model: report.update(audio_mask_all_zero=False), "nonzero"),
        (
            lambda report, latent, _model: latent["noise_mask"].unbind()[0].fill_(0.25),
            "does not match",
        ),
        (
            lambda _report, _latent, model: model.object_patches.update(
                {"scale_latent_inpaint": object()}
            ),
            "must be callable",
        ),
    ],
)
def test_patch_fails_closed_for_invalid_contract(mutation, message):
    model, latent, report_json = _fixture()
    report = json.loads(report_json)
    mutation(report, latent, model)
    with pytest.raises((ValueError, TypeError), match=message):
        apply_face_refine_sampler_mask_patch(model, latent, json.dumps(report), enabled=True)


def test_foreign_scale_and_mask_owner_are_preserved_and_execute():
    model, latent, report_json = _fixture()
    calls = []
    def mask(*args):
        calls.append('mask')
        return {'denoise_mask': 1, 'audio_denoise_mask': 2, 'user': 3}
    def scale(**kwargs):
        calls.append('scale')
        return torch.tensor([17.])
    model.object_patches.update(_denoise_mask_conds=mask, scale_latent_inpaint=scale)
    patched, returned, report_json = apply_face_refine_sampler_mask_patch(model, latent, report_json, enabled=True)
    assert returned is latent and patched.object_patches['scale_latent_inpaint'] is scale
    assert patched.get_model_object('_denoise_mask_conds')(None, None) == {'audio_denoise_mask': 2, 'user': 3}
    assert patched.get_model_object('scale_latent_inpaint')().item() == 17
    assert calls == ['mask', 'scale']
    assert model.object_patches['_denoise_mask_conds'] is mask
    report = json.loads(report_json)
    assert report['status'] == 'executed_user_stack_unverified' and report['composition_verified'] is False


@pytest.mark.parametrize("kwargs", [{}, {"enabled": False}])
def test_disabled_is_exact_passthrough_without_validation_or_model_access(kwargs):
    model = object()
    latent = {"samples": object()}
    returned_model, returned_latent, report_json = apply_face_refine_sampler_mask_patch(
        model, latent, "not a report", **kwargs
    )
    assert returned_model is model
    assert returned_latent is latent
    report = json.loads(report_json)
    assert report["status"] == "disabled_passthrough"
    assert report["enabled"] is False
    assert report["object_patch_paths"] == []


def test_node_schema_is_append_only_experimental_model_latent_gate():
    assert len(FACE_REFINE_SAMPLER_MASK_ADVANCED_NODE_CLASSES) == 1
    schema = FACE_REFINE_SAMPLER_MASK_ADVANCED_NODE_CLASSES[0].define_schema()
    assert schema.node_id == "MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced"
    assert schema.is_experimental is True
    assert [item.id for item in schema.inputs] == [
        "model",
        "av_latent",
        "denoise_report_json",
        "enabled",
    ]
    assert schema.inputs[-1].default is False
    assert [item.id for item in schema.outputs] == ["model", "av_latent", "report_json"]


def test_published_workflows_and_quick_subgraph_keep_correction_disabled():
    root = Path(__file__).resolve().parents[1]
    kind = "MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced"
    found = 0
    for path in (root / "examples/workflows/06-face-refine").glob("*.json"):
        for node in json.loads(path.read_text(encoding="utf-8"))["nodes"]:
            if node["type"] == kind:
                assert node["widgets_values"] == [False]
                # Native ComfyUI saves unlinked widgets in widgets_values, not sockets.
                assert [item["name"] for item in node["inputs"]] == ["model", "av_latent", "denoise_report_json"]
                found += 1
    assert found == 3
    for name in ("face_refine_parity_advanced_api", "face_refine_window_advanced_api", "face_refine_window_studio_advanced_api"):
        graph = json.loads((root / "tests/fixtures/api" / f"{name}.json").read_text(encoding="utf-8"))
        patches = [node for node in graph.values() if node["class_type"] == kind]
        assert len(patches) == 1
        assert patches[0]["inputs"]["enabled"] is False
    quick = json.loads((root / "subgraphs/2026-08-22_H3_Quick_Face_Repair.json").read_text(encoding="utf-8"))
    patches = [node for node in quick["definitions"]["subgraphs"][0]["nodes"] if node["type"] == kind]
    assert len(patches) == 1
    assert patches[0]["widgets_values"] == [False]
