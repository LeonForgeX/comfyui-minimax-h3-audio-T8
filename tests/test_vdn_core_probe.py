from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.tools import build_vdn_two_pass_workflows as builder
from h3_audio_t8_pkg.tools.run_vdn_core_compat_probe import finalize_resources
from h3_audio_t8_pkg.tools import run_vdn_core_compat_probe as probe


@pytest.mark.parametrize("minimum,samples,expected", [(498, 292, False), (512, 1, True), (None, 0, False), (2000, 0, False)])
def test_resource_gate_is_not_media_success(minimum, samples, expected):
    report = {"status": "mechanical_pass_human_review_pending"}
    finalize_resources(report, {"minimum_free_mib": minimum, "samples": samples})
    assert (report["status"] == "mechanical_pass_human_review_pending") is expected


def test_resource_gate_does_not_overwrite_execution_failure():
    report = {"status": "failed", "error": "execution failed"}
    finalize_resources(report, {"minimum_free_mib": 15000, "samples": 5})
    assert report["status"] == "failed"


@pytest.mark.parametrize("stdout,valid", [("SHA256=" + "a" * 64 + "\n", True), ("", False), ("SHA256=bad", False)])
def test_decoded_audio_hash_is_strict(monkeypatch, stdout, valid):
    def run(command, **kwargs):
        assert "pcm_f32le" in command
        assert kwargs["check"] is True
        return SimpleNamespace(stdout=stdout)
    monkeypatch.setattr(probe.subprocess, "run", run)
    if valid:
        assert probe.decoded_audio_sha256("test.mp4", "ffmpeg") == "a" * 64
    else:
        with pytest.raises(RuntimeError, match="SHA256"):
            probe.decoded_audio_sha256("test.mp4", "ffmpeg")


@pytest.mark.parametrize("attention", ["default", "pytorch_before", "sparse_before", "sparse_after"])
def test_two_pass_graph_preserves_completed_first_pass_and_reports(attention):
    args = SimpleNamespace(width=320, height=192, frame_count=39, seed=5,
                           base_model=builder.base.BASE_MODEL, two_pass=True,
                           scale_by=2., refine_steps=4, attention=attention)
    graph = builder.build_prompt(args, "test")
    assert graph["60"]["inputs"]["av_latent"] == ["10", 0]
    assert graph["63"]["inputs"]["first_pass_latent"] == ["10", 0]
    assert graph["70"]["inputs"]["av_latent"] == ["10", 0]
    assert graph["11"]["inputs"]["av_latent"] == ["67", 0]
    assert graph["62"]["inputs"]["second_pass_audio_strength"] == 0.
    assert graph["74"]["inputs"]["source"] == ["67", 1]
    assert graph["63"]["inputs"]["model"] == (["50", 0] if attention == "sparse_after" else ["5", 0])
    assert "13" not in graph


def test_native_second_pass_uses_clean_model_not_vdn_or_its_adapters():
    args = SimpleNamespace(width=320, height=192, frame_count=39, seed=5,
                           base_model=builder.base.BASE_MODEL, two_pass=True,
                           scale_by=2., refine_steps=4, attention="sparse_after",
                           refine_backend="native_h3", native_refine_lora="new-ema-B.safetensors")
    graph = builder.build_prompt(args, "native-second")
    assert graph["80"] == graph["4"] and graph["80"] is not graph["4"]
    assert graph["81"]["inputs"]["model"] == ["80", 0]
    assert graph["63"]["inputs"]["model"] == ["81", 0]
    assert graph["66"]["inputs"]["sigmas"] == ["82", 1]
    assert graph["10"]["inputs"]["sigmas"] == ["7", 2]
    assert not any(value == ["82", 0] for item in graph.values() for value in item["inputs"].values())


def test_safe_probe_save_consumes_raw_av_not_native_encoded_video():
    args = SimpleNamespace(width=320, height=192, frame_count=39, seed=5,
                           base_model=builder.base.BASE_MODEL, two_pass=True,
                           scale_by=2., refine_steps=4, safe_probe_save=True)
    graph = builder.build_prompt(args, "safe")
    assert graph["18"]["class_type"] == graph["73"]["class_type"] == "T8VDNSafeProbeSave"
    assert graph["18"]["inputs"]["images"] == ["12", 0]
    assert graph["73"]["inputs"]["images"] == ["71", 0]
    assert "video" not in graph["18"]["inputs"]


def test_registered_safe_saver_is_the_workflow_output_not_probe_or_double_encode():
    args = SimpleNamespace(width=320, height=192, frame_count=39, seed=5,
                           base_model=builder.base.BASE_MODEL, two_pass=True,
                           scale_by=2., refine_steps=4, video_save="isolated")
    graph = builder.build_prompt(args, "safe")
    assert graph["18"]["class_type"] == graph["73"]["class_type"] == "MiniMaxH3SafeAVSaveT8Advanced"
    assert graph["88"]["inputs"]["source"] == ["18", 2]
    assert not any(item["class_type"] in ("SaveVideo", "CreateVideo", "T8VDNSafeProbeSave") for item in graph.values())


@pytest.mark.parametrize("variant", list(probe.multimodal.VARIANTS))
def test_core_probe_reuses_all_existing_multimodal_inputs(variant):
    args = SimpleNamespace(width=320, height=192, frame_count=39, seed=5,
                           base_model=builder.base.BASE_MODEL, two_pass=False, variant=variant,
                           attention="sparse_before", image="image.png", first_image="first.png",
                           last_image="last.png", ref_image_1="ref1.png", ref_image_2="ref2.png",
                           ref_video="ref.mp4")
    graph = builder.build_prompt(args, "matrix")
    expected = probe.multimodal.build_variant_prompt(args, "matrix", variant_name=variant)
    assert graph["6"] == expected["6"]
    assert "13" not in graph
    assert graph["18"]["class_type"] == "SaveVideo"
    assert graph["5"]["inputs"]["model"] == ["50", 0]
