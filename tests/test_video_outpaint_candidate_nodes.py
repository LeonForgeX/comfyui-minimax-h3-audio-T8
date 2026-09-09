from __future__ import annotations

from copy import deepcopy
import json

import pytest

import h3_audio_t8_pkg.nodes_video_outpaint_candidates as module
import h3_audio_t8_pkg.nodes_video_outpaint as stages
from h3_audio_t8_pkg.video_outpaint_candidate_preview import render_candidate_first_frame
from h3_audio_t8_pkg.video_outpaint_candidates import candidate_receipt
from h3_audio_t8_pkg.video_outpaint_compose import compose_sampled_outpaint
from test_video_outpaint_compose import CombinedVAE, _prepared
from test_video_outpaint_execution import TinyClip, _model
from test_video_outpaint_media import _clip
from test_video_outpaint_sampling import _backend


def test_candidate_schemas_keep_confirmation_explicit_and_finishing_settings_bound():
    classes = module.VIDEO_OUTPAINT_CANDIDATE_DRAFT_NODE_CLASSES
    schemas = [cls.define_schema() for cls in classes]
    assert len({s.node_id for s in schemas}) == 7
    assert all(s.is_experimental for s in schemas)
    assert [s.is_output_node for s in schemas] == [True, False, False, True, True, True, True]
    assert next(i for i in schemas[0].inputs if i.id == "color_match").default is True
    assert next(i for i in schemas[0].inputs if i.id == "geometry_align").default is False
    assert next(i for i in schemas[0].inputs if i.id == "source_mode").default == "joint_decode"
    assert next(i for i in schemas[1].inputs if i.id == "confirm_selection").default is False
    assert [i.id for i in schemas[3].inputs] == ["sampled", "video_vae", "output_name"]
    with pytest.raises(ValueError, match="confirm_selection"):
        classes[1].execute({}, False)


@pytest.mark.parametrize('source_mode',['preserve_source','joint_decode'])
def test_native_nodes_preview_select_continue_and_bound_save(tmp_path, monkeypatch, source_mode):
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path / "outputs"))
    sample = module.sample_verified_first_candidate
    continuation = module.continue_verified_candidate
    monkeypatch.setattr(module, "sample_verified_first_candidate", lambda **kw: sample(**kw, sample_function=_backend))
    monkeypatch.setattr(module, "continue_verified_candidate", lambda **kw: continuation(**kw, sample_function=_backend))
    previews = []
    monkeypatch.setattr(module.ui, "PreviewImage", lambda image, **kw: previews.append(image) or {"images": []})
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=90, sound=False)
    plan = stages.MiniMaxH3VideoOutpaintPlanT8.execute(source, "custom", 32, 0, 32, 0,
        0.5, 0.5, 0.5, "73", "[]").result[0]
    vae, model = CombinedVAE(), _model()
    prepared = stages.MiniMaxH3VideoOutpaintPrepareT8.execute(plan, TinyClip(), vae,
        "scene", "[]", "candidates", 0, 64, False).result[0]
    result = module.MiniMaxH3VideoOutpaintCandidateT8.execute(model, prepared, vae,
        "candidate_a", 42, 20, False, True, source_mode=source_mode)
    candidate = result.result[0]
    assert previews[0].shape == (1, 32, 96, 3)
    assert candidate["windows"].snapshot()["status"] == "paused"
    assert candidate["settings"]["seed"] == 42 and result.ui == {"images": []}
    with pytest.raises(ValueError, match="confirm_selection"):
        module.MiniMaxH3VideoOutpaintSelectCandidateT8.execute(candidate, False)
    selected = module.MiniMaxH3VideoOutpaintSelectCandidateT8.execute(candidate, True).result[0]
    with pytest.raises(ValueError, match="尚未完成"):
        module.MiniMaxH3VideoOutpaintLoadCompletedSelectionT8.execute(
            prepared, "candidate_a", selected["selection_id"]
        )
    broken = {**candidate, "preview_report": deepcopy(candidate["preview_report"])}
    broken["preview_report"]["color_settings"]["enabled"] = not broken["preview_report"]["color_settings"]["enabled"]
    with pytest.raises(ValueError, match="preview identity"):
        module.MiniMaxH3VideoOutpaintSelectCandidateT8.execute(broken, True)
    broken = {**candidate, "preview_report": deepcopy(candidate["preview_report"])}
    broken['preview_report']['geometry_settings']['geometry_align'] = True
    with pytest.raises(ValueError, match='preview identity'):
        module.MiniMaxH3VideoOutpaintSelectCandidateT8.execute(broken, True)
    broken = {**candidate, "preview_report": deepcopy(candidate["preview_report"])}
    broken['preview_report']['source_mode'] = 'joint_decode' if source_mode == 'preserve_source' else 'preserve_source'
    with pytest.raises(ValueError, match='preview identity'):
        module.MiniMaxH3VideoOutpaintSelectCandidateT8.execute(broken, True)
    completed = module.MiniMaxH3VideoOutpaintContinueCandidateT8.execute(model, selected).result[0]
    assert len(completed["windows"].snapshot()["committed"]) == 2
    saved = module.MiniMaxH3VideoOutpaintComposeCandidateT8.execute(completed, vae, "chosen")
    report = json.loads(saved.result[1])
    assert saved.result[0].get_frame_count() == 90
    assert report["selected_first_frame_verified_before_encoding"]
    assert report["selected_first_frame_rgb8_sha256"] == candidate["preview_report"]["rgb8_sha256"]
    assert report["color_match_enabled"] is (source_mode == 'preserve_source')
    assert report['source_mode'] == source_mode and not report["perceptual_acceptance"]
    assert (tmp_path / "outputs/T8_H3_Outpaint/chosen_00001.mp4").is_file()
    assert not (prepared["root"] / "sampling").exists()
    restored = module.MiniMaxH3VideoOutpaintLoadCandidateT8.execute(prepared, "candidate_a", result.result[3])
    assert restored.result[0]["windows"] is not completed["windows"]
    assert restored.result[0]['preview_report']['source_mode'] == source_mode
    assert len(restored.result[0]["windows"].snapshot()["committed"]) == 2
    assert (restored.result[1] == result.result[1]).all()
    assert "selection" not in restored.result[0]
    recovered = module.MiniMaxH3VideoOutpaintLoadSelectionT8.execute(prepared, "candidate_a", selected["selection_id"])
    assert recovered.result[0]["selection"] == selected["selection"]
    completed_reload = module.MiniMaxH3VideoOutpaintLoadCompletedSelectionT8.execute(
        prepared, "candidate_a", selected["selection_id"]
    )
    assert completed_reload.result[0]["windows"].snapshot()["status"] == "sampled"
    assert json.loads(completed_reload.result[2])["sampling_called"] is False
    again = module.MiniMaxH3VideoOutpaintContinueCandidateT8.execute(model, recovered.result[0])
    assert json.loads(again.result[1])["windows_sampled_this_call"] == 0


def test_invalid_source_mode_is_rejected_before_candidate_generation():
    with pytest.raises(ValueError, match='source_mode'):
        module.MiniMaxH3VideoOutpaintCandidateT8.execute(None, {}, None,
            'invalid',42,20,False,True,source_mode='unknown')


def test_incorrect_selected_first_frame_never_publishes_video(tmp_path):
    inputs = _prepared(tmp_path)
    before = inputs["window_store"].path.read_bytes()
    with pytest.raises(ValueError, match="first-frame RGB differs"):
        compose_sampled_outpaint(**inputs, output_path=tmp_path / "wrong.mp4",
                                expected_first_frame_sha256="0" * 64)
    assert not (tmp_path / "wrong.mp4").exists()
    assert not (tmp_path / "wrong.mp4.outpaint.json").exists()
    assert not list(tmp_path.glob(".*.video-only-*.mp4"))
    assert inputs["window_store"].path.read_bytes() == before
    _, preview = render_candidate_first_frame(**inputs, candidate=candidate_receipt(inputs["window_store"]))
    _, report = compose_sampled_outpaint(**inputs, output_path=tmp_path / "right.mp4",
                                       expected_first_frame_sha256=preview["rgb8_sha256"])
    assert report["selected_first_frame_verified_before_encoding"]


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "z" * 64, False])
def test_invalid_selected_frame_digest_rejected_before_work(digest):
    with pytest.raises(ValueError, match="first-frame SHA256"):
        compose_sampled_outpaint(vae=None, inspection=None, source_store=None, window_store=None,
                                output_path="unused.mp4", expected_first_frame_sha256=digest)
