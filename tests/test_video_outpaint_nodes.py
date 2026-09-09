import json

import pytest

import h3_audio_t8_pkg.nodes_video_outpaint as module
from test_video_outpaint_execution import TinyClip, _model
from test_video_outpaint_compose import CombinedVAE
from test_video_outpaint_media import _clip
from test_video_outpaint_sampling import _backend


def test_draft_schemas_are_distinct_and_color_match_defaults_on():
    schemas = [cls.define_schema() for cls in module.VIDEO_OUTPAINT_DRAFT_NODE_CLASSES]
    assert len({schema.node_id for schema in schemas}) == 4
    assert all(schema.is_experimental for schema in schemas)
    assert next(i for i in schemas[-1].inputs if i.id == "color_match").default is True
    assert schemas[-1].is_output_node


def test_four_draft_stages_form_reusable_video_without_overwriting_output(tmp_path, monkeypatch):
    monkeypatch.setattr(module.folder_paths, "get_output_directory", lambda: str(tmp_path / "outputs"))
    original = module.sample_verified_outpaint
    monkeypatch.setattr(module, "sample_verified_outpaint", lambda **kw: original(**kw, sample_function=_backend))
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=39, sound=False)
    planned = module.MiniMaxH3VideoOutpaintPlanT8.execute(source, "custom", 32, 0, 32, 0, 0.5, 0.5, 0.5, "73", "[]")
    handle = planned.result[0]
    vae = CombinedVAE()
    prepared = module.MiniMaxH3VideoOutpaintPrepareT8.execute(handle, TinyClip(), vae, "scene", "[]",
                                                            "test_run", 0, 64, False).result[0]
    sampled = module.MiniMaxH3VideoOutpaintSampleT8.execute(_model(), prepared, 42, 20, False).result[0]
    first = module.MiniMaxH3VideoOutpaintComposeT8.execute(sampled, vae, "result", True)
    assert first.result[0].get_dimensions() == (96, 32)
    assert first.result[0].get_frame_count() == first.result[0].get_frame_count() == 39
    report = json.loads(first.result[1])
    assert report["pre_encode_pixel_evidence_verified"] and not report["perceptual_acceptance"]
    path = tmp_path / "outputs/T8_H3_Outpaint/result_00001.mp4"
    before = path.read_bytes()
    module.MiniMaxH3VideoOutpaintComposeT8.execute(sampled, vae, "result", False)
    assert path.read_bytes() == before
    assert (tmp_path / "outputs/T8_H3_Outpaint/result_00002.mp4").exists()
    orphan = path.with_name("result_00003.mp4.outpaint.json")
    orphan.write_bytes(b"orphan report is not overwritten")
    module.MiniMaxH3VideoOutpaintComposeT8.execute(sampled, vae, "result", False)
    assert orphan.read_bytes() == b"orphan report is not overwritten"
    assert (path.parent / "result_00004.mp4").exists()


@pytest.mark.parametrize("name", ["../escape", "C:/bad", "CON", "a.b", ""])
def test_output_and_cache_names_cannot_escape_or_use_windows_devices(name):
    with pytest.raises(ValueError, match="name"):
        module._name(name)


@pytest.mark.parametrize("prompts,track", [('["one", "extra"]', 0), ('[123]', 0), ('[]', 1)])
def test_invalid_shot_or_track_is_rejected_before_expensive_preparation(tmp_path, monkeypatch, prompts, track):
    source = _clip(tmp_path / "source.mp4", 32, 32, frames=39, sound=False)
    handle = module.MiniMaxH3VideoOutpaintPlanT8.execute(source, "custom", 32, 0, 32, 0,
                                                       0.5, 0.5, 0.5, "73", "[]").result[0]
    def forbidden(*args, **kwargs):
        pytest.fail("VAE preparation must not run for invalid shot/track inputs")
    monkeypatch.setattr(module, "prepare_outpaint_source_cache", forbidden)
    with pytest.raises(ValueError, match="shot|track"):
        module.MiniMaxH3VideoOutpaintPrepareT8.execute(handle, TinyClip(), CombinedVAE(), "scene", prompts,
                                                      "test_run", track, 64, False)
