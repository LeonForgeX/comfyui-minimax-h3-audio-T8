"""Public wiring/CPU setup and actual OS-locked non-destructive restart guard."""

import asyncio
import copy
import json
import math

import pytest
import torch

from h3_audio_t8_pkg.nodes_progressive_long_video import MiniMaxH3ProgressiveSetupEXPT8 as Setup
from h3_audio_t8_pkg.nodes_progressive_long_video import MiniMaxH3ProgressiveLongVideoEXPT8 as Long
from h3_audio_t8_pkg import nodes as registry
from h3_audio_t8_pkg import progressive_job as jobs
from h3_audio_t8_pkg import progressive_delivery as deliveries
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_job import job


def test_setup_real_native_pair_preserves_original_and_clocks():
    low, high = tiny_model(), tiny_model()
    originals = [(dict(m.object_patches), copy.deepcopy(m.model_options)) for m in (low, high)]
    ready_low, ready_high, sampler, sigmas = Setup.execute(low, 8, 12., 3., high).result
    assert ready_low is not low and ready_high is not high
    assert ready_low.model is low.model and ready_high.model is high.model
    assert sigmas.shape == (9,) and sigmas[0] == 1 and sigmas[-1] == 0
    assert sampler.sampler_function.__name__ == 'sample_euler'
    for model, before in zip((low, high), originals):
        assert model.object_patches == before[0] and model.model_options == before[1]
    for model in (ready_low, ready_high):
        assert model.get_model_object('model_sampling').audio_shift == 3.


def test_setup_implicit_high_and_full_schedule():
    source = tiny_model()
    low, high, _, sigmas = Setup.execute(source, 20, 12., 3.).result
    assert low.model is high.model is source.model
    assert torch.count_nonzero(sigmas) == 20


def test_registration_appends_two_nodes_and_exposes_disk_checked_output():
    classes = asyncio.run(registry.comfy_entrypoint().get_node_list())
    assert classes[-2:] == [Setup, Long]
    ids = [cls.define_schema().node_id for cls in classes]
    assert len(ids) == len(set(ids))
    schema = Long.define_schema()
    assert schema.is_output_node
    inputs = {v.id: v for v in schema.inputs}
    assert inputs['resume_existing'].default is True
    assert inputs['context_frames'].options == [22, 39]
    assert inputs['prompt_relay_plan'].io_type == 'H3_T8_PROMPT_RELAY_PLAN'
    assert inputs['eav_tau'].default == pytest.approx(4.)
    assert inputs['eav_start_video_progress'].default == pytest.approx(.15)
    assert inputs['eav_end_video_progress'].default == pytest.approx(.90)
    assert math.isnan(Long.fingerprint_inputs(chain_id='test'))


@pytest.mark.usefixtures('stub_lifter')
def test_resume_false_rejects_existing_job_without_touching_data(tmp_path):
    value = job()
    with value.exclusive(tmp_path, resume_existing=False):
        pass
    saved = (tmp_path / 'progressive_job.json').read_bytes()
    with pytest.raises(ValueError, match='chain contains data'):
        with value.exclusive(tmp_path, resume_existing=False):
            pytest.fail('Must not yield an existing chain')
    assert (tmp_path / 'progressive_job.json').read_bytes() == saved
    assert not value.active
    # The rejecting path released its OS lock; unchanged resume still works.
    with value.exclusive(tmp_path, resume_existing=True):
        assert value.active


@pytest.mark.usefixtures('stub_lifter')
def test_resume_false_does_not_adopt_unreceipted_artifacts(tmp_path):
    value = job()
    artifact = tmp_path / 'orphan.mp4'
    artifact.write_bytes(b'keep this user-owned evidence')
    with pytest.raises(ValueError, match='chain contains data'):
        with value.exclusive(tmp_path, resume_existing=False):
            pytest.fail('Must not adopt old artifacts')
    assert artifact.read_bytes() == b'keep this user-owned evidence'
    assert not (tmp_path / 'progressive_job.json').exists()


def test_public_execution_forwards_inputs_and_resume_without_own_sampling(monkeypatch):
    received = {}
    class Bound:
        sigmas = torch.linspace(1, 0, 9)
        segments = [object(), object()]
    def construct(**kwargs):
        received['job'] = kwargs
        return Bound()
    class Delivery:
        def __init__(self, value, **kwargs):
            assert isinstance(value, Bound)
            received['delivery'] = kwargs
        def run(self, **kwargs):
            received['run'] = kwargs
            return '/unreviewed.mp4', json.dumps({'human_qualified': False})
    monkeypatch.setattr(jobs, 'NativeProgressiveJob', construct)
    monkeypatch.setattr(deliveries, 'ProgressiveChainDelivery', Delivery)
    from h3_audio_t8_pkg import nodes_long_video_in_node_loop_advanced as preview
    video, ui = object(), object()
    monkeypatch.setattr(preview, '_preview_video', lambda path: (video, ui))
    values = {item.id: item.default for item in Long.define_schema().inputs if hasattr(item, 'default') and item.default is not None}
    model, high, clip, va, aa, frame, audio = [object() for _ in range(7)]
    values.update(model=model, model_hires=high, clip=clip, video_vae=va, audio_vae=aa,
        sampler=object(), sigmas=torch.linspace(1, 0, 9), upscaler_model='test',
        first_frame=frame, final_audio=audio, eav_mode='apply_exp', resume_existing=False)
    result = Long.execute(**values)
    assert result.result[0] is video
    assert received['job']['model'] is model and received['job']['model_hires'] is high
    assert received['job']['shared_condition_options']['first_frame'] is frame
    assert received['job']['shared_condition_options']['final_audio'] is audio
    assert received['job']['sampling_options']['eav_mode'] == 'apply_exp'
    assert received['run']['resume_existing'] is False
    assert callable(received['run']['callback'])
