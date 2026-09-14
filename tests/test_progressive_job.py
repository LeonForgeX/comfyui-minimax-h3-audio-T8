"""Bound job/lock/real tiny continuation; native encoder shells are not trained."""

import json

import comfy.samplers
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg.progressive_job import NativeProgressiveJob
from h3_audio_t8_pkg.progressive_continuation import ProgressiveContinuationSource
from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.long_video_in_node_loop_advanced import LOOP_LOCK_NAME
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_progressive_producers import component
from test_progressive_sampling_runtime import tiny_model, stub_lifter  # noqa: F401
from test_progressive_continuation import accepted  # noqa: F401
from test_progressive_checkpoint import same
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE


def job(**kwargs):
    values = dict(clip=component('clip'), video_vae=component('video_vae'), audio_vae=component('audio_vae'),
        chain_id='chain', total_duration_seconds=8, width=128, height=64, upscaler_model='test',
        global_prompt='Continue walking.', base_seed=8)
    values.update(kwargs)
    return NativeProgressiveJob(tiny_model(), tiny_model(), comfy.samplers.ksampler('euler'),
                               native_flow_sigmas(8, 12.), **values)


def bind_parent(case, sha):
    # Fixture was generated as actual MP4/context. Rebind its descriptor and
    # both context copies to the newly derived actual job SHA, never a supplied
    # string accepted by the production job constructor.
    # Detach safetensors' mmap storage before rewriting this fixture on Windows.
    tensors = {key: value.clone() for key, value in load_file(str(case.accepted_context)).items()}
    with safe_open(str(case.accepted_context), framework='pt') as handle:
        metadata = dict(handle.metadata())
    metadata['sampling_summary'] = sha
    candidate_context = case.root / case.info['context_path']
    save_file(tensors, str(candidate_context), metadata=metadata)
    case.accepted_context.write_bytes(candidate_context.read_bytes())
    context_sha = delivery._sha256_file(candidate_context)
    case.info.update(sampling_summary=sha, context_sha256=context_sha)
    case.manifest['segments'][0].update(sampling_summary=sha, context_sha256=context_sha)
    case.descriptor.write_text(json.dumps(case.info))
    case.manifest_path.write_text(json.dumps(case.manifest))


@pytest.mark.parametrize('field', ['prompt', 'weights', 'vae', 'sigmas', 'options', 'media'])
def test_mutations_invalidate_actual_job(stub_lifter, field):  # noqa: F811
    value = job(segment_condition_options={1: {'first_frame': torch.zeros(1, 32, 64, 3)}})
    assert len(value.segments) == 2 and value.segments[1].seed == 9
    assert value.verify() == value.sha256
    if field == 'prompt':
        value.plan_inputs['global_prompt'] += 'Changed.'
    elif field == 'weights':
        with torch.no_grad():
            next(value.models[1].model.parameters()).add_(.1)
    elif field == 'vae':
        value.producers.components['video_vae'].crop_input = False
    elif field == 'sigmas':
        value.sigmas[1] += .001
    elif field == 'media':
        value.condition_options[1]['first_frame'].add_(.1)
    else:
        value.options['eav_tau'] += .1
    with pytest.raises(ValueError):
        value.verify()


def test_job_source_requires_lock_and_derives_actual_parent(accepted, stub_lifter):  # noqa: F811
    value = job()
    with pytest.raises(RuntimeError, match='lock'):
        value.source(1)
    with pytest.raises(ValueError, match='another progressive job'):
        with value.exclusive(accepted.root):
            pytest.fail('Foreign parent accepted')
    assert not (accepted.root / 'progressive_job.json').exists()
    bind_parent(accepted, value.sha256)
    with value.exclusive(accepted.root):
        source = value.source(1)
        assert source.binding['request']['job_sha256'] == value.sha256
        assert source.binding['request']['parent_candidate_id'] == 'parent'
        handle = delivery._open_advisory_lock(accepted.root / LOOP_LOCK_NAME)
        try:
            assert not delivery._try_advisory_lock(handle)
        finally:
            handle.close()
    assert not value.active


def test_changed_contract_cannot_adopt_saved_empty_job(tmp_path, stub_lifter):  # noqa: F811
    first = job()
    with first.exclusive(tmp_path):
        pass
    previous = (tmp_path / 'progressive_job.json').read_bytes()
    second = job(global_prompt='Another job.')
    with pytest.raises(ValueError, match='contract differs'):
        with second.exclusive(tmp_path):
            pytest.fail('Rebound immutable job')
    assert (tmp_path / 'progressive_job.json').read_bytes() == previous


def test_job_executes_bound_continuation_and_replays_low(
        accepted, stub_lifter, monkeypatch):  # noqa: F811
    value = job()
    bind_parent(accepted, value.sha256)
    prepare = ProgressiveContinuationSource.prepare_conditions
    seen = []
    def prepared(source, *, clip, video_vae, audio_vae, **kwargs):
        assert all(actual is value.producers.components[role] for role, actual in
                   [('clip', clip), ('video_vae', video_vae), ('audio_vae', audio_vae)])
        seen.append(kwargs['prompt'])
        # Explicit encoder test doubles; actual H3/Euler below is unmocked.
        return prepare(source, clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), **kwargs)
    from test_progressive_continuation_runtime import run
    with value.exclusive(accepted.root):
        baseline, _ = run(value.source(1), value.models[0], model_hires=value.models[1])
    monkeypatch.setattr(ProgressiveContinuationSource, 'prepare_conditions', prepared)
    with value.exclusive(accepted.root):
        def cancel(step, *_):
            if step == 5:
                raise InterruptedError('HIGH interrupted')
        with pytest.raises(InterruptedError):
            value.sample_continuation(1, callback=cancel)
    callbacks = []
    with value.exclusive(accepted.root):
        first, report = value.sample_continuation(1, callback=lambda step, *_: callbacks.append(step))
    assert callbacks == [4, 5, 6, 7]
    same(first, baseline)
    callbacks = []
    with value.exclusive(accepted.root):
        second, text = value.sample_continuation(1, callback=lambda step, *_: callbacks.append(step))
    same(first, second)
    assert seen == ['Continue walking.'] * 2
    assert callbacks == []
    assert json.loads(report)['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert json.loads(text)['counts']['actual_forwards'] == {'low': 0, 'high': 0}
    assert json.loads(text)['checkpoint']['reused_completed']
    assert json.loads(text)['job']['sha256'] == value.sha256
    # A damaged completed receipt must fail before preparing/launching either
    # stage, not silently fall back to an expensive new sampling run.
    path = next((accepted.root / 'progressive_segments/1/completed').glob('high_output-*.json'))
    record = json.loads(path.read_text())
    record['report']['sampling']['counts']['actual_forwards']['high'] = 3
    path.write_text(json.dumps(record))
    with value.exclusive(accepted.root):
        with pytest.raises(ValueError, match='report integrity'):
            value.sample_continuation(1)
    assert seen == ['Continue walking.'] * 2
