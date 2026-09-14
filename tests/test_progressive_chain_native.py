"""Actual tiny H3/Euler -> original AV decode wrapper -> real 8s media chain.

CLIP, VAE networks and learned lifter are explicitly doubled, not trained.
Neither sampling/job/cache nor candidate/context/assembly is replaced.
"""

import json
from pathlib import Path

import av
import pytest
import torch

from h3_audio_t8_pkg import progressive_delivery as runner
from h3_audio_t8_pkg import progressive_first_segment as first
from h3_audio_t8_pkg import progressive_continuation_relay as relay
from h3_audio_t8_pkg.progressive_continuation import ProgressiveContinuationSource
from test_progressive_job import job
from test_progressive_first_segment import plan
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from helpers import FakeAudioVAE, FakeVideoVAE


@pytest.mark.parametrize('mode', ['t2va_joint', 'i2va_locked', 'public_i2va_locked'])
def test_native_sampling_through_whole_media_chain(monkeypatch, tmp_path, stub_lifter, mode):  # noqa: F811
    monkeypatch.setattr(runner.delivery.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    source = {'waveform': torch.linspace(.02, .1, 8 * 32000).reshape(1, 1, -1), 'sample_rate': 32000}
    options = {'final_audio': source}
    if mode in ('i2va_locked', 'public_i2va_locked'):
        options = dict(drive_audio=source, audio_mode='lock_source', add_source_as_reference=False,
            first_frame=torch.linspace(0, 1, 32 * 64 * 3).reshape(1, 32, 64, 3))
    if mode == 'public_i2va_locked':
        options.update(context_audio='video_and_audio', audio_denoise_strength=.35,
                       first_frame_reuse='segment0_only')
    value = job(shared_condition_options=options,
        prompt_relay_plan=plan(route='video_only_paper' if mode.endswith('i2va_locked') else 'joint_av_exp'),
        sampling_options={'eav_mode': 'apply_exp', 'eav_tau': .2})
    fake_clip, fake_video, fake_audio = NativeLikeFakeClip(), FakeVideoVAE(), FakeAudioVAE()
    prepared, decoded, audio_windows = {}, [], []

    def encoders(kwargs):
        for key, component in value.producers.components.items():
            assert kwargs[key] is component
        return {**kwargs, 'clip': fake_clip, 'video_vae': fake_video, 'audio_vae': fake_audio}

    original_first = first.prepare_first_segment
    def prepare_first(**kwargs):
        result = original_first(**encoders(kwargs))
        prepared[0] = result[0][3]
        return result
    monkeypatch.setattr(first, 'prepare_first_segment', prepare_first)
    original_bind = first.bind_first_relay
    def bind(model, result, projected, clip, chunks):
        assert clip is value.producers.components['clip']
        return original_bind(model, result, projected, fake_clip, chunks)
    monkeypatch.setattr(first, 'bind_first_relay', bind)

    original_prepare = ProgressiveContinuationSource.prepare_conditions
    def prepare_continuation(source_binding, **kwargs):
        result = original_prepare(source_binding, **encoders(kwargs))
        prepared[source_binding.binding['request']['segment_index']] = result[1][3]
        return result
    monkeypatch.setattr(ProgressiveContinuationSource, 'prepare_conditions', prepare_continuation)
    original_relay = relay.bind_prepared_relay
    def bind_continuation(clip, *args):
        assert clip is value.producers.components['clip']
        return original_relay(fake_clip, *args)
    monkeypatch.setattr(relay, 'bind_prepared_relay', bind_continuation)

    original_decode = runner.decode_av_latent
    def decode(latent, video_vae, audio_vae):
        assert video_vae is value.producers.components['video_vae']
        assert audio_vae is value.producers.components['audio_vae']
        decoded.append(len(decoded))
        # Execute original production AV unpack/decode/audio normalize logic.
        return original_decode(latent, fake_video, fake_audio)
    monkeypatch.setattr(runner, 'decode_av_latent', decode)
    original_save = runner.delivery.save_long_video_candidate
    def save(frames, audio, latent, chain_id, index, *args):
        start, end = (0, 165333) if index == 0 else (165333, 256000)
        torch.testing.assert_close(audio['waveform'], source['waveform'][..., start:end], rtol=0, atol=0)
        audio_windows.append(index)
        return original_save(frames, audio, latent, chain_id, index, *args)
    monkeypatch.setattr(runner.delivery, 'save_long_video_candidate', save)

    controller = runner.ProgressiveChainDelivery(value)
    def public_run():
        from h3_audio_t8_pkg.nodes_progressive_long_video import MiniMaxH3ProgressiveLongVideoEXPT8 as Public
        from h3_audio_t8_pkg import nodes_long_video_in_node_loop_advanced as preview
        # Only UI preview packaging is stubbed. The public node constructs the
        # real content-bound job, performs native sampling and writes full media.
        monkeypatch.setattr(preview, '_preview_video', lambda path: (path, None))
        inputs = {item.id: item.default for item in Public.define_schema().inputs
                  if hasattr(item, 'default') and item.default is not None}
        inputs.update(value.plan_inputs)
        inputs.update(model=value.models[0], model_hires=value.models[1],
            sampler=value.sampler, sigmas=value.sigmas, **value.producers.components,
            width=value.width, height=value.height, upscaler_model='test',
            prompt_relay_plan=value.relay_plan, eav_mode='apply_exp', eav_tau=.2,
            drive_audio=source, audio_mode='lock_source', first_frame=options['first_frame'])
        result = Public.execute(**inputs)
        return result.result[1], result.result[2]
    calls = []
    def progress(index, step, *args):
        calls.append((index, step))
        if mode == 't2va_joint' and index == 1 and step == 5:
            raise InterruptedError('native HIGH interruption')
    if mode == 't2va_joint':
        with pytest.raises(InterruptedError, match='native HIGH'):
            controller.run(callback=progress)
        assert decoded == [0] and audio_windows == [0]
        assert calls == [(0, i) for i in range(8)] + [(1, i) for i in range(6)]
        calls.clear()
        output, text = controller.run(callback=lambda index, step, *_: calls.append((index, step)))
        assert calls == [(1, i) for i in range(4, 8)]
    elif mode == 'public_i2va_locked':
        output, text = public_run()
    else:
        output, text = controller.run(callback=progress)
        assert calls == [(index, i) for index in range(2) for i in range(8)]
    assert decoded == [0, 1] and audio_windows == [0, 1]
    assert not json.loads(text)['human_qualified']
    for index in range(2):
        body = runner._read_receipt(controller.root / 'progressive_delivery' / f'segment_{index:05d}.json')
        report = body['sampling']
        assert report['prepared_delivery']['conditioned_prompt'] == prepared[index]
        descriptor = json.loads((controller.root / body['candidate_path']).read_text())
        assert descriptor['prompt'] == prepared[index]
        assert report['counts']['actual_forwards']['high'] == 4
        assert report['prompt_relay']['high']['completed_calls']['forward'] == 4
        if index:
            assert report['eav']['high']['verified_native_mask_forwards'] == 4
    with av.open(output) as container:
        assert len(list(container.decode(video=0))) == 192
    with av.open(output) as container:
        assert list(container.decode(audio=0))
    before = Path(output).read_bytes()
    again, cached = (public_run() if mode == 'public_i2va_locked' else
                     controller.run(callback=lambda *_: pytest.fail('Final resume sampled again')))
    assert again == output and Path(again).read_bytes() == before and json.loads(cached)['reused_final']
    assert decoded == [0, 1] and not value.active
