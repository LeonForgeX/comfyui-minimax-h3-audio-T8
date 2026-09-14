"""Real job/locks/MP4/context/assembly; sampling and VAE decode are explicit doubles.

This tests durable delivery, not trained-model or image-quality qualification.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import av
import pytest
import torch

from h3_audio_t8_pkg import progressive_delivery as runner
from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.core import empty_av_latent
from test_progressive_job import job
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def install(monkeypatch, tmp_path):
    monkeypatch.setattr(delivery.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    source = {'waveform': torch.linspace(0., .2, 8 * 32000).reshape(1, 1, -1), 'sample_rate': 32000}
    value = job(shared_condition_options={'drive_audio': source, 'add_source_as_reference': False})
    sampled_indices, decoded_indices, cache = [], [], {}

    def sample(index, callback=None):
        value._locked()
        value.verify()
        if index:
            source_binding = value.source(index).binding
            assert source_binding['accepted']['timeline_end_frame'] == 124
        sampled_indices.append(index)
        if index in cache:
            latent, historical = cache[index]
            return latent, json.dumps(dict(historical_sampling=historical,
                counts=dict(actual_forwards=dict(low=0, high=0))))
        latent, _ = empty_av_latent(128, 64, 124)
        for tensor in latent['samples'].unbind():
            tensor.fill_(index * .1)
        report = dict(job=dict(sha256=value.sha256, segment_index=index),
                      counts=dict(actual_forwards=dict(low=4, high=4)))
        _, audio_report = value.delivery_audio(index, None)
        report['prepared_delivery'] = dict(conditioned_prompt=value.segments[index].prompt,
                                           mux_audio_identity=audio_report['identity'])
        cache[index] = latent, report
        return latent, json.dumps(report)

    monkeypatch.setattr(value, 'sample_first', lambda **kw: sample(0, **kw))
    monkeypatch.setattr(value, 'sample_continuation', sample)

    def decode(latent, video_vae, audio_vae):
        assert video_vae is value.producers.components['video_vae']
        assert audio_vae is value.producers.components['audio_vae']
        index = round(float(latent['samples'].unbind()[0][0, 0, 0, 0, 0]) * 10)
        decoded_indices.append(index)
        frames = torch.linspace(.1, .7, 124).reshape(-1, 1, 1, 1).expand(-1, 64, 128, 3).clone()
        return frames, {'waveform': torch.zeros(1, 1, 165334), 'sample_rate': 32000}, None, None

    monkeypatch.setattr(runner, 'decode_av_latent', decode)
    return value, sampled_indices, decoded_indices


def test_real_two_segment_delivery_accept_and_compose_recovery(monkeypatch, tmp_path, stub_lifter):  # noqa: F811
    value, sampled, decoded = install(monkeypatch, tmp_path)
    controller = runner.ProgressiveChainDelivery(value)
    original_accept, original_compose = delivery.accept_long_video_candidate, delivery.compose_accepted_long_video
    accept_fail, compose_fail = [True], [True]

    def accept(path, *args):
        descriptor = json.loads(Path(path).read_text())
        if descriptor['index'] == 1 and accept_fail:
            accept_fail.pop()
            raise InterruptedError('after candidate audit before acceptance')
        return original_accept(path, *args)

    def compose(*args, **kwargs):
        if compose_fail:
            compose_fail.pop()
            raise InterruptedError('assembly interrupted')
        return original_compose(*args, **kwargs)

    monkeypatch.setattr(delivery, 'accept_long_video_candidate', accept)
    monkeypatch.setattr(delivery, 'compose_accepted_long_video', compose)
    with pytest.raises(InterruptedError, match='before acceptance'):
        controller.run()
    manifest, _ = delivery.load_delivery_manifest('chain')
    assert len(manifest['segments']) == 1 and sampled == decoded == [0, 1]
    with pytest.raises(InterruptedError, match='assembly interrupted'):
        controller.run()
    assert sampled == decoded == [0, 1]
    manifest, _ = delivery.load_delivery_manifest('chain')
    assert len(manifest['segments']) == 2 and manifest['segments'][-1]['timeline_end_frame'] == 192
    output, text = controller.run()
    report = json.loads(text)
    assert report['current_segments'] == [] and not report['human_qualified']
    assert sampled == decoded == [0, 1] and not report['reused_final']
    with av.open(output) as container:
        assert len(list(container.decode(video=0))) == 192
    with av.open(output) as container:
        assert container.streams.audio[0].sample_rate == 32000
        assert list(container.decode(audio=0))

    def unexpected(*args, **kwargs):
        pytest.fail('Completed delivery repeated sampling/decoding/assembly')

    monkeypatch.setattr(value, 'sample_first', unexpected)
    monkeypatch.setattr(value, 'sample_continuation', unexpected)
    monkeypatch.setattr(runner, 'decode_av_latent', unexpected)
    monkeypatch.setattr(delivery, 'compose_accepted_long_video', unexpected)
    again, text = controller.run()
    assert again == output and json.loads(text)['reused_final']
    assert not value.active
    with pytest.raises(ValueError, match='settings differ'):
        runner.ProgressiveChainDelivery(value, crf=19).run()
    # Retain original bytes after deliberate local-fixture corruption.
    path = Path(output)
    original_bytes = path.read_bytes()
    path.write_bytes(original_bytes + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        controller.run()
    assert path.read_bytes() == original_bytes + b'changed'
    path.write_bytes(original_bytes)
    receipt = controller.root / 'progressive_delivery/segment_00001.json'
    payload = json.loads(receipt.read_text())
    payload['body']['expected']['seed'] += 1
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='receipt integrity'):
        controller.run()


def test_orphan_candidate_is_not_overwritten_or_silently_adopted(monkeypatch, tmp_path, stub_lifter):  # noqa: F811
    value, sampled, decoded = install(monkeypatch, tmp_path)
    controller = runner.ProgressiveChainDelivery(value)
    original = runner._write_receipt
    once = [True]

    def write(path, body):
        if path.name == 'segment_00000.json' and once:
            once.pop()
            raise InterruptedError('power loss before candidate receipt')
        return original(path, body)

    monkeypatch.setattr(runner, '_write_receipt', write)
    with pytest.raises(InterruptedError, match='power loss'):
        controller.run()
    orphans = list((controller.root / 'candidates/segment_00000').glob('*/candidate.json'))
    assert len(orphans) == 1
    old_bytes = orphans[0].read_bytes()
    _, text = controller.run()
    assert sampled == decoded == [0, 0, 1]
    assert json.loads(text)['current_segments'][0]['actual_forwards'] == {'low': 0, 'high': 0}
    assert orphans[0].read_bytes() == old_bytes
    assert len(list(orphans[0].parent.parent.glob('*/candidate.json'))) == 2


def test_reject_wrong_job_type():
    with pytest.raises(ValueError, match='content-bound'):
        runner.ProgressiveChainDelivery(object())


@pytest.mark.parametrize('failure', ['mux', 'final_receipt'])
def test_native_delivery_failure_reuses_sampling_and_retains_audio(monkeypatch, tmp_path, stub_lifter, failure, record_property):  # noqa: F811
    value, sampled, decoded = install(monkeypatch, tmp_path)
    controller = runner.ProgressiveChainDelivery(value)
    original_audio_write = delivery._write_planar_audio_raw
    raw_writes = []

    def audited_audio_write(path, pcm):
        import numpy as np
        assert pcm.shape[0] == 2 and np.array_equal(pcm[0], pcm[1])
        raw_writes.append(pcm.shape[1])
        return original_audio_write(path, pcm)
    monkeypatch.setattr(delivery, '_write_planar_audio_raw', audited_audio_write)
    once = [True]
    if failure == 'mux':
        original = delivery._mux_video_with_raw_audio

        def fail_mux(*args, **kwargs):
            if once:
                once.pop()
                raise OSError('injected native mux failure')
            return original(*args, **kwargs)
        monkeypatch.setattr(delivery, '_mux_video_with_raw_audio', fail_mux)
    else:
        original = runner._write_receipt

        def fail_receipt(path, body):
            if path.name == 'assembled.json' and once:
                once.pop()
                raise OSError('injected final receipt failure')
            return original(path, body)
        monkeypatch.setattr(runner, '_write_receipt', fail_receipt)
    with pytest.raises(OSError, match='injected'):
        controller.run()
    assert not value.active
    assert not (controller.root / 'progressive_delivery/assembled.json').exists()
    if failure == 'mux':
        assert sampled == decoded == [0]
        assert not list((controller.root / 'candidates').rglob('candidate.json'))
    else:
        assert sampled == decoded == [0, 1]
    output, text = controller.run()
    report = json.loads(text)
    if failure == 'mux':
        assert sampled == decoded == [0, 0, 1]
        assert report['current_segments'][0]['actual_forwards'] == {'low': 0, 'high': 0}
    else:
        assert sampled == decoded == [0, 1]
        assert report['current_segments'] == []
    with av.open(output) as container:
        assert len(list(container.decode(video=0))) == 192
    with av.open(output) as container:
        stream = container.streams.audio[0]
        # Existing native delivery explicitly normalizes mono to stereo;
        # preserve that contract rather than silently changing old output.
        assert stream.sample_rate == 32000 and stream.channels == 2
        frames = list(container.decode(audio=0))
        assert frames
        import numpy as np
        pcm = np.concatenate([frame.to_ndarray() for frame in frames], axis=1)
        assert pcm.shape[1] >= 8 * 32000
        assert np.isfinite(pcm).all()
        # AAC is not bit-exact: validate exact channels BEFORE encoding above,
        # and report codec residuals instead of an unjustified peak-error gate.
        record_property('aac_channel_max_difference', float(np.max(np.abs(pcm[0] - pcm[1]))))
        record_property('aac_channel_correlation', float(np.corrcoef(pcm)[0, 1]))
        assert np.corrcoef(pcm)[0, 1] > .99
    assert len(raw_writes) >= 2
    assert not report['human_qualified']


def path_controller(monkeypatch, tmp_path):
    """Only the pure pre-write path guard, not a model or sampling test."""
    monkeypatch.setattr(delivery.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    controller = object.__new__(runner.ProgressiveChainDelivery)
    controller.chain_id = 'paths'
    controller.root = delivery.long_video_chain_root('paths')
    controller.root.mkdir(parents=True)
    controller.settings = {}
    controller.contract = {'job_sha256': 'a' * 64, 'settings': {}}
    controller.contract_sha = runner.digest(controller.contract)
    controller.job = SimpleNamespace(sha256='a' * 64, plan_inputs={'chain_id': 'paths'},
                                     segments=[SimpleNamespace(index=0)])
    return controller


@pytest.mark.parametrize('name', ['progressive_delivery', 'candidates', 'accepted',
                                 'assembled', 'progressive_segments/0'])
def test_redirected_directory_refused_before_job_lock(monkeypatch, tmp_path, name):
    controller = path_controller(monkeypatch, tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = controller.root / name
    link.parent.mkdir(parents=True, exist_ok=True)
    # Windows host supports symlinks in developer/admin mode. A platform that
    # cannot create them must report a skip, never fake a passed path test.
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f'Symlink creation unavailable: {error}')
    with pytest.raises(ValueError, match='escaped'):
        controller.run()
    assert list(outside.iterdir()) == []
    assert not (controller.root / 'progressive_job.json').exists()


def test_mutated_root_refused_before_creating_any_directory(monkeypatch, tmp_path):
    controller = path_controller(monkeypatch, tmp_path)
    controller.root = tmp_path / 'not_the_chain'
    with pytest.raises(ValueError, match='root/settings changed'):
        controller.run()
    assert not controller.root.exists()


@pytest.mark.parametrize('fault', ['none', 'missing_video', 'missing_context', 'unexpected_context'])
def test_accepted_asset_presence_is_checked_before_final_reuse(tmp_path, fault):
    # Unit check of the final-reuse verifier after candidate validation; other
    # tests above exercise real candidate parsing and MP4/context serialization.
    controller = object.__new__(runner.ProgressiveChainDelivery)
    controller.root = tmp_path
    video, context = tmp_path / 'video.bin', tmp_path / 'context.bin'
    video.write_bytes(b'video-content')
    context.write_bytes(b'context-content')
    candidate = dict(candidate_id='fixed', video_path='video.bin', context_path='context.bin',
        video_sha256=delivery._sha256_file(video), context_sha256=delivery._sha256_file(context))
    entry = dict(candidate)
    if fault == 'missing_video':
        entry['video_path'] = ''
    elif fault == 'missing_context':
        entry['context_path'] = ''
    elif fault == 'unexpected_context':
        candidate['context_path'] = ''
    controller._candidate = lambda index, parent: (None, candidate, {})
    controller._expected = lambda index, parent: {}
    if fault == 'none':
        controller._verify_accepted({'segments': [entry]})
    else:
        with pytest.raises(ValueError, match='asset path'):
            controller._verify_accepted({'segments': [entry]})
