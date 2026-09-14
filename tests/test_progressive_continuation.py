"""Real durable-source/MP4 CPU checks; encoder fixture is not a trained VAE."""

import copy
import json
from types import SimpleNamespace

import av
import numpy as np
import pytest
import torch

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg import progressive_continuation as continuation
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.long_video import _motion_context_blocks
from h3_audio_t8_pkg.long_video_dual_stage_cache import AVStageCache


@pytest.fixture
def accepted(tmp_path):
    root = tmp_path / 'chain'
    directory = root / 'candidates/segment_00000/parent'
    directory.mkdir(parents=True)
    media = directory / 'movie.mp4'
    with av.open(str(media), 'w') as output:
        stream = output.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 128, 64, 'yuv420p'
        for index in range(124):
            array = np.full((64, 128, 3), index, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    job = 'a' * 64
    latent, _ = empty_av_latent(128, 64, 124)
    video, audio = latent['samples'].unbind()
    video.copy_(torch.arange(video.numel()).reshape(video.shape) / video.numel())
    audio.copy_(torch.arange(audio.numel()).reshape(audio.shape) / audio.numel())
    context_path = directory / 'context.safetensors'
    context = delivery._write_context_candidate(latent, context_path, 'chain', 0, 'tiny', job)
    info = dict(candidate_id='parent', chain_id='chain', index=0,
        video_path=media.relative_to(root).as_posix(), video_sha256=delivery._sha256_file(media),
        context_path=context_path.relative_to(root).as_posix(), context_sha256=context['sha256'],
        frame_count=124, fps=24, width=128, height=64, sample_rate=32000,
        audio_start_sample=0, audio_end_sample=165333, timeline_start_frame=0,
        timeline_end_frame=124, model_id='tiny', sampling_summary=job, is_final_segment=False)
    descriptor = directory / 'candidate.json'
    descriptor.write_text(json.dumps(info), encoding='utf-8')
    # Accepted delivery uses a distinct copy, as the real durable layer does.
    (root / 'accepted').mkdir()
    accepted_media = root / 'accepted/movie.mp4'
    accepted_media.write_bytes(media.read_bytes())
    accepted_context = root / 'accepted/context.safetensors'
    accepted_context.write_bytes(context_path.read_bytes())
    manifest = delivery._new_manifest('chain')
    manifest.update(revision=1, segments=[{**info, 'video_path': 'accepted/movie.mp4',
                                        'context_path': 'accepted/context.safetensors'}])
    manifest_path = root / delivery.MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    request = dict(chain_id='chain', segment_index=1, parent_candidate_id='parent', parent_revision=1,
        job_sha256=job, context_frames=22, width=128, height=64, low_width=64, low_height=32)
    return SimpleNamespace(root=root, descriptor=descriptor, manifest_path=manifest_path,
        manifest=manifest, info=info, media=media, accepted_media=accepted_media,
        accepted_context=accepted_context, request=request)


def capture(case, **overrides):
    return continuation.capture_continuation_source(case.root, **{**case.request, **overrides})


def test_binding_is_detached_and_repeat_read_has_exact_identity(accepted):
    source = capture(accepted)
    assert source.sha256 == capture(accepted).sha256
    source.binding['request']['context_frames'] = 39
    assert source.binding['request']['context_frames'] == 22
    assert source.revalidate() == source.binding
    assert source.binding['picture']['source_frame_interval'] == [85, 124]


def test_same_filename_new_accepted_video_changes_real_binding_and_cache(accepted):
    original = capture(accepted)
    cache = AVStageCache(accepted.root / 'replacement-cache')
    latent, _ = empty_av_latent(128, 64, 124)
    cache.save('low_x0', original.binding, latent, {})
    with av.open(str(accepted.media), 'w') as output:
        stream = output.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 128, 64, 'yuv420p'
        for index in range(124):
            array = np.full((64, 128, 3), 255-index, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    accepted.accepted_media.write_bytes(accepted.media.read_bytes())
    with pytest.raises(ValueError):
        original.revalidate()
    sha = delivery._sha256_file(accepted.media)
    accepted.info['video_sha256'] = sha
    accepted.manifest['segments'][0]['video_sha256'] = sha
    accepted.descriptor.write_text(json.dumps(accepted.info), encoding='utf8')
    accepted.manifest_path.write_text(json.dumps(accepted.manifest), encoding='utf8')
    changed = capture(accepted)
    assert changed.sha256 != original.sha256
    assert changed.binding['picture']['source_frame_interval'] == [85, 124]
    assert cache.load('low_x0', changed.binding) is None
    with pytest.raises(ValueError):
        original.revalidate()


@pytest.mark.parametrize('frames,steps', [(5, 2), (22, 7), (39, 12)])
def test_prepare_uses_real_rgb24_decode_and_existing_resize_only_for_low(accepted, monkeypatch, frames, steps):
    from h3_audio_t8_pkg import core
    source = capture(accepted, context_frames=frames)
    original_resize = core.resize_image
    calls = []

    def resize(images, width, height):
        assert images.shape == (39, 64, 128, 3)
        assert (width, height) == (64, 32)
        with av.open(str(accepted.accepted_media)) as container:
            expected = [torch.from_numpy(f.to_ndarray(format='rgb24')) for f in container.decode(video=0)]
        assert torch.equal(images, torch.stack(expected[-39:]).float() / 255)
        calls.append('rgb_resize')
        return original_resize(images, width, height)

    def encode(images):
        assert calls == ['rgb_resize'] and images.shape == (39, 32, 64, 3)
        calls.append('vae_encode_fixture')
        return torch.full((1, 24, 12, 2, 4), .25)

    monkeypatch.setattr(core, 'resize_image', resize)
    low, high, report = source.prepare_contexts(SimpleNamespace(encode=encode))
    assert calls == ['rgb_resize', 'vae_encode_fixture']
    assert torch.all(low['video_tail'] == .25)
    assert high['video_tail'].shape == (1, 24, 12, 4, 8)
    assert low['audio_tail'] is high['audio_tail']
    assert low['metadata']['audio_overhang'] == high['metadata']['audio_overhang']
    assert report['additional_sampling_nfe'] == 0 and not report['sampling_and_resume_qualified']
    assert report['context_steps'] == steps and report['tail_capacity_frames'] == 39
    low_guides, low_audio = _motion_context_blocks(low, frames, True)
    high_guides, high_audio = _motion_context_blocks(high, frames, True)
    assert len(low_guides) == len(high_guides) == steps
    assert torch.equal(low_audio['audio_latent'], high_audio['audio_latent'])
    assert [f['t8_long_video_frame_index'] for f in low_guides] == [f['t8_long_video_frame_index'] for f in high_guides]


@pytest.mark.parametrize('fault', ['revision', 'selected', 'final', 'job', 'dimensions', 'fps',
    'candidate_hash', 'accepted_bytes', 'context_bytes', 'context_path', 'media_path', 'descriptor_path'])
def test_bad_parent_rejected_before_decode_or_vae(accepted, monkeypatch, fault):
    monkeypatch.setattr(continuation.picture, 'prepare_context', lambda *a: pytest.fail('must not prepare'))
    entry = accepted.manifest['segments'][0]
    if fault == 'revision':
        accepted.manifest['revision'] = 2
    elif fault == 'selected':
        entry['candidate_id'] = 'other'
    elif fault == 'final':
        entry['is_final_segment'] = True
    elif fault == 'job':
        entry['sampling_summary'] = 'b' * 64
    elif fault == 'dimensions':
        entry['width'] = 256
    elif fault == 'fps':
        entry['fps'] = 25
    elif fault == 'candidate_hash':
        accepted.info['video_sha256'] = '0' * 64
        accepted.descriptor.write_text(json.dumps(accepted.info))
    elif fault == 'accepted_bytes':
        accepted.accepted_media.write_bytes(b'changed')
    elif fault == 'context_bytes':
        accepted.accepted_context.write_bytes(b'changed')
    elif fault == 'context_path':
        entry['context_path'] = '../outside.safetensors'
    elif fault == 'media_path':
        entry['video_path'] = '../outside.mp4'
    elif fault == 'descriptor_path':
        accepted.info['video_path'] = '../outside.mp4'
        accepted.descriptor.write_text(json.dumps(accepted.info))
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        capture(accepted)


@pytest.mark.parametrize('overrides', [dict(segment_index=0), dict(segment_index=True),
    dict(parent_candidate_id='../parent'), dict(parent_revision=True), dict(context_frames=23),
    dict(context_frames=True), dict(low_width=65), dict(low_height=64), dict(job_sha256='filename'),
    dict(chain_id='../chain')])
def test_invalid_requests_fail_before_io(accepted, overrides):
    with pytest.raises(ValueError):
        capture(accepted, **overrides)


@pytest.mark.parametrize('field', ['context_frames', 'job_sha256', 'parent_revision'])
def test_source_binding_changes_stage_cache_identity(accepted, field):
    original = capture(accepted)
    request = copy.deepcopy(accepted.request)
    if field == 'context_frames':
        request[field] = 39
    elif field == 'parent_revision':
        request[field] = 2
        accepted.manifest['revision'] = 2
        accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    else:
        # A foreign job may not reuse this accepted parent, even if filenames match.
        with pytest.raises(ValueError, match='execution contract'):
            capture(accepted, job_sha256='b' * 64)
        return
    changed = continuation.capture_continuation_source(accepted.root, **request)
    cache = AVStageCache(accepted.root / 'progressive_stages')
    assert cache._key('high_output', original.binding) != cache._key('high_output', changed.binding)


def test_changed_manifest_during_encode_is_rejected(accepted):
    source = capture(accepted)

    def encode(images):
        accepted.manifest['revision'] = 2
        accepted.manifest_path.write_text(json.dumps(accepted.manifest))
        return torch.ones(1, 24, 12, 2, 4)

    with pytest.raises(ValueError, match='revision'):
        source.prepare_contexts(SimpleNamespace(encode=encode))


@pytest.mark.parametrize('fault', ['audio_nan', 'video_nan', 'overhang', 'foreign_job', 'short_audio'])
def test_self_consistent_but_invalid_context_is_rejected(accepted, fault):
    from safetensors import safe_open
    from safetensors.torch import save_file
    with safe_open(str(accepted.accepted_context), framework='pt') as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    if fault.endswith('_nan'):
        name = 'audio_tail' if fault == 'audio_nan' else 'video_tail'
        tensors[name].flatten()[0] = float('nan')
        metadata[name.replace('_tail', '_sha256')] = delivery._tensor_sha256(tensors[name])
    elif fault == 'overhang':
        metadata['audio_overhang'] = 'nan'
    elif fault == 'short_audio':
        tensors['audio_tail'] = tensors['audio_tail'][..., :1].contiguous()
        metadata['audio_shape'] = json.dumps(list(tensors['audio_tail'].shape))
        metadata['audio_sha256'] = delivery._tensor_sha256(tensors['audio_tail'])
    else:
        metadata['sampling_summary'] = 'b' * 64
    save_file(tensors, str(accepted.accepted_context), metadata)
    digest = delivery._sha256_file(accepted.accepted_context)
    accepted.manifest['segments'][0]['context_sha256'] = digest
    (accepted.root / accepted.info['context_path']).write_bytes(accepted.accepted_context.read_bytes())
    accepted.info['context_sha256'] = digest
    accepted.descriptor.write_text(json.dumps(accepted.info))
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError):
        capture(accepted)


def test_bad_encoder_output_and_cancellation_do_not_mutate_durable_parent(accepted):
    source = capture(accepted)
    before = accepted.accepted_context.read_bytes()
    with pytest.raises(ValueError, match='geometry'):
        source.prepare_contexts(SimpleNamespace(encode=lambda x: torch.zeros(1, 24, 7, 2, 4)))

    def cancel(images):
        raise InterruptedError('cancel fixture')

    with pytest.raises(InterruptedError):
        source.prepare_contexts(SimpleNamespace(encode=cancel))
    assert accepted.accepted_context.read_bytes() == before
    assert source.revalidate() == source.binding


@pytest.mark.parametrize('frames,steps', [(22, 7), (39, 12)])
@pytest.mark.parametrize('context_audio', ['video_only', 'video_and_audio'])
@pytest.mark.parametrize('audio_mode', ['native', 'lock_source', 'remix_source'])
def test_existing_condition_builder_and_native_high_prefix_are_preserved(
        accepted, frames, steps, context_audio, audio_mode):
    from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE, make_audio
    from h3_audio_t8_pkg.long_video import build_long_video_conditioning, repair_long_video_layout
    from h3_audio_t8_pkg.conditioning import build_packed_layout
    source = capture(accepted, context_frames=frames)
    options = dict(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt='Continue the action.', length=124, context_audio=context_audio, audio_mode=audio_mode,
        drive_audio=None if audio_mode == 'native' else make_audio(6), add_source_as_reference=False)
    low, high, report = source.prepare_conditions(**options)
    low_context, high_context, _ = source.prepare_contexts(options['video_vae'])
    common = {**options, 'segment_index': 1, 'context_frames': frames, 'return_details': True}
    expected_low = build_long_video_conditioning(context=low_context, width=64, height=32, **common)
    expected_high = build_long_video_conditioning(context=high_context, width=128, height=64, **common)
    for actual, expected in ((low, expected_low), (high, expected_high)):
        assert torch.equal(actual[0][0][0], expected[0][0][0])
        assert actual[5] == expected[5]  # complete native builder report, not a new formula
        details = actual[-1]
        assert len(details['keyframes']) == steps
        for left, right in zip(details['keyframes'], expected[-1]['keyframes'], strict=True):
            assert torch.equal(left['latent'], right['latent'])
        assert actual[2] is expected[2]  # explicit delivery-audio object policy
        assert torch.equal(actual[1]['samples'].unbind()[1], expected[1]['samples'].unbind()[1])
    assert not report['long_video_model_owner_installed']
    assert report['high_prefix']['context_steps'] == steps
    hv, ha = high[1]['samples'].unbind()
    hm, ham = high[1]['noise_mask'].unbind()
    assert torch.equal(hv[:, :, :steps], high_context['video_tail'][:, :, -steps:])
    assert torch.count_nonzero(hv[:, :, steps:]) == 0
    assert torch.all(hm[:, :, :steps] == 0) and torch.all(hm[:, :, steps:] == 1)
    if 'noise_mask' in expected_high[1]:
        assert torch.equal(ham, expected_high[1]['noise_mask'].unbind()[1])
    else:
        assert torch.all(ham == 1)
    assert torch.equal(ha, low[1]['samples'].unbind()[1])
    # Actual Core PackedLayout + existing motion-time repair at both grids.
    positions = []
    for stage in (low, high):
        details = stage[-1]
        video, audio = stage[1]['samples'].unbind()
        layout = build_packed_layout(4, video.shape[2], video.shape[3], video.shape[4],
            audio.shape[-1], keyframes=details['keyframes'], refs=details['refs'], frame_count=124)
        repair_long_video_layout(layout, details['keyframes'], details['refs'], 124)
        positions.append([float(layout.position_ids[start, 0]) for start, _, kind in layout.segments
                          if kind in {'cond', 'ref_audio', 'audio', 'video'}])
    assert positions[0] == positions[1]


def test_stage_option_override_is_rejected_before_preparation(accepted, monkeypatch):
    monkeypatch.setattr(continuation.picture, 'prepare_context', lambda *a: pytest.fail('must not prepare'))
    with pytest.raises(ValueError, match='override'):
        capture(accepted).prepare_conditions(clip=None, video_vae=None, audio_vae=None,
            prompt='test', length=124, segment_index=0)
