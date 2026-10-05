"""Real decoded media/native builder/tiny H3; trained encoders/lifter are doubles."""
import copy
import json
import sys
import wave
from pathlib import Path

import av
import comfy.nested_tensor
import comfy.samplers
import numpy as np
import pytest
import torch
from h3_audio_t8_pkg import external_rgb_pcm_context as external
from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE
from test_progressive_sampling_runtime import stub_lifter, tiny_model  # noqa: F401


def write_video(path, fps=24):
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = 128, 64, 'yuv420p'
        stream.codec_context.options = {'crf': '18', 'threads': '1'}
        for index in range(80):
            frame = av.VideoFrame.from_ndarray(np.full((64, 128, 3), index, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def write_audio(path, channels=2, samples=106667):
    values = np.arange(samples, dtype=np.int64) % 25000
    data = np.stack([values + index for index in range(channels)], axis=1).astype('<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(32000)
        output.writeframes(data.tobytes())
    return data


@pytest.fixture
def media(tmp_path):
    write_video(tmp_path / 'source.mp4')
    values = write_audio(tmp_path / 'original.wav')
    return tmp_path, values


def capture(media, cut=52, count=22, policy='reencode_stereo_pcm_context'):
    root, _ = media
    return external.capture_external_rgb_pcm_source(root, 'source.mp4', external._file_sha(root / 'source.mp4'),
        start_frame=0, end_frame=cut, context_frames=count, audio_policy=policy,
        audio_relative_path='original.wav', audio_sha256=external._file_sha(root / 'original.wav'),
        source_binding_json=json.dumps({'branch_id': 'branch', 'selected_candidate_id': 'delivery-version'}))


def contexts(source, video=None, audio=None):
    video, audio = video or FakeVideoVAE(), audio or FakeAudioVAE()
    low, high, report = external.prepare_external_rgb_pcm_contexts(source, video, audio,
        low_width=64, low_height=32, width=128, height=64,
        context_audio='video_only' if source.binding['audio_policy'] == 'video_only' else 'video_and_audio')
    return low, high, report, video, audio


@pytest.mark.parametrize('count', [5, 22, 39])
@pytest.mark.parametrize('cut', [51, 52, 53])
def test_absolute_audio_endpoints_and_exact_selected_rgb_tail(media, count, cut):
    source = capture(media, cut=cut, count=count)
    low, high, report, video, audio = contexts(source)
    start, end = round((cut - count) * 32000 / 24), round(cut * 32000 / 24)
    binding = source.binding
    assert binding['tail_frame_interval'] == [cut - count, cut]
    assert binding['decoded_clock']['audio']['tail_sample_interval'] == [start, end]
    encoded = audio.encode_calls[0].movedim(-1, 1)
    expected = torch.from_numpy(media[1][start:end].T.astype(np.float32) / 32768).unsqueeze(0)
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    assert len(audio.encode_calls) == 1
    assert low['audio_tail'] is high['audio_tail']
    with av.open(str(media[0] / 'source.mp4')) as container:
        actual = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)][cut - count:cut]
    expected_rgb = torch.from_numpy(np.stack(actual)).float() / 255
    torch.testing.assert_close(video.encode_calls[1], expected_rgb, rtol=0, atol=0)
    assert low['video_tail'].shape == (1, 24, {5: 2, 22: 7, 39: 12}[count], 2, 4)
    assert high['video_tail'].shape[-2:] == (4, 8)
    assert report['audio_tensor_shared_once_encoded']
    assert report['additional_sampling_nfe'] == 0
    assert not report['original_native_latent_available']
    assert not report['has_sampling_ancestor']
    assert report['gpu_acceptance'] == 'NOT_RUN'
    assert 'chain_id' not in high['metadata'] and 'accepted_candidate_id' not in high['metadata']
    assert high['metadata']['conditioning_coordinate_only']
    assert high['metadata']['origin'] == 'external_rgb_pcm_reencoded'


def test_endpoint_policy_keeps_source_phase_difference():
    left, right = (external.rounded_audio_frame_sample(frame, 32000) for frame in (30, 52))
    assert right - left == 29333
    left, right = (external.rounded_audio_frame_sample(frame, 32000) for frame in (31, 53))
    assert right - left == 29334


def test_original_float_wav_preserves_exact_stereo_pcm(media):
    root, values = media
    waveform = np.ascontiguousarray(values.T.astype(np.float32) / 32768)
    with av.open(str(root / 'original.wav'), 'w') as output:
        stream = output.add_stream('pcm_f32le', rate=32000)
        stream.layout = 'stereo'
        frame = av.AudioFrame.from_ndarray(waveform, format='fltp', layout='stereo')
        frame.sample_rate = 32000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    source = capture(media, cut=53)
    _, _, _, _, audio = contexts(source)
    left, right = source.binding['decoded_clock']['audio']['tail_sample_interval']
    expected = torch.from_numpy(waveform[:, left:right]).unsqueeze(0)
    torch.testing.assert_close(audio.encode_calls[0].movedim(-1, 1), expected, rtol=0, atol=0)


def test_video_only_does_not_encode_or_silently_use_original_audio(media):
    source = capture(media, policy='video_only')
    low, high, report, _, audio = contexts(source)
    assert not audio.encode_calls
    assert not bool(torch.count_nonzero(high['audio_tail']))
    assert high['metadata']['audio_context_source'] == 'unused_placeholder'
    assert high['metadata']['audio_context_sample_interval'] is None
    assert low['audio_tail'] is high['audio_tail']
    assert report['high']['audio_vae'] is None


@pytest.mark.parametrize('fault', ['video', 'audio', 'caller'])
def test_bound_source_rejects_media_and_caller_mutation(media, fault):
    source = capture(media)
    if fault == 'caller':
        with pytest.raises(ValueError, match='binding changed'):
            source.verify(source_binding_json='{"branch_id":"other"}')
    else:
        target = media[0] / ('source.mp4' if fault == 'video' else 'original.wav')
        with target.open('ab') as output:
            output.write(b'changed')
        with pytest.raises(ValueError, match='media bytes changed'):
            source.verify()


@pytest.mark.parametrize('fault', ['video_tensor', 'audio_tensor', 'geometry', 'origin',
                                 'schema', 'empty', 'encode', 'rate', 'replacement'])
def test_prepared_source_rejects_context_and_encoder_mutation(media, fault):
    low, high, _, video, audio = contexts(capture(media))
    if fault == 'video_tensor':
        with torch.inference_mode():
            high['video_tail'][0, 0, 0, 0, 0] += 1
    elif fault == 'audio_tensor':
        with torch.inference_mode():
            low['audio_tail'][0, 0, 0, 0] += 1
    elif fault == 'geometry':
        high['metadata']['width'] = 256
    elif fault == 'origin':
        high['origin'] = 'accepted_native_context'
    elif fault == 'schema':
        high['schema'] = 99
    elif fault == 'empty':
        high['empty'] = True
    elif fault == 'encode':
        video.encode = lambda value: value
    elif fault == 'rate':
        audio.audio_sample_rate = 44100
    else:
        with pytest.raises(ValueError, match='another video VAE'):
            external.validate_external_rgb_pcm_context(high, video_vae=FakeVideoVAE())
        return
    with pytest.raises(ValueError, match='changed'):
        external.validate_external_rgb_pcm_context(high)


@pytest.mark.parametrize('fault', ['sha', 'outside', 'short', 'fps', 'mono', 'audio_short'])
def test_bad_media_contract_fails_before_any_encoder(media, fault):
    root, _ = media
    kwargs = {'start_frame': 0, 'end_frame': 52, 'context_frames': 22,
              'audio_relative_path': 'original.wav', 'audio_sha256': external._file_sha(root / 'original.wav')}
    relative, checksum = 'source.mp4', external._file_sha(root / 'source.mp4')
    if fault == 'sha':
        checksum = '0' * 64
    elif fault == 'outside':
        relative = '../source.mp4'
    elif fault == 'short':
        kwargs['end_frame'] = 10
    elif fault == 'fps':
        write_video(root / 'source.mp4', fps=25)
        checksum = external._file_sha(root / 'source.mp4')
    elif fault == 'mono':
        write_audio(root / 'original.wav', channels=1)
        kwargs['audio_sha256'] = external._file_sha(root / 'original.wav')
    else:
        write_audio(root / 'original.wav', samples=500)
        kwargs['audio_sha256'] = external._file_sha(root / 'original.wav')
    with pytest.raises(ValueError):
        external.capture_external_rgb_pcm_source(root, relative, checksum, **kwargs)


def test_native_builder_and_high_prefix_preserve_source_audio_and_masks(media):
    low, high, _, video_vae, audio_vae = contexts(capture(media))
    built = long_video.build_long_video_conditioning(FakeClip(), video_vae, audio_vae, high,
        1, 22, 'video_and_audio', 'Continue the action.', 128, 64, 124, return_details=True)
    video, audio = built[1]['samples'].unbind()
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :3] = .25
    video_mask = torch.ones_like(video)
    video_mask[:, :, -1] = .4
    latent = {**built[1], 'noise_mask': comfy.nested_tensor.NestedTensor((video_mask, audio_mask))}
    original_video = video.clone()
    output, report = external.apply_external_high_prefix(latent, high)
    ov, oa = output['samples'].unbind()
    vm, am = output['noise_mask'].unbind()
    assert oa is audio and am is audio_mask
    assert torch.equal(video, original_video)
    assert torch.equal(ov[:, :, :7], high['video_tail'])
    assert torch.equal(ov[:, :, 7:], video[:, :, 7:])
    assert torch.all(vm[:, :, :7] == 0)
    assert torch.equal(vm[:, :, 7:], video_mask[:, :, 7:])
    assert report['origin'] == 'external_rgb_pcm_reencoded'
    assert report['audio_touched'] is False
    external.validate_external_rgb_pcm_context(low)
    external.validate_external_rgb_pcm_context(high)


def test_explicit_caller_redraw_mask_is_preserved_and_authenticated(media):
    _, high, _, video_vae, audio_vae = contexts(capture(media))
    built = long_video.build_long_video_conditioning(FakeClip(), video_vae, audio_vae, high,
        1, 22, 'video_and_audio', 'Continue the action.', 128, 64, 124)
    locked, _ = external.apply_external_high_prefix(built[1], high)
    locked['noise_mask'].unbind()[0][:, :, :7] = .65
    before = locked['noise_mask'].unbind()[0].clone()
    result, report = external.apply_external_high_prefix(locked, high, preserve_existing_mask=True)
    assert result is locked
    assert torch.equal(result['noise_mask'].unbind()[0], before)
    assert report['mode'] == 'preserved_caller_external_prefix_mask'
    with pytest.raises(ValueError, match='already owned'):
        external.apply_external_high_prefix(locked, high)
    locked['samples'].unbind()[0][:, :, 0] += 1
    with pytest.raises(ValueError, match='exact reencoded prefix'):
        external.apply_external_high_prefix(locked, high, preserve_existing_mask=True)


def test_external_pair_executes_real_tiny_core_euler_without_native_ancestor(media, stub_lifter):  # noqa: F811
    low_context, high_context, _, video_vae, audio_vae = contexts(capture(media))
    low = long_video.build_long_video_conditioning(FakeClip(), video_vae, audio_vae, low_context,
        1, 22, 'video_and_audio', 'Continue the action.', 64, 32, 124)
    high = list(long_video.build_long_video_conditioning(FakeClip(), video_vae, audio_vae, high_context,
        1, 22, 'video_and_audio', 'Continue the action.', 128, 64, 124))
    high[1], _ = external.apply_external_high_prefix(high[1], high_context)
    models = [long_video.patch_long_video_model(tiny_model()) for _ in range(2)]
    before = [copy.deepcopy(model.model_options) for model in models]
    output, text = runtime.sample_progressive_h3(models[0], high[0], high[0], high[1],
        comfy.samplers.ksampler('euler'), native_flow_sigmas(4, 12.),
        upscaler_model='test', seed=9, model_hires=models[1], low_evaluations=2, task='t2va',
        input_mode='prepared_pair_exp', av_latent_low=low[1], positive_low=low[0], negative_low=low[0])
    report = json.loads(text)
    assert report['counts']['actual_forwards'] == {'low': 2, 'high': 2}
    assert len(stub_lifter) == 1
    assert all(bool(torch.isfinite(value).all()) for value in output['samples'].unbind())
    torch.testing.assert_close(output['samples'].unbind()[0][:, :, :7], high_context['video_tail'], rtol=0, atol=2e-6)
    assert all(model.model_options == original for model, original in zip(models, before))
    external.validate_external_rgb_pcm_context(low_context)
    external.validate_external_rgb_pcm_context(high_context)


def test_real_miniature_core_vae_encoders_and_weight_binding(media, monkeypatch):
    """Actual native encoder math, reduced random networks; no trained weights."""
    import comfy.model_patcher
    import comfy.sd
    from comfy.ldm.minimax import vae as video_module
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE

    decoder = video_module.ViT3DDecoder
    # The decoder is not executed here. Construct its actual native class at
    # reduced size to avoid allocating the production 36-layer architecture.
    with monkeypatch.context() as limited:
        limited.setattr(video_module, 'ViT3DDecoder', lambda **kwargs:
            decoder(**kwargs, num_layers=1, heads=2, dim_head=16))
        video_network = video_module.MiniMaxH3VideoVAE(ch=32, ch_mult=(1,) * 6,
            num_res_blocks=0, tiling=False)
    audio_network = MiniMaxH3AudioVAE(encoder_dim=2, latent_dim=32, decoder_dim=128)
    generator = torch.Generator().manual_seed(941)
    for network in (video_network, audio_network):
        for parameter in network.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .02)
        # Core's checkpoint-facing constructors deliberately leave some
        # buffers empty, too. Initialize every floating buffer explicitly.
        for name, buffer in network.named_buffers():
            if buffer.is_floating_point():
                buffer.fill_(1. if 'std' in name else 0.)
        network.eval()

    wrappers = []
    for network, channels, dims in ((video_network, 24, 3), (audio_network, 32, 2)):
        # Initialize generic public wrapper defaults, then attach the actual
        # reduced native network. Public VAE.encode performs both forwards.
        wrapper = comfy.sd.VAE(sd={})
        wrapper.first_stage_model = network
        wrapper.latent_channels, wrapper.latent_dim = channels, dims
        wrapper.output_channels = 3 if dims == 3 else 2
        wrapper.crop_input = False
        wrapper.vae_dtype = torch.float32
        wrapper.device = wrapper.output_device = torch.device('cpu')
        wrapper.memory_used_encode = lambda shape, dtype: 1
        if dims == 2:
            wrapper.audio_sample_rate = 32000
            wrapper.process_input = lambda waveform: waveform
        wrapper.patcher = comfy.model_patcher.ModelPatcher(network,
            torch.device('cpu'), torch.device('cpu'))
        wrappers.append(wrapper)

    low, high, report = external.prepare_external_rgb_pcm_contexts(capture(media), *wrappers,
        low_width=64, low_height=32, width=128, height=64)
    assert low['audio_tail'] is high['audio_tail']
    assert all(bool(torch.isfinite(context['video_tail']).all()) for context in (low, high))
    assert bool(torch.isfinite(high['audio_tail']).all())
    for key in ('video_vae', 'audio_vae'):
        assert report['high'][key]['scope'] == 'actual_native_weights_tokenizer_configuration_and_implementation'
    external.validate_external_rgb_pcm_context(high)
    with torch.no_grad():
        next(video_network.parameters()).add_(.01)
    with pytest.raises(ValueError, match='VAE changed'):
        external.validate_external_rgb_pcm_context(high)


def test_alternate_public_vae_is_advisory_and_still_binds_its_weights(media, caplog):
    import comfy.model_patcher
    import comfy.sd

    wrapper = comfy.sd.VAE(sd={})
    wrapper.first_stage_model = torch.nn.Linear(1, 1)
    wrapper.patcher = comfy.model_patcher.ModelPatcher(wrapper.first_stage_model,
        torch.device('cpu'), torch.device('cpu'))
    wrapper.encode = FakeVideoVAE().encode
    _, high, report = external.prepare_external_rgb_pcm_contexts(capture(media, policy='video_only'),
        wrapper, FakeAudioVAE(), low_width=64, low_height=32, width=128, height=64,
        context_audio='video_only')
    assert report['high']['video_vae']['portable_cache_reuse'] is False
    assert 'compatibility advisory' in caplog.text
    external.validate_external_rgb_pcm_context(high)
    with torch.no_grad():
        wrapper.first_stage_model.weight.add_(.01)
    with pytest.raises(ValueError, match='VAE changed'):
        external.validate_external_rgb_pcm_context(high)


def test_runtime_helper_is_loaded_for_installed_adapters():
    assert sys.modules['h3_audio_t8_pkg.external_rgb_pcm_context'] is external
    assert Path(external.__file__).name == 'external_rgb_pcm_context.py'
