from pathlib import Path
import json
import shutil
import subprocess

import av
import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.dance_audio_delivery import prepare_source_audio, finalize_source_audio, video_packet_identity
from h3_audio_t8_pkg import dance_audio_delivery as delivery


@pytest.mark.parametrize('rate', [24000, 32000, 44100, 48000])
def test_whole_audio_exact_prefix_without_changes(rate):
    wave = torch.linspace(-.5, .5, rate * 2).reshape(1, 1, -1)
    before = wave.clone()
    pcm, report = prepare_source_audio({'waveform': wave, 'sample_rate': rate}, 24)
    assert report['samples'] == rate
    assert torch.equal(torch.from_numpy(pcm[:, 0]), wave[0, 0, :rate])
    assert torch.equal(wave, before)


@pytest.mark.parametrize('kind', ['short', 'nonfinite', 'channels', 'rate'])
def test_audio_does_not_silently_pad_or_downmix(kind):
    audio = {'waveform': torch.zeros(1, 2, 32000), 'sample_rate': 32000}
    if kind == 'short':
        audio['waveform'] = audio['waveform'][:, :, :-1]
    elif kind == 'nonfinite':
        audio['waveform'][0, 0, 4] = float('nan')
    elif kind == 'channels':
        audio['waveform'] = torch.zeros(1, 6, 32000)
    else:
        audio['sample_rate'] = 12345
    with pytest.raises(ValueError):
        prepare_source_audio(audio, 24)


def test_whole_audio_real_mux_packet_identity_and_cached_recovery(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('FFmpeg not installed')
    source = tmp_path / 'source.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i', 'color=c=red:s=128x128:r=24:d=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True, capture_output=True)
    wave = .2 * torch.sin(torch.arange(32000) * (2 * torch.pi * 440 / 32000))
    audio = {'waveform': wave.reshape(1, 1, -1), 'sample_rate': 32000}
    original = source.read_bytes()
    output, report = finalize_source_audio(source, audio, 24)
    assert source.read_bytes() == original
    assert video_packet_identity(output) == video_packet_identity(source)
    assert report['audio_duration_seconds'] == pytest.approx(1, abs=1/32000)
    with av.open(output) as container:
        assert container.streams.audio[0].codec_context.channels == 1
    assert finalize_source_audio(output, audio, 24, cached_report=report) == (output, report)
    # Simulate process stopping after the output and sidecar, before state update.
    assert finalize_source_audio(source, audio, 24) == (output, report)
    saved = json.loads(Path(output).with_suffix('.audio.json').read_text())
    assert saved['identity'] == report['identity']
    assert not list(tmp_path.glob('.dance-audio-*'))


@pytest.fixture
def stereo_case(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('FFmpeg not installed')
    source = tmp_path / 'stereo-source.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-n', '-f', 'lavfi', '-i',
                    'color=c=blue:s=128x128:r=24:d=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)],
                   check=True, capture_output=True)
    time = torch.arange(32000) / 32000
    # Distinct channels and time-local markers, not a repeated stationary tone.
    channels = []
    for centers, hz in [([.12, .44, .79], 431), ([.22, .59, .91], 743)]:
        envelope = sum(torch.exp(-((time-center)/.025)**2) for center in centers)
        channels.append(.2 * envelope * torch.sin(2 * torch.pi * (hz*time + 80*time*time)))
    return source, {'waveform': torch.stack(channels).unsqueeze(0), 'sample_rate': 32000}


@pytest.mark.parametrize('failure', ['mux', 'receipt_replace', 'output_replace'])
def test_failed_delivery_preserves_source_and_recovers_without_generation(stereo_case, monkeypatch, failure):
    source, audio = stereo_case
    before = source.read_bytes()
    original_run, original_replace = delivery._run_isolated_ffmpeg, delivery.os.replace
    seen = []

    def run(*args, **kwargs):
        seen.append('mux')
        if failure == 'mux':
            raise RuntimeError('injected mux failure')
        return original_run(*args, **kwargs)

    def replace(src, dst):
        if ((failure == 'receipt_replace' and str(dst).endswith('.audio.json'))
                or (failure == 'output_replace' and str(dst).endswith('.mp4'))):
            raise OSError('injected delivery rename failure')
        return original_replace(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(delivery, '_run_isolated_ffmpeg', run)
        patch.setattr(delivery.os, 'replace', replace)
        with pytest.raises((RuntimeError, OSError), match='injected'):
            finalize_source_audio(source, audio, 24)
    assert seen == ['mux']
    assert source.read_bytes() == before
    assert not list(source.parent.glob('dance_music_*.mp4'))
    assert not list(source.parent.glob('.dance-audio-*'))
    output, report = finalize_source_audio(source, audio, 24)
    assert video_packet_identity(output) == video_packet_identity(source)
    assert report['channels'] == 2 and not report['aac_bitexact']
    # Successful retry becomes a cache hit; even FFmpeg must not run again.
    with monkeypatch.context() as patch:
        patch.setattr(delivery, '_run_isolated_ffmpeg', lambda *a, **k: pytest.fail('cached mux rerun'))
        assert finalize_source_audio(source, audio, 24) == (output, report)


def test_real_stereo_markers_keep_channels_and_front_middle_tail_timing(stereo_case):
    source, audio = stereo_case
    output, report = finalize_source_audio(source, audio, 24)
    with av.open(output) as container:
        stream = container.streams.audio[0]
        assert stream.codec_context.channels == 2 and stream.codec_context.sample_rate == 32000
        decoded = np.concatenate([frame.to_ndarray() for frame in container.decode(audio=0)], axis=1)
    expected = audio['waveform'][0].numpy()
    assert decoded.shape[1] >= report['samples']
    # AAC may expose encoder tail padding in raw decode; stream duration, not
    # raw padded frame count, defines playback. Compare only intended samples.
    for channel in range(2):
        for start, end in [(0, 10666), (10666, 21333), (21333, 32000)]:
            a, b = expected[channel, start:end], decoded[channel, start:end]
            assert np.corrcoef(a, b)[0, 1] > .94
            wrong = decoded[1-channel, start:end]
            assert abs(np.corrcoef(a, wrong)[0, 1]) < .3
            # Envelope center must stay within 1ms at all three time regions.
            axis = np.arange(end-start)
            ca = np.sum(axis*a*a) / np.sum(a*a)
            cb = np.sum(axis*b*b) / np.sum(b*b)
            assert abs(ca-cb) < 32
