from pathlib import Path
import json
import shutil
import subprocess

import av
import pytest
import torch

from h3_audio_t8_pkg.dance_audio_delivery import prepare_source_audio, finalize_source_audio, video_packet_identity


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
