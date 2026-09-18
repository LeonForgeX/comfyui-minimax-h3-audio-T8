from fractions import Fraction
import hashlib
import json

import av
import numpy as np
import pytest

from h3_audio_t8_pkg.opening_fade_file import mute_fade_file_video
from h3_audio_t8_pkg.opening_fade_file_worker import boundary, process_file, video_signature


def make_media(path, *, timestamps=None, audio=True, offset=0, frames=12, chirp=False):
    pts = timestamps or [i * 40 for i in range(frames)]
    with av.open(str(path), 'w') as container:
        video = container.add_stream('libx264', rate=25)
        video.width = video.height = 32
        video.pix_fmt = 'yuv420p'
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        video.options = {'preset': 'ultrafast', 'bf': '0', 'threads': '1'}
        if audio:
            sound = container.add_stream('aac', rate=32000)
            sound.layout = 'stereo'
        for i, tick in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, np.uint8), format='rgb24')
            frame.pts, frame.time_base = tick, Fraction(1, 1000)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        if audio:
            count = 12800
            ticks = np.arange(count)
            signal = (np.sin(ticks * .08 + (.000015 * ticks**2 if chirp else 0)) * .3).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(np.stack([signal, signal]), format='fltp', layout='stereo')
            frame.sample_rate, frame.pts, frame.time_base = 32000, offset, Fraction(1, 32000)
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode(None):
                container.mux(packet)
    return pts


def decoded_times(path):
    result = {'video': [], 'audio': []}
    with av.open(str(path)) as container:
        durations = [(s.type, s.duration * Fraction(s.time_base) if s.duration is not None else None) for s in container.streams]
        for packet in container.demux():
            for frame in packet.decode():
                if packet.stream.type == 'video':
                    result['video'].append(frame.pts * Fraction(frame.time_base))
                else:
                    result['audio'].append((frame.pts * Fraction(frame.time_base), frame.samples, frame.to_ndarray()))
    return result, durations


@pytest.mark.parametrize('offset', [0, 1600, -1600])
@pytest.mark.parametrize('timestamps', [None, [0, 40, 110, 150, 220, 300, 340, 410, 450, 530, 600, 660]])
def test_video_packet_and_actual_pts_preservation_with_audio_offsets(tmp_path, offset, timestamps):
    source, output = tmp_path / 'source.mp4', tmp_path / 'changed.mp4'
    pts = make_media(source, timestamps=timestamps, offset=offset)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = process_file(source, output, 2, '10')
    assert report['mute_end_seconds'] == str(Fraction(pts[2], 1000))
    assert report['video_packet_bytes_and_timestamps_unchanged']
    a, ad = decoded_times(source)
    b, bd = decoded_times(output)
    assert a['video'] == b['video']
    assert video_signature(source) == video_signature(output)
    assert report['audio_tracks'][0]['first_pcm_pts_seconds'] == str(a['audio'][0][0])
    assert report['audio_tracks'][0]['decoded_source_samples'] == sum(row[1] for row in a['audio'])
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    # AAC decoder padding is not the effective track duration. Do not hide
    # drift by comparing only padded block counts or container success.
    assert dict(ad)['audio'] == dict(bd)['audio']
    with av.open(str(source)) as inp, av.open(str(output)) as out:
        s, t = inp.streams.audio[0], out.streams.audio[0]
        assert s.start_time * Fraction(s.time_base) == t.start_time * Fraction(t.time_base)
        assert report['audio_tracks'][0]['samples'] == s.duration * Fraction(s.time_base) * s.codec_context.sample_rate


def test_actual_vfr_boundary_and_end_without_average_fps(tmp_path):
    source = tmp_path / 'source.mp4'
    make_media(source, timestamps=[0, 40, 110, 150])
    assert boundary(source, 2)[:2] == (Fraction(0), Fraction(11, 100))
    first, end, count, basis = boundary(source, 999)
    assert first == 0 and end > Fraction(15, 100)
    assert count == 4 and basis == 'actual_last_frame_end'


def test_no_audio_and_disabled_file_are_same_object(tmp_path):
    from comfy_api.latest import InputImpl
    from h3_audio_t8_pkg.audio_opening_fade import mute_fade_video_components
    source = tmp_path / 'silent.mp4'
    make_media(source, audio=False)
    video = InputImpl.VideoFromFile(str(source))
    result, report = mute_fade_file_video(video, 1, 10, output_directory=tmp_path, interrupt_check=lambda: None)
    assert result is video and json.loads(report)['status'] == 'no_audio_passthrough'
    assert mute_fade_video_components(video, False)[0] is video


def test_owned_adapter_and_native_savevideo_stream_reuse(tmp_path):
    from comfy_api.latest import InputImpl
    source = tmp_path / 'source.mp4'
    make_media(source)
    video = InputImpl.VideoFromFile(str(source))
    result, report = mute_fade_file_video(video, 1, 10, output_directory=tmp_path, interrupt_check=lambda: None)
    report = json.loads(report)
    assert report['process']['active_after_cleanup'] == 0
    assert report['process']['exit_code'] == 0
    assert not report['core_active_view_materialized']
    assert not report['frame_batch_materialized']
    saved = tmp_path / 'saved.mp4'
    result.save_to(str(saved))
    assert video_signature(source) == video_signature(saved)
    assert decoded_times(saved)[0]['video'] == decoded_times(source)[0]['video']


def test_input_overwrite_existing_target_and_bad_parameters_rejected(tmp_path):
    source = tmp_path / 'source.mp4'
    make_media(source)
    with pytest.raises(ValueError, match='overwrite'):
        process_file(source, source, 1, 10)
    with pytest.raises(ValueError):
        process_file(source, tmp_path / 'bad.mp4', -1, 10)


@pytest.mark.parametrize('offset', [0, 1600, -1600])
def test_decoded_waveform_no_hidden_AAC_content_shift(tmp_path, offset):
    source, target = tmp_path / 'chirp.mp4', tmp_path / 'faded.mp4'
    make_media(source, offset=offset, chirp=True)
    process_file(source, target, 2, '10')
    def pcm(path):
        rows, _ = decoded_times(path)
        samples = np.zeros((2, 16000), np.float32)
        for start, count, values in rows['audio']:
            tick = int(start * 32000)
            lo, hi = max(0, tick), min(16000, tick+count)
            if hi > lo:
                samples[:, lo:hi] = values[:, lo-tick:hi-tick]
        return samples
    before, after = pcm(source), pcm(target)
    # Nonperiodic chirp makes a priming/content shift observable. Compare
    # after envelope/codec transition, but before effective audio tail.
    reference = before[0, 4800:10000]
    scores = [np.corrcoef(reference, after[0, 4800+lag:10000+lag])[0, 1] for lag in range(-64, 65)]
    assert np.argmax(scores)-64 == 0, 'AAC content shifted despite preserved packet PTS'
    assert scores[64] > .98
    assert np.sqrt(np.mean(after[:, 600:2000]**2)) < .003


@pytest.mark.parametrize('view', ['trim', 'crop', 'bytes', 'mkv'])
def test_actual_native_active_view_and_streaming_save(tmp_path, view):
    from io import BytesIO
    from comfy_api.latest import InputImpl
    source = tmp_path / ('source.mkv' if view == 'mkv' else 'source.mp4')
    make_media(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    kwargs = dict(start_time=.08, duration=.2) if view == 'trim' else dict(crop=(0, 0, 16, 32)) if view == 'crop' else {}
    video = InputImpl.VideoFromFile(BytesIO(source.read_bytes()) if view == 'bytes' else str(source), **kwargs)
    result, raw = mute_fade_file_video(video, 1, 10, output_directory=tmp_path, interrupt_check=lambda: None)
    report = json.loads(raw)
    assert report['core_active_view_materialized'] and not report['frame_batch_materialized']
    assert result.get_dimensions() == video.get_dimensions()
    assert report['process']['active_after_cleanup'] == 0
    saved = tmp_path / 'saved.mp4'
    result.save_to(str(saved))
    assert video_signature(result.get_stream_source()) == video_signature(saved)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_actual_owned_worker_cancel_and_next_request_recovery(tmp_path):
    import comfy.model_management as mm
    from comfy_api.latest import InputImpl
    source = tmp_path / 'source.mp4'
    make_media(source)
    video = InputImpl.VideoFromFile(str(source))
    checks = []
    def cancel():
        checks.append(True)
        if len(checks) >= 2:
            raise mm.InterruptProcessingException()
    with pytest.raises(mm.InterruptProcessingException):
        mute_fade_file_video(video, 1, 10, output_directory=tmp_path, interrupt_check=cancel)
    result, report = mute_fade_file_video(video, 1, 10, output_directory=tmp_path, interrupt_check=lambda: None)
    assert json.loads(report)['process']['active_after_cleanup'] == 0
    assert result is not video and video_signature(source) == video_signature(result.get_stream_source())
