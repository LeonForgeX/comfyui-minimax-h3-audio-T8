"""Actual CPU codec/mux tests with synthetic buffers, not DLSS GPU proof."""
from fractions import Fraction
from pathlib import Path
import subprocess

import pytest

from dlss_fi_contract import TwoXTimeline
from tools.dlss_fi_media import inspect_source, decode_frames, encode_video, mux_and_validate, encoder_time_base


FFMPEG = Path('C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffmpeg.exe')


def source_clip(tmp_path, *, rate=24, offset=0, audio=True, pixel_format='yuv420p', colors=True):
    if not FFMPEG.exists():
        pytest.skip('Local CPU codec integration requires installed FFmpeg')
    path = tmp_path/'source.mp4'
    args = [str(FFMPEG), '-nostdin', '-v', 'error', '-f', 'lavfi', '-i', f'testsrc2=size=64x64:rate={rate}:duration=0.5']
    if audio:
        args += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.5', '-c:a', 'aac']
    args += ['-fps_mode', 'cfr', '-r', str(rate), '-output_ts_offset', str(offset), '-c:v', 'libx264', '-threads', '1', '-bf', '0', '-pix_fmt', pixel_format,
             ]
    if colors:
        args += ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv']
    args += [str(path)]
    completed = subprocess.run(args, timeout=30, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode('utf8', 'replace')
    return path


def synthetic_outputs(source):
    # Source duplication is deliberate ONLY to test media timing, not a FI pass.
    slots = iter(source['plan'].slots())
    for frame in decode_frames(source):
        yield next(slots), frame.tobytes()
        yield next(slots), frame.tobytes()


@pytest.mark.parametrize('rate,offset,audio', [(24,0,True),(30,0,True),(24,2,True),(24,0,False)])
def test_real_cpu_encode_mux_exact_audio_and_timeline(tmp_path, rate, offset, audio):
    path = source_clip(tmp_path, rate=rate, offset=offset, audio=audio)
    source = inspect_source(path)
    assert source['plan'].source_rate == rate and source['plan'].origin == offset
    video, target = tmp_path/'video.mp4', tmp_path/'final.mp4'
    report = encode_video(video, source, synthetic_outputs(source))
    assert report['encoded_frames'] == source['plan'].output_count
    result = mux_and_validate(video, source, target)
    assert result['fps'] == str(2*rate) and result['origin'] == str(offset)
    assert result['audio_streams'] == int(audio) and all(result['audio'].values())
    assert not result['quality_qualified']
    with pytest.raises(FileExistsError):
        mux_and_validate(video, source, target)


def test_10bit_input_rejected_before_worker(tmp_path):
    path = source_clip(tmp_path, pixel_format='yuv420p10le')
    with pytest.raises(ValueError, match='8-bit'):
        inspect_source(path)


def test_encoder_rejects_incomplete_and_never_overwrites(tmp_path):
    source = inspect_source(source_clip(tmp_path))
    target = tmp_path/'partial.mp4'
    with pytest.raises(ValueError, match='Incomplete'):
        encode_video(target, source, [])
    with pytest.raises(FileExistsError):
        encode_video(target, source, synthetic_outputs(source))


def test_encoder_exact_fractional_clock():
    plan = TwoXTimeline(4,Fraction(30000,1001),Fraction(-1,10),frozenset())
    clock = encoder_time_base(plan)
    assert all((s.pts/clock).denominator == 1 for s in plan.slots())
    with pytest.raises(ValueError, match='time-base'):
        encoder_time_base(TwoXTimeline(4,Fraction(24),Fraction(1,2_000_000_001),frozenset()))


def test_cancel_preflight_stops_cpu_decode(tmp_path):
    path = source_clip(tmp_path)
    def cancelled():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        inspect_source(path, check=cancelled)


def test_unspecified_sdr_color_tags_not_fabricated(tmp_path):
    source = inspect_source(source_clip(tmp_path, colors=False))
    video, target = tmp_path/'video.mp4',tmp_path/'candidate.mp4'
    encode_video(video, source, synthetic_outputs(source))
    mux_and_validate(video, source, target)
    assert inspect_source(target)['colors'] == source['colors']


def test_display_matrix_input_rejected(tmp_path):
    import av
    source = source_clip(tmp_path)
    rotated = tmp_path/'rotated.mp4'
    with av.open(str(source)) as original, av.open(str(rotated),'w') as target:
        stream = original.streams.video[0]
        out_stream = target.add_stream_from_template(stream, opaque=True)
        out_stream.set_display_rotation(90)
        for packet in original.demux(stream):
            if packet.dts is not None:
                packet.stream = out_stream
                target.mux(packet)
    with av.open(str(rotated)) as container:
        assert next(container.decode(video=0)).rotation == 90
    with pytest.raises(ValueError, match='transform|Rotated'):
        inspect_source(rotated)


def test_duplicate_and_out_of_range_cuts_rejected(tmp_path):
    source = source_clip(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        inspect_source(source, cuts=(2,2))
    with pytest.raises(ValueError, match='cut'):
        inspect_source(source, cuts=(999,))
