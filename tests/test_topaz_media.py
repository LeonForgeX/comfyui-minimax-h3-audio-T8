from copy import deepcopy
from fractions import Fraction

import pytest

from h3_audio_t8_pkg.topaz_media import analyze_video, compare_video, compare_audio_packets, file_identity


def video(*, fps='24', clock='1/24000', width=512, height=256):
    rate, tick = Fraction(fps), Fraction(clock)
    return {'streams': [{'codec_type': 'video', 'width': width, 'height': height,
        'pix_fmt': 'yuv420p', 'r_frame_rate': fps, 'time_base': clock, 'sample_aspect_ratio': '1:1'}],
        'frames': [{'width': width, 'height': height, 'best_effort_timestamp': round(i / rate / tick)} for i in range(73)]}


def test_fractional_cfr_can_be_remuxed_to_coarser_clock():
    a = analyze_video(video(fps='24000/1001'))
    b = analyze_video(video(fps='24000/1001', clock='1/1000', width=1024, height=512))
    assert compare_video(a, b, 1024, 512)['frames'] == 73


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'vfr', 'missing_pts', 'size', 'interlace', 'hdr', 'rotation', 'sar'])
def test_media_hazards_are_not_silently_normalized(kind):
    p = video()
    if kind == 'missing':
        p['frames'].pop(4)
    elif kind == 'duplicate':
        p['frames'][4] = deepcopy(p['frames'][3])
    elif kind == 'vfr':
        p['frames'][4]['best_effort_timestamp'] += 50
    elif kind == 'missing_pts':
        p['frames'][4].pop('best_effort_timestamp')
    elif kind == 'size':
        p['frames'][4]['width'] += 32
    elif kind == 'interlace':
        p['frames'][4]['interlaced_frame'] = 1
    elif kind == 'hdr':
        p['streams'][0]['color_transfer'] = 'smpte2084'
    elif kind == 'rotation':
        p['streams'][0]['side_data_list'] = [{'rotation': 90}]
    else:
        p['streams'][0]['sample_aspect_ratio'] = '4:3'
    with pytest.raises(ValueError):
        analyze_video(p)


def audio():
    return {'streams': [{'codec_type': 'audio', 'index': 1, 'codec_name': 'aac',
        'sample_rate': '48000', 'channels': 2, 'channel_layout': 'stereo', 'time_base': '1/48000'}],
        'packets': [{'stream_index': 1, 'pts': i * 1024, 'data_hash': 'SHA256:' + str(i)} for i in range(10)]}


def test_original_audio_packets_allow_only_common_av_shift():
    a, b = audio(), audio()
    for p in b['packets']:
        p['pts'] += 48000
    assert compare_audio_packets(a, b, '1')['packets'] == 10
    with pytest.raises(ValueError, match='relative timing'):
        compare_audio_packets(a, b, '0')


@pytest.mark.parametrize('kind', ['bytes', 'missing', 'codec', 'channels', 'pts', 'streams'])
def test_audio_copy_failures_rejected(kind):
    a, b = audio(), audio()
    if kind == 'bytes':
        b['packets'][4]['data_hash'] = 'SHA256:changed'
    elif kind == 'missing':
        b['packets'].pop(4)
    elif kind in ('codec', 'channels'):
        b['streams'][0]['codec_name' if kind == 'codec' else 'channels'] = 'changed'
    elif kind == 'pts':
        b['packets'][4].pop('pts')
    else:
        b['streams'] = []
    with pytest.raises(ValueError):
        compare_audio_packets(a, b, '0')


def test_same_filename_changed_bytes_get_new_identity(tmp_path):
    p = tmp_path / 'video.mp4'
    p.write_bytes(b'one')
    first = file_identity(p)
    p.write_bytes(b'two')
    assert file_identity(p)['sha256'] != first['sha256']
