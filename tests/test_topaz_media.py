from copy import deepcopy
from fractions import Fraction

import pytest

from h3_audio_t8_pkg.topaz_media import (
    AUTOMATIC_H264_INPUT_BIT_DEPTHS, analyze_video, compare_video,
    compare_audio_packets, compare_transcoded_audio, delivery_audio_mode,
    delivery_suffix, file_identity, qualified_input_bit_depths,
)


def video(*, fps='24', clock='1/24000', width=512, height=256, pix_fmt='yuv420p', bits=None):
    rate, tick = Fraction(fps), Fraction(clock)
    stream = {'codec_type': 'video', 'width': width, 'height': height,
        'pix_fmt': pix_fmt, 'r_frame_rate': fps, 'time_base': clock, 'sample_aspect_ratio': '1:1'}
    if bits is not None:
        stream['bits_per_raw_sample'] = str(bits)
    return {'streams': [stream],
        'frames': [{'width': width, 'height': height, 'best_effort_timestamp': round(i / rate / tick)} for i in range(73)]}


def test_fractional_cfr_can_be_remuxed_to_coarser_clock():
    a = analyze_video(video(fps='24000/1001'))
    b = analyze_video(video(fps='24000/1001', clock='1/1000', width=1024, height=512))
    assert compare_video(a, b, 1024, 512)['frames'] == 73


@pytest.mark.parametrize('pixel_format,expected_depth', [
    ('yuv420p', 8), ('yuv420p10le', 10), ('rgb48le', 16),
])
def test_automatic_h264_route_accepts_qualified_sdr_input_depths(pixel_format, expected_depth):
    probe = video()
    probe['streams'][0]['pix_fmt'] = pixel_format
    result = analyze_video(probe, allowed_bit_depths=AUTOMATIC_H264_INPUT_BIT_DEPTHS)
    assert result['bit_depth'] == expected_depth


def test_output_policy_requires_no_user_bit_depth_choice_for_default_h264():
    assert qualified_input_bit_depths('delivery_h264') == (8, 10, 16)
    assert qualified_input_bit_depths('lossless_master') == (8, 10, 16)
    assert qualified_input_bit_depths('delivery_hevc_main10') == (10,)
    with pytest.raises(ValueError, match='output profile'):
        qualified_input_bit_depths('unknown')


def test_explicit_sdr_bit_depth_profiles_never_silently_convert():
    ten = analyze_video(video(pix_fmt='yuv420p10le', bits=10), allowed_bit_depths=(10,))
    sixteen = analyze_video(video(pix_fmt='gbrp16le', bits=16), allowed_bit_depths=(16,))
    assert ten['bit_depth'] == 10 and sixteen['bit_depth'] == 16
    with pytest.raises(ValueError, match='8-bit'):
        analyze_video(video(pix_fmt='yuv420p10le', bits=10))
    with pytest.raises(ValueError, match='10-bit'):
        analyze_video(video(), allowed_bit_depths=(10,))


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'vfr', 'missing_pts', 'size',
    'interlace', 'hdr', 'ten_bit', 'rotation', 'sar'])
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
    elif kind == 'ten_bit':
        p['streams'][0]['pix_fmt'] = 'yuv420p10le'
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


def test_delivery_container_is_selected_before_inference():
    assert delivery_suffix({'streams': [], 'packets': []}) == '.mp4'
    assert delivery_suffix(audio()) == '.mp4'
    assert delivery_audio_mode(audio()) == 'copy'
    pcm = audio()
    pcm['streams'][0]['codec_name'] = 'pcm_s16le'
    assert delivery_suffix(pcm) == '.mp4'
    assert delivery_audio_mode(pcm) == 'aac'
    opus = audio()
    opus['streams'][0]['codec_name'] = 'opus'
    opus['packets'][0]['side_data_list'] = [{'side_data_type': 'Skip Samples', 'skip_samples': 312}]
    assert delivery_suffix(opus) == '.mp4'
    assert delivery_audio_mode(opus) == 'aac'


def test_aac_fallback_audits_layout_timing_and_pcm_duration():
    source, result = audio(), audio()
    source['streams'][0]['codec_name'] = 'pcm_s16le'
    result['streams'][0]['codec_name'] = 'aac'
    source_pcm = [{'stream': 1, 'bytes': 48000 * 2 * 4, 'sha256': 'a' * 64}]
    output_pcm = [{'stream': 1, 'bytes': (48000 + 1024) * 2 * 4, 'sha256': 'b' * 64}]
    audit = compare_transcoded_audio(source, result, '0', source_pcm, output_pcm)
    assert audit['status'] == 'automatically_transcoded_to_aac_for_mp4'
    assert audit['pcm_frame_deltas'] == [1024]
    output_pcm[0]['bytes'] += 4096 * 2 * 4
    with pytest.raises(ValueError, match='PCM duration'):
        compare_transcoded_audio(source, result, '0', source_pcm, output_pcm)


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
