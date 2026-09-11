"""Regression for MP4 AAC priming; red until the runtime postflight is strengthened."""
from copy import deepcopy

import pytest

from h3_audio_t8_pkg.topaz_media import compare_audio_packets


def source():
    return {'streams': [{'codec_type': 'audio', 'index': 1, 'codec_name': 'aac',
        'sample_rate': '32000', 'channels': 2, 'time_base': '1/32000'}],
        'packets': [{'stream_index': 1, 'pts': -1024, 'data_hash': 'SHA256:same',
            'side_data_list': [{'side_data_type': 'Skip Samples', 'skip_samples': 1024,
                'discard_padding': 0, 'skip_reason': 0, 'discard_reason': 0}]}]}


def test_unchanged_priming_is_allowed():
    a = source()
    assert compare_audio_packets(a, deepcopy(a), '0')['packets'] == 1


@pytest.mark.parametrize('kind', ['missing', 'changed_skip', 'changed_discard'])
def test_equal_packet_bytes_do_not_hide_changed_priming(kind):
    a, b = source(), source()
    if kind == 'missing':
        b['packets'][0].pop('side_data_list')
    else:
        field = 'skip_samples' if kind == 'changed_skip' else 'discard_padding'
        b['packets'][0]['side_data_list'][0][field] += 1
    with pytest.raises(ValueError, match='(?i)priming|padding|skip'):
        compare_audio_packets(a, b, '0')
