"""Audio/container publication contracts; implementation waits for live H3 terminal."""
from copy import deepcopy

import pytest

from h3_audio_t8_pkg import topaz_media as media
from h3_audio_t8_pkg.topaz_contract import regular_command
from tests.test_topaz_contract import runtime  # noqa: F401
from tests.test_topaz_media_priming_regression import source


def test_aac_master_uses_container_that_preserves_priming():
    assert media.lossless_master_suffix(source()) == '.mov'
    without_side_data = source()
    without_side_data['packets'][0].pop('side_data_list')
    assert media.lossless_master_suffix(without_side_data) == '.mov'


def test_unqualified_other_codec_priming_is_not_silently_dropped():
    probe = source()
    probe['streams'][0]['codec_name'] = 'opus'
    with pytest.raises(ValueError, match='(?i)priming|padding'):
        media.lossless_master_suffix(probe)


def test_silent_or_unprimed_pcm_can_keep_ffv1_mkv():
    assert media.lossless_master_suffix({'streams': [], 'packets': []}) == '.mkv'
    probe = source()
    probe['streams'][0]['codec_name'] = 'pcm_s16le'
    probe['packets'][0].pop('side_data_list')
    assert media.lossless_master_suffix(probe) == '.mkv'


@pytest.mark.parametrize('value', [True, -1, '1024', None, 2**40])
def test_invalid_priming_values_are_rejected_even_if_both_sides_match(value):
    probe = source()
    probe['packets'][0]['side_data_list'][0]['skip_samples'] = value
    with pytest.raises(ValueError, match='(?i)priming|padding|skip'):
        media.compare_audio_packets(probe, deepcopy(probe), '0')


def test_png_mov_preserves_audio_without_lossy_intermediate(runtime, tmp_path):  # noqa: F811
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'fixture')
    command = regular_command(runtime, path, tmp_path / 'new.mov', 'iris-3', 1024, 512,
        output_profile='lossless_master')
    assert command[command.index('-c:v') + 1] == 'png'
    assert command[command.index('-pix_fmt') + 1] == 'rgb48be'
    assert command[command.index('-c:a') + 1] == 'copy'
    assert command[command.index('-enc_time_base:v') + 1] == 'demux'
    assert 'format=rgb48be' in command[command.index('-vf') + 1]
    assert '-r' not in command and '-y' not in command


def test_decoded_pcm_requires_same_bytes_and_count():
    a = [{'stream': 1, 'bytes': 8192, 'sha256': 'a' * 64}]
    b = [{'stream': 4, 'bytes': 8192, 'sha256': 'a' * 64}]
    assert media.compare_pcm_digests(a, b)['streams'] == 1
    for field, value in [('bytes', 16384), ('sha256', 'b' * 64)]:
        changed = deepcopy(b)
        changed[0][field] = value
        with pytest.raises(ValueError, match='PCM'):
            media.compare_pcm_digests(a, changed)
