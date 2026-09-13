from fractions import Fraction
import json

import pytest

from h3_audio_t8_pkg.topaz_contract import OfficialTopaz, regular_command, regular_filter, validate_cfr_timeline


@pytest.fixture
def runtime(tmp_path):
    install, definitions, data = (tmp_path / name for name in ('Official Program', 'definitions', 'weights'))
    for path in (install, definitions, data):
        path.mkdir()
    for name in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe'):
        (install / name).write_bytes(b'fixture-not-executable')
    (definitions / 'iris-3.json').write_text(json.dumps({'backends': {'onnx': {}},
        'parameters': [{'name': 'preBlur', 'min': -1, 'max': 1}, {'name': 'noise', 'min': 0, 'max': 1}]}))
    (definitions / 'slp-2.5.json').write_text(json.dumps({'backends': {}, 'isNeuroserverModel': True}))
    return OfficialTopaz(install, definitions, data)


def test_explicit_paths_and_environment_are_local_to_child(runtime):
    inherited = {'PATH': 'unchanged', 'TVAI_MODEL_DIR': 'other'}
    child = runtime.child_environment(inherited)
    assert inherited == {'PATH': 'unchanged', 'TVAI_MODEL_DIR': 'other'}
    assert child['PATH'].startswith(str(runtime.install))
    assert child['PATH'].endswith('unchanged')
    assert child['TVAI_MODEL_DIR'] != child['TVAI_MODEL_DATA_DIR']


def test_regular_arguments_preserve_audio_fps_and_lossless_master(runtime, tmp_path):
    source = tmp_path / 'source & spaces.mp4'
    source.write_bytes(b'fixture-only')
    command = regular_command(runtime, source, tmp_path / 'new result.mkv', 'iris-3', 1920, 1080,
        parameters={'preblur': -.25, 'noise': .5}, output_profile='lossless_master')
    assert command[0] == str(runtime.install / 'ffmpeg.exe')
    assert command[command.index('-i') + 1] == str(source)
    assert 'download=0' in command[command.index('-vf') + 1]
    assert 'instances=0' in command[command.index('-vf') + 1]
    assert command[command.index('-c:a') + 1] == 'copy'
    assert command[command.index('-c:v') + 1] == 'ffv1'
    assert command[command.index('-enc_time_base:v') + 1] == 'demux'
    assert '-r' not in command and 'tvai_fi' not in ' '.join(command)
    assert '-n' in command and '-y' not in command


def test_default_delivery_is_compact_gpu_h264_with_stream_copied_audio(runtime, tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture-only')
    command = regular_command(runtime, source, tmp_path / 'result.mp4', 'iris-3', 1920, 1080)
    assert command[command.index('-c:v') + 1] == 'h264_nvenc'
    assert command[command.index('-cq') + 1] == '16'
    assert command[command.index('-c:a') + 1] == 'copy'
    assert 'format=yuv420p' in command[command.index('-vf') + 1]
    assert '+faststart' in command


def test_delivery_mkv_preserves_uncommon_audio_without_mp4_flags(runtime, tmp_path):
    source = tmp_path / 'source.avi'
    source.write_bytes(b'fixture-only')
    command = regular_command(runtime, source, tmp_path / 'result.mkv', 'iris-3', 1920, 1080)
    assert command[command.index('-c:v') + 1] == 'h264_nvenc'
    assert command[command.index('-c:a') + 1] == 'copy'
    assert '+faststart' not in command


@pytest.mark.parametrize('name', ['../iris-3', 'iris-3:scale=4', 'iris-3,scale=2', '', 'G:/star2.6/model'])
def test_invalid_model_identifier_cannot_inject_filter_or_path(runtime, name):
    with pytest.raises(ValueError, match='model ID'):
        regular_filter(runtime, name, 1920, 1080)


def test_starlight_is_never_fed_into_regular_filter(runtime):
    with pytest.raises(ValueError, match='Neuroserver'):
        regular_filter(runtime, 'slp-2.5', 1920, 1080)


@pytest.mark.parametrize('fields', [{'changesFPS': 1}, {'modelType': 2}, {'modelType': 4}])
def test_regular_filter_rejects_explicit_fps_or_auxiliary_model(runtime, fields):
    definition = {'backends': {}, **fields}
    (runtime.definitions / 'wrong-1.json').write_text(json.dumps(definition))
    with pytest.raises(ValueError, match='not a regular enhancement'):
        regular_filter(runtime, 'wrong-1', 1920, 1080)


@pytest.mark.parametrize('settings', [{'vram': float('nan')}, {'vram': True}, {'device': -2},
    {'parameters': {'download': 1}}, {'parameters': {'noise': 2}},
    {'parameters': {'details': .5}}, {'parameters': {'noise': True}}, {'parameters': {'estimate': 1.5}}])
def test_invalid_or_unsupported_settings_refuse(runtime, settings):
    with pytest.raises(ValueError):
        regular_filter(runtime, 'iris-3', 1920, 1080, **settings)


def test_existing_output_not_overwritten(runtime, tmp_path):
    source = tmp_path / 'source.mp4'
    result = tmp_path / 'existing.mkv'
    source.write_bytes(b'source')
    result.write_bytes(b'keep')
    with pytest.raises(ValueError):
        regular_command(runtime, source, result, 'iris-3', 1920, 1080)
    assert result.read_bytes() == b'keep'


def test_exact_fractional_fps_and_nonzero_origin():
    rate = Fraction(24000, 1001)
    points = [Fraction(3) + i / rate for i in range(73)]
    result = validate_cfr_timeline(points, rate)
    assert result == {'frames': 73, 'fps': '24000/1001', 'origin': '3', 'duration': str(73 / rate)}


@pytest.mark.parametrize('points', [[0, Fraction(1, 24), Fraction(3, 24)],
    [0, Fraction(1, 24), Fraction(1, 24)], [0, Fraction(1, 25)]])
def test_missing_duplicate_or_variable_timestamps_rejected(points):
    with pytest.raises(ValueError, match='VFR'):
        validate_cfr_timeline(points, Fraction(24))
