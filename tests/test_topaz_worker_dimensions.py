"""Run the real worker control flow with fake external executables, CPU only."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg import topaz_contract as contract, topaz_media as media, topaz_worker as worker
from tests.test_topaz_contract import runtime  # noqa: F401


def video_probe(width, height):
    return {'streams': [{'codec_type': 'video', 'width': width, 'height': height,
        'r_frame_rate': '24/1', 'time_base': '1/12288', 'pix_fmt': 'yuv420p'}],
        'frames': [{'width': width, 'height': height, 'best_effort_timestamp': i * 512}
                   for i in range(2)]}


@pytest.mark.parametrize('case', ['custom', 'legacy', 'invalid_target', 'wrong_output',
    'missing_trace', 'engine_failed', 'decode_failed', 'source_changed', 'disk_advisory'])
def test_worker_geometry_and_publication_gates(runtime, monkeypatch, tmp_path, case):  # noqa: F811
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fake-source-not-real-media')
    job = tmp_path / 'job'
    job.mkdir()
    width = 1537 if case == 'invalid_target' else 1536
    settings = {'model_id': 'iris-3', 'width': width, 'height': 768, 'scale': 2,
        'device': 0, 'vram': .8, 'parameters': {}, 'size_mode': 'target_dimensions'}
    if case == 'legacy':
        settings.update(width=None, height=None)
        settings.pop('size_mode')
    spec = {'install': str(runtime.install), 'definitions': str(runtime.definitions),
        'data': str(runtime.data), 'job': str(job), 'source': media.file_identity(source),
        'installation': {'executables': []},
        'model': {'definition': media.file_identity(runtime.definitions / 'iris-3.json'),
                  'candidate_weights': []}, 'settings': settings}
    request = job / 'request.json'
    request.write_text(json.dumps(spec))
    monkeypatch.setattr(worker.sys, 'argv', ['worker', str(request)])
    monkeypatch.setattr(worker, 'local_module', lambda name:
        {'topaz_contract': contract, 'topaz_media': media}[name])
    monkeypatch.setattr(media, 'decoded_pcm_digests', lambda *a: [])
    monkeypatch.setattr(worker.shutil, 'disk_usage',
        lambda _: SimpleNamespace(free=700 * 1024**2 if case == 'disk_advisory' else 10**12))
    calls = []

    def external(command, *, stdout, stderr, **kwargs):
        calls.append(command)
        code = 0
        if '-show_frames' in command:
            original = command[-1] == str(source)
            dimensions = (1024, 512) if original else (
                (2048, 1024) if case in ('legacy', 'wrong_output') else (1536, 768))
            stdout.write(json.dumps(video_probe(*dimensions)).encode())
        elif '-show_packets' in command:
            stdout.write(b'{"streams":[],"packets":[]}')
        elif '-h' in command:
            stdout.write(b'Encoder h264_nvenc [NVIDIA NVENC H.264 encoder]:')
        elif '-vf' in command:
            Path(command[-1]).write_bytes(b'fake-output-not-real-media')
            if case == 'source_changed':
                source.write_bytes(b'changed')
            if case != 'missing_trace':
                for index in range(2):
                    stderr.write(f'[showinfo@t8_topaz_native @ 1] n: {index} pts: {index} s:2048x1024 i:P\n'.encode())
            code = 1 if case == 'engine_failed' else 0
        elif '-xerror' in command:
            code = 1 if case == 'decode_failed' else 0
        else:
            pytest.fail(f'Unexpected external call: {command}')
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(worker.subprocess, 'run', external)
    if case in ('custom', 'legacy', 'disk_advisory'):
        worker.main()
        result = json.loads((job / 'result.json').read_text())
        assert result['geometry']['ratio'] == ('2' if case == 'legacy' else '3/2')
        assert result['geometry']['post_ai_resampling'] == (
            'none' if case == 'legacy' else 'lanczos_exact_target_dimensions')
        assert (job / 'enhanced.mp4').is_file()
        if case == 'disk_advisory':
            preflight = json.loads((job / 'disk_preflight.json').read_text())
            assert preflight['available_bytes'] < preflight['required_bytes']
            assert preflight['status'] == 'advisory_below_profile_estimate'
            assert preflight['blocking'] is False
            assert preflight['runtime_stop_floor_bytes'] == 256 * 1024**2
    else:
        with pytest.raises((ValueError, RuntimeError)):
            worker.main()
        assert not (job / 'result.json').exists()
        assert not (job / 'enhanced.mp4').exists()
    enhancement = [command for command in calls if '-vf' in command]
    assert len(enhancement) == (0 if case == 'invalid_target' else 1)
    assert all('download=0' in command[command.index('-vf') + 1] for command in enhancement)
