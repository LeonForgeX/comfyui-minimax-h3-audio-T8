"""Timeout accounting and encoder regression; no model/GPU execution."""
import ast
from pathlib import Path

import pytest

from h3_audio_t8_pkg import prepared_generation_runtime as runtime


@pytest.mark.parametrize('kind', ['tao_stream', 'tao5s', 'ltx_refine'])
def test_decoder_reserves_two_hash_scans(tmp_path, kind):
    path = tmp_path / 'bound-input.bin'
    path.write_bytes(b'fixture')
    result = runtime.worker_budget(runtime.ROUTES[kind][1], {'identities': {str(path): 'digest'}})
    assert result == dict(base_seconds=900, verification_bytes=7, hash_scans=2,
                          assumed_hash_bytes_per_second=32 * 1024**2, seconds=901)


def test_real_failure_size_and_cap(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(Path, 'resolve', lambda self, **kwargs: self)
    monkeypatch.setattr(Path, 'is_file', lambda self: True)
    for size, expected in [(74587693930, 5346), (10**13, 7200), (0, 900)]:
        monkeypatch.setattr(Path, 'stat', lambda self, size=size: SimpleNamespace(st_size=size))
        assert runtime.worker_budget(runtime.ROUTES['tao_stream'][1],
                                     {'identities': {'fixture': 'digest'}})['seconds'] == expected


@pytest.mark.parametrize('payload', [{}, {'identities': None}, {'identities': {}}])
def test_missing_decoder_identities_are_not_guessed(payload):
    with pytest.raises(ValueError, match='bound identities'):
        runtime.worker_budget(runtime.ROUTES['tao_stream'][1], payload)


def test_missing_file_fails_and_generation_budget_unchanged(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.worker_budget(runtime.ROUTES['tao_stream'][1],
                              {'identities': {str(tmp_path / 'missing'): 'digest'}})
    assert runtime.worker_budget(runtime.ROUTES['tao_stream'][0], {})['seconds'] == 7200
    assert runtime.worker_budget(('fixture.py', 'ok', 'out', 5), {})['seconds'] == 5


def test_exact_stream_encoder_command_keeps_timing_and_bounds_threads():
    # Evaluate the actual command expression, not a separately copied command.
    path = runtime.BACKEND / 'taomate_stream_decode_worker.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    assignment = next(node for node in run.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'command' for t in node.targets))
    command = eval(compile(ast.Expression(assignment.value), str(path), 'eval'),
                   {'root': Path('fixture'), 'destination': Path('output.mp4'),
                    'video_filter': 'bound-video-filter', 'audio_filter': 'bound-audio-filter',
                    'pf': 240, 'seconds': 10})
    for option, value in {'-threads:v': '1', '-c:v': 'libx264', '-crf': '18',
                          '-vf': 'bound-video-filter', '-af': 'bound-audio-filter',
                          '-frames:v': '240', '-t': '10', '-c:a': 'aac'}.items():
        assert command.count(option) == 1 and command[command.index(option) + 1] == value
