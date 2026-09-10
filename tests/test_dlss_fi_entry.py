import io
import json
import os
import sys
from pathlib import Path

import pytest

from h3_audio_t8_pkg.dlss_fi_backend import entry
from h3_audio_t8_pkg.dlss_fi_backend.process import IsolatedTaskError, run_isolated
from h3_audio_t8_pkg.dlss_fi_backend.resources import file_identity
from h3_audio_t8_pkg.nodes_dlss_fi import MiniMaxH3DLSSFrameInterpolationEXPT8 as Node


class VideoFromFile:
    def __init__(self, file, start=0, duration=0, crop=None):
        self.__file, self.__start_time, self.__duration, self.__crop = file, start, duration, crop

    def get_dimensions(self):
        raise AssertionError('No parent decode')

    get_active_trim_window = get_dimensions
    get_stream_source = get_dimensions


def test_file_source_never_calls_codec_methods(tmp_path):
    source = tmp_path/'source.mp4'
    source.touch()
    assert entry.file_video_path(VideoFromFile(str(source)), VideoFromFile) == source


@pytest.mark.parametrize('settings', [{'start': -1}, {'start': 1}, {'duration': 3}, {'crop': (0,0,2,2)}, {'start': float('nan')}])
def test_file_source_rejects_unrendered_edits_without_decode(tmp_path, settings):
    with pytest.raises(ValueError, match='trim/crop'):
        entry.file_video_path(VideoFromFile(str(tmp_path/'unused'), **settings), VideoFromFile)


def test_rejects_memory_unknown_and_subclasses():
    class Edited(VideoFromFile):
        pass
    for value in (object(), Edited('unused'), VideoFromFile(io.BytesIO())):
        with pytest.raises(ValueError):
            entry.file_video_path(value, VideoFromFile)


@pytest.mark.parametrize('text', ['0', '-1', '1.5', 'true', '1,1', '2,', '[]', '1000000'])
def test_invalid_cut_input(text):
    with pytest.raises(ValueError):
        entry.parse_cuts(text)


def test_cut_indices_are_sorted_exact_integers():
    assert entry.parse_cuts('48, 24') == [24,48]
    assert entry.parse_cuts(' ') == []


def test_public_schema_is_optional_and_has_no_license_checkbox():
    schema = Node.define_schema()
    assert schema.is_experimental
    assert [item.id for item in schema.inputs] == ['source_video', 'runtime_directory', 'cut_frames', 'timeout_seconds']
    assert not any(name.startswith('h3_audio_t8_pkg.dlss_fi_backend.file_task') for name in sys.modules)


def fake_task(args):
    task = Path(args[args.index('--task-dir')+1])
    source = Path(args[args.index('--source')+1])
    candidate = task/'candidate.mp4'
    candidate.write_bytes(b'CPU fixture not a real video')
    result = {'status': 'file_validated_pending_atomic_publish', 'media': {'file': file_identity(candidate)},
              'source': file_identity(source), 'worker_stop': {'owned_children_remaining': [], 'unfinished_threads': []}}
    (task/'result.json').write_text(json.dumps(result))


@pytest.mark.skipif(os.name != 'nt', reason='Windows FI parent route')
def test_atomic_output_keeps_source_and_never_decodes_parent(tmp_path, monkeypatch):
    source = tmp_path/'source.mp4'
    source.write_bytes(b'original')
    def launch(script, args, **kwargs):
        fake_task(args)
        return {'status': 'complete', 'active_after_cleanup': 0}
    monkeypatch.setattr(entry, 'run_isolated', launch)
    saved, report = entry.process_file(source, tmp_path, tmp_path)
    assert saved != source and source.read_bytes() == b'original'
    assert saved.read_bytes() == b'CPU fixture not a real video'
    assert report['status'] == 'completed'


@pytest.mark.skipif(os.name != 'nt', reason='Windows FI parent route')
def test_invalid_result_does_not_publish(tmp_path, monkeypatch):
    source = tmp_path/'source.mp4'
    source.write_bytes(b'original')
    def launch(script, args, **kwargs):
        fake_task(args)
        source.write_bytes(b'changed concurrently')
        return {}
    monkeypatch.setattr(entry, 'run_isolated', launch)
    with pytest.raises(ValueError, match='identity'):
        entry.process_file(source, tmp_path, tmp_path)
    assert not list(tmp_path.glob('H3_DLSS*.mp4'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object')
def test_packaged_isolation_callback_failure_cleans_tree():
    def stop():
        raise RuntimeError('CPU test cancellation')
    task = Path(__file__).parent/'fixtures/dlss_fi_isolation_task.py'
    with pytest.raises(IsolatedTaskError) as caught:
        run_isolated(task, ['tree_hang'], timeout=10, check=stop)
    assert caught.value.receipt['status'] == 'controller_failed'
    assert caught.value.receipt['active_after_cleanup'] == 0
    assert 'CPU test cancellation' in caught.value.receipt['error']
