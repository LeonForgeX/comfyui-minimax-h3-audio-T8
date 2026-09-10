"""Exercise shipped modules, not merely the research counterparts."""
from fractions import Fraction
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from h3_audio_t8_pkg.dlss_fi_contract import TwoXTimeline
from h3_audio_t8_pkg.dlss_fi_backend import media
from h3_audio_t8_pkg.dlss_fi_backend.frame_stream import FrameStream
from h3_audio_t8_pkg.dlss_fi_backend.process import IsolatedTaskError, run_isolated
from h3_audio_t8_pkg.dlss_fi_backend.transport import BinarySession
from .test_dlss_fi_media import source_clip


@pytest.mark.parametrize('rate,offset,audio', [(24,0,True), (30,0,True), (24,2,True), (24,0,False)])
def test_packaged_cpu_audio_and_cfr_contract(tmp_path, rate, offset, audio):
    source = media.inspect_source(source_clip(tmp_path, rate=rate, offset=offset, audio=audio))
    def slots():
        # Explicit CPU duplication only to test codec timing, not FI quality.
        plan = iter(source['plan'].slots())
        for frame in media.decode_frames(source):
            yield next(plan), frame.tobytes()
            yield next(plan), frame.tobytes()
    media.encode_video(tmp_path/'video.mp4', source, slots())
    result = media.mux_and_validate(tmp_path/'video.mp4', source, tmp_path/'result.mp4')
    assert result['fps'] == str(2*rate) and result['origin'] == str(offset)
    assert result['audio_streams'] == int(audio) and all(result['audio'].values())


def test_packaged_rejects_hdr_before_worker(tmp_path):
    with pytest.raises(ValueError, match='8-bit'):
        media.inspect_source(source_clip(tmp_path, pixel_format='yuv420p10le'))


def test_packaged_real_pipe_fixture_and_cleanup():
    fixture = Path(__file__).parent/'fixtures/dlss_fi_fake_worker.py'
    with BinarySession([sys.executable,'-u',str(fixture),'fragment'], cwd=fixture.parent,
                       width=2,height=2,frame_count=3,timeout=2) as worker:
        assert worker.frame(bytes(16),bytes(16),Fraction(0),reset=True)['rgba'] is None
        assert worker.frame(bytes(16),bytes(16),Fraction(1,24))['usable_generated_frames'] == 1
    assert worker.process.poll() is not None and not worker.stop_receipt['unfinished_threads']


@pytest.mark.parametrize('case', ['hang','tree_hang'])
@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object')
def test_packaged_job_deadline(case):
    with pytest.raises(IsolatedTaskError) as caught:
        run_isolated(Path(__file__).parent/'fixtures/dlss_fi_isolation_task.py', [case], timeout=1)
    assert caught.value.receipt['status'] == 'timeout'
    assert caught.value.receipt['active_after_cleanup'] == 0


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object')
def test_packaged_job_success():
    report = run_isolated(Path(__file__).parent/'fixtures/dlss_fi_isolation_task.py', ['complete'], timeout=10)
    assert report['job_assigned_before_task'] and report['active_after_cleanup'] == 0


@pytest.mark.parametrize('copy', [False,True])
def test_packaged_frame_provenance(copy):
    class Peer:
        def frame(self, source, motion, stamp, *, reset):
            return {'usable_generated_frames': 0 if reset else 1,
                    'rgba': source if copy else bytes([80])*len(source)}
    plan = TwoXTimeline(3,Fraction(24),Fraction(0),frozenset())
    stream = FrameStream(plan,width=64,height=64,session=Peer())
    frames = [np.full((64,64,4), i*20, dtype=np.uint8) for i in range(3)]
    if copy:
        with pytest.raises(ValueError, match='endpoint copy'):
            list(stream.outputs(frames))
    else:
        assert len(list(stream.outputs(frames))) == 6
        assert stream.report['counts'] == {'source':3,'generated':2,'cut_hold':0,'tail_hold':1}


def test_native_workflow_schema_and_explicit_widgets():
    import json
    from comfy_extras.nodes_video import LoadVideo
    from h3_audio_t8_pkg.nodes_dlss_fi import MiniMaxH3DLSSFrameInterpolationEXPT8 as Node
    from tools.repair_frontend_workflow_order import node_needs_repair
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads((root/'examples/workflows/29-dlss-fi/2026-09-10_H3_DLSS_FI_File_2x_EXP.json').read_text(encoding='utf8'))
    infos = json.loads(json.dumps({c.define_schema().node_id:c.GET_NODE_INFO_V1() for c in (LoadVideo,Node)}))
    assert all(not node_needs_repair(n, infos[n['type']]) for n in workflow['nodes'])
    assert workflow['nodes'][1]['widgets_values'] == ['', '', 600]
    assert workflow['links'] == [[1,1,0,2,0,'VIDEO']]
