import hashlib
import json
import numpy as np
import pytest
import torch
import av

from h3_audio_t8_pkg import long_video_dual_color as dc
from h3_audio_t8_pkg import long_video_color_match_advanced as cm
from test_long_video_color_match_advanced import _context, _frames, isolated_state


def test_direct_reference_matches_existing_state_algorithm_without_writes(isolated_state, monkeypatch):
    previous, current = _frames(.4), _frames(.41)
    cm.process_long_video_color_match(previous, _context('color-chain', 0), 'color-chain', 0)
    expected = cm.process_long_video_color_match(current, _context('color-chain', 1), 'color-chain', 1)[0]
    monkeypatch.setattr(cm, '_save_reference', lambda *a, **k: pytest.fail('unexpected state write'))
    actual, _, report = cm.process_long_video_color_match(current, _context('color-chain', 1),
        'color-chain', 1, _reference_frames=previous[-5:], _persist_state=False)
    assert torch.equal(actual, expected)
    assert json.loads(report)['written_state_path'] is None


@pytest.fixture
def accepted(tmp_path):
    path = tmp_path/'previous.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
        stream.codec_context.thread_count = 1
        for _ in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 102, np.uint8), format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    record = {'index':0, 'candidate_id':'parent', 'video_path':path.name,
        'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'frame_count':8}
    manifest = {'chain_id':'chain', 'segments':[record]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    return tmp_path, manifest


def test_real_accepted_video_repeat_is_deterministic_and_no_sidecar(accepted):
    root, _ = accepted
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    source = torch.full((32,32,32,3), .41)
    original = source.clone()
    first, report = dc.correct_dual_segment_color(source,root,'chain',1,'parent')
    second, repeat = dc.correct_dual_segment_color(source,root,'chain',1,'parent')
    assert torch.equal(first,second) and report == repeat
    assert torch.equal(source,original)
    assert report['audio_touched'] is report['latent_touched'] is False
    assert report['maximum_total_rgb_delta'] <= .020001
    assert report['written_state_path'] is None
    assert before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


@pytest.mark.parametrize('mutation', ['candidate','hash','count','chain','escape'])
def test_wrong_predecessor_is_not_silently_used(accepted,mutation):
    root, manifest = accepted
    record = manifest['segments'][0]
    if mutation == 'candidate': record['candidate_id']='wrong'
    elif mutation == 'hash': record['video_sha256']='0'*64
    elif mutation == 'count': record['frame_count']=9
    elif mutation == 'chain': manifest['chain_id']='wrong'
    else: record['video_path']='../outside.mp4'
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
    with pytest.raises((ValueError, FileNotFoundError)):
        dc.correct_dual_segment_color(torch.full((32,32,32,3),.41),root,'chain',1,'parent')


@pytest.mark.parametrize('index,enabled', [(0,True),(1,False)])
def test_disabled_and_first_segment_are_identity_without_loading(tmp_path,index,enabled):
    frames = torch.full((2,32,32,3),.4)
    assert dc.correct_dual_segment_color(frames,tmp_path,'chain',index,'parent',enabled)[0] is frames


def test_dual_color_control_is_append_only_default_on():
    from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8 as Node
    fields = Node.define_schema().inputs
    assert fields[-2].id == 'color_match' and fields[-2].default is True and fields[-2].optional
    assert fields[-1].id == 'video_context_mode' and fields[-1].default == 'reference_only' and fields[-1].optional
