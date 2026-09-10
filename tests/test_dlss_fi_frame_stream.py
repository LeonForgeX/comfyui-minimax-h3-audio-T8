from fractions import Fraction

import numpy as np
import pytest

from dlss_fi_contract import TwoXTimeline
from tools.dlss_fi_frame_stream import FrameStream


class Peer:
    """Synthetic CPU responses only, deliberately no image interpolation algorithm."""
    def __init__(self, mode='normal'):
        self.index, self.resets, self.mode = 0, [], mode
    def frame(self, source, flow, stamp, *, reset):
        assert type(stamp) is Fraction
        self.index += 1
        self.resets.append(reset)
        payload = bytes([20+self.index])*len(source) if self.mode == 'normal' else source
        return {'usable_generated_frames': 0 if reset or self.mode == 'missing' else 1, 'rgba': payload}


def frames(count):
    for i in range(count):
        rgba = np.full((64,64,4), i*3, dtype=np.uint8)
        rgba[...,3] = 255
        yield rgba


@pytest.mark.parametrize('rate', [24,Fraction(24),Fraction(30),Fraction(24000,1001)])
def test_2x_full_sequence_source_cut_and_tail_counts(rate):
    peer, records = Peer(), []
    plan = TwoXTimeline(4,rate,Fraction(-1,10),frozenset({2}))
    stream = FrameStream(plan,width=64,height=64,session=peer,observer=records.append)
    result = list(stream.outputs(frames(4)))
    assert len(result) == 8 and peer.resets == [True,False,True,False]
    assert [slot.pts for slot,_ in result] == [plan.origin+Fraction(i)/(2*rate) for i in range(8)]
    assert result[3][1] == result[2][1] and result[-1][1] == result[-2][1]
    assert stream.report['counts'] == {'source':4,'generated':2,'cut_hold':1,'tail_hold':1}
    assert len(records) == 8 and not stream.report['audio_qualified']
    with pytest.raises(RuntimeError):
        list(stream.outputs(frames(4)))


@pytest.mark.parametrize('mode', ['copy','missing'])
def test_missing_or_endpoint_copy_stops_stream_without_completion(mode):
    stream = FrameStream(TwoXTimeline(3,Fraction(24),Fraction(0),frozenset()),width=64,height=64,session=Peer(mode))
    with pytest.raises(ValueError):
        list(stream.outputs(frames(3)))
    assert stream.report is None


@pytest.mark.parametrize('actual', [0,2,4])
def test_source_length_mismatch_stops_stream(actual):
    stream = FrameStream(TwoXTimeline(3,Fraction(24),Fraction(0),frozenset()),width=64,height=64,session=Peer())
    with pytest.raises(ValueError):
        list(stream.outputs(frames(actual)))
    assert stream.report is None


def test_early_consumer_stop_does_not_create_pass_report():
    stream = FrameStream(TwoXTimeline(3,Fraction(24),Fraction(0),frozenset()),width=64,height=64,session=Peer())
    iterator = stream.outputs(frames(3))
    next(iterator)
    iterator.close()
    assert stream.report is None
