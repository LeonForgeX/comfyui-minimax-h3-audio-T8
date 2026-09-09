import copy

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_color import color_match_outpaint_frames
from h3_audio_t8_pkg.video_outpaint_finish import finish_outpaint_frames
from test_video_outpaint_alignment import _fixture


def test_disabled_geometry_exactly_matches_old_color_and_state_contract():
    source,candidate,plan,_=_fixture()
    expected,expected_state,expected_report=color_match_outpaint_frames(source,candidate,plan)
    actual,state,report=finish_outpaint_frames(source,candidate,plan)
    assert torch.equal(actual,expected)
    assert state==expected_state and report==expected_report


@pytest.mark.parametrize('color',[False,True])
def test_geometry_is_independent_of_color_and_chunk_invariant(color):
    source,candidate,plan,_=_fixture()
    whole,state,report=finish_outpaint_frames(source,candidate,plan,geometry_align=True,enabled=color)
    first,continuation,_=finish_outpaint_frames(source[:1],candidate[:1],plan,geometry_align=True,enabled=color)
    last,final,_=finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=continuation,
                                      geometry_align=True,enabled=color)
    assert torch.equal(whole,torch.cat((first,last))) and final==state
    assert torch.equal(whole[:,64:192,64:192],source)
    assert report['source_exact_before_encoding']
    assert report['geometry_alignment']['frames'][0]['changed_pixels']>0


def test_finishing_rejects_changed_geometry_gaps_and_state_tamper():
    source,candidate,plan,_=_fixture()
    _,state,_=finish_outpaint_frames(source[:1],candidate[:1],plan,geometry_align=True)
    for extra in ({'alignment_band_pixels':32},{'alignment_max_displacement':4},{'geometry_align':False}):
        with pytest.raises(ValueError):
            finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=state,
                                   **dict({'geometry_align':True},**extra))
    broken=copy.deepcopy(state)
    broken['next_frame']=0
    with pytest.raises(ValueError):
        finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=broken,geometry_align=True)


def test_joint_decode_is_unmodified_candidate_and_never_claims_source_exact(monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_alignment as alignment
    source,candidate,plan,_=_fixture()
    before=candidate.clone()
    def forbidden(*args,**kwargs):
        raise AssertionError('joint decode must not apply source-edge geometry')
    monkeypatch.setattr(alignment,'register_outpaint_candidate',forbidden)
    whole,state,report=finish_outpaint_frames(source,candidate,plan,source_mode='joint_decode',geometry_align=True)
    assert torch.equal(whole,before) and torch.equal(candidate,before)
    assert whole.data_ptr()!=candidate.data_ptr()
    assert report['source_exact_before_encoding'] is False and report['source_reconstructed']
    assert report['source_postprocessing_bypassed'] and not report['color_match_applied']
    first,continuation,_=finish_outpaint_frames(source[:1],candidate[:1],plan,source_mode='joint_decode',geometry_align=True)
    second,last,_=finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=continuation,
                                        source_mode='joint_decode',geometry_align=True)
    assert torch.equal(whole,torch.cat((first,second))) and state==last
    for mode in ('preserve_source','invalid'):
        with pytest.raises(ValueError):
            finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=continuation,source_mode=mode)


def test_joint_decode_rejects_prior_exact_state_changed_settings_or_bad_pixels():
    source,candidate,plan,_=_fixture()
    _,exact_state,_=finish_outpaint_frames(source[:1],candidate[:1],plan)
    with pytest.raises(ValueError):
        finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=exact_state,source_mode='joint_decode')
    _,joint_state,_=finish_outpaint_frames(source[:1],candidate[:1],plan,source_mode='joint_decode')
    with pytest.raises(ValueError):
        finish_outpaint_frames(source[1:],candidate[1:],plan,start_frame=1,state=joint_state,source_mode='joint_decode',enabled=False)
    candidate[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):
        finish_outpaint_frames(source,candidate,plan,source_mode='joint_decode')
