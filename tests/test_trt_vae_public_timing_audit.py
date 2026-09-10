from copy import deepcopy

import pytest

from tools.audit_trt_vae_public_timing import summarize


def rows():
    return [{'route':route,'sequence_index':i,'actual_tile_calls':60,
        'seconds':{'load':4.,'decode_including_transfer':20. if route=='native' else 10.,
                   'verify_and_tensor_save':1.,'mp4_and_audio_copy':2.,
                   'load_to_final_mp4':27. if route=='native' else 17.}}
            for i,route in enumerate(('native','trt','trt','native','native','trt'))]


def test_whole_output_stage_not_bare_decode_ratio():
    result = summarize(rows())
    assert result['later_output_stage_speed_ratio'] == pytest.approx(27/17)
    assert result['later_output_stage_speed_ratio'] != 2


@pytest.mark.parametrize('mutation',('order','sum','nan','negative','bool','calls','index','count'))
def test_misleading_timing_records_rejected(mutation):
    data = deepcopy(rows())
    if mutation == 'order':
        data[0]['route'] = 'trt'
    elif mutation == 'sum':
        data[0]['seconds']['load_to_final_mp4'] = 20.
    elif mutation == 'calls':
        data[0]['actual_tile_calls'] = 1
    elif mutation == 'index':
        data[0]['sequence_index'] = 3
    elif mutation == 'count':
        data.pop()
    else:
        data[0]['seconds']['load'] = {'nan':float('nan'),'negative':-1.,'bool':True}[mutation]
    with pytest.raises(ValueError):
        summarize(data)


def test_first_observation_not_used_in_later_medians():
    data = rows()
    data[0]['seconds']['load'] += 100
    data[0]['seconds']['load_to_final_mp4'] += 100
    result = summarize(data)
    assert result['native']['first_observation']['load'] == 104
    assert result['native']['later_two_medians']['load'] == 4
