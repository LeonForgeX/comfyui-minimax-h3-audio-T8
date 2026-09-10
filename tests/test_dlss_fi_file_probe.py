from fractions import Fraction
import json

import pytest

from dlss_fi_contract import TwoXTimeline
from tools import run_dlss_fi_file_probe as control


def test_serialized_plan_preserves_rational_time_and_cut_order():
    source = {'plan':TwoXTimeline(10,Fraction(30000,1001),Fraction(2),frozenset({7,2}))}
    value = control.serialize_source(source)
    assert json.loads(json.dumps(value))['plan'] == {'source_count':10,'source_rate':'30000/1001','origin':'2','cuts':[2,7]}
    assert isinstance(source['plan'],TwoXTimeline)


def test_unqualified_frame_diagnostic_cannot_start_file_route(tmp_path,monkeypatch):
    monkeypatch.setattr(control,'RESEARCH',tmp_path)
    probe = tmp_path/'dlss-fi-frame-probe-20260910-v2'
    probe.mkdir()
    for name,payload in [('terminal',{'status':'failed','generated_frames':0}),('source',{}),('result',{})]:
        (probe/f'{name}.json').write_text(json.dumps(payload))
    def forbidden(*args):
        pytest.fail('Unqualified diagnostic must stop before runtime or GPU access')
    monkeypatch.setattr(control,'runtime_identity',forbidden)
    with pytest.raises(ValueError,match='qualification'):
        control.fixed_source(tmp_path)
