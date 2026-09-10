from copy import deepcopy
import json
import sys

import pytest

from tools.progressive_qualification import qualification_recipe, requested_frames, CallAudit, require_sage
from tools.run_progressive_pilot import RESEARCH, instrument_recipe
from tools.progressive_pilot_analysis import common_graph
from tools.run_progressive_qualifications import frozen_plan
from tools.run_progressive_pilot import server_command


@pytest.mark.parametrize('qualification',['long32','sage_short'])
def test_two_routes_same_conditions_and_original_unchanged(qualification):
    results = []
    for route in ('native8','progressive6plus2'):
        case = 'T2VA_'+route
        graph = instrument_recipe(json.loads((RESEARCH/'pilot-api-drafts'/f'{case}.prompt.json').read_text(encoding='utf8')))
        original = deepcopy(graph)
        result = qualification_recipe(graph,case,qualification)
        assert graph == original
        assert result['6']['inputs']['width'] == 1024 and result['6']['inputs']['height'] == 512
        assert result['90'] == original['90']
        assert result['10']['inputs']['seed'] == original['10']['inputs']['seed']
        if qualification == 'long32':
            assert result['91']['inputs']['scene_duration_seconds'] == 32
            result['91'] = original['91']
        else:
            assert result['106']['inputs']['model'] == ['104',0]
            assert result['7']['inputs']['model'] == ['106',0]
            result.pop('106')
            result['7'] = original['7']
            result['10']['class_type'] = original['10']['class_type']
        result['18'] = original['18']
        assert result == original
        results.append(qualification_recipe(graph,case,qualification))
    assert common_graph(results[0]) == common_graph(results[1])
    assert requested_frames(qualification) == (768 if qualification=='long32' else 73)


@pytest.mark.parametrize('case,qualification',[('I2VA_native8','long32'),('T2VA_native8','long12'),('unknown','sage_short')])
def test_no_arbitrary_qualification(case,qualification):
    with pytest.raises(ValueError):
        qualification_recipe({},case,qualification)


@pytest.mark.parametrize('qualification',['long32','sage_short'])
def test_serial_plan_has_exactly_two_distinct_matched_jobs(qualification):
    a,b = frozen_plan(qualification)
    assert (a['folder'],b['folder']) == ('native8','progressive6plus2')
    assert a['qualification_case'] == b['qualification_case'] == qualification
    assert common_graph(a['graph']) == common_graph(b['graph'])


def test_extra_headroom_does_not_replace_simple_reserve_or_other_options(tmp_path):
    before = server_command(tmp_path,8197,False)
    after = server_command(tmp_path,8197,False,2)
    assert after == before+['--vram-headroom','2']
    assert before[before.index('--reserve-vram')+1] == '5'
    assert '--vram-headroom' not in before
    with pytest.raises(ValueError):
        server_command(tmp_path,8197,False,-1)


def test_profiler_observes_real_calls_and_restores_after_exception():
    def good():
        return object()
    def bad():
        raise ValueError('sample failure')
    observer = CallAudit({'good':good,'bad':bad})
    with pytest.raises(ValueError,match='sample failure'):
        with observer:
            good()
            bad()
    assert sys.getprofile() is None
    assert observer.counts == {'good':{'calls':1,'successful_returns':1,'empty_returns':0},'bad':{'calls':1,'successful_returns':0,'empty_returns':1}}


def test_existing_profile_is_never_replaced():
    def original(*args):
        return None
    def function():
        return 1
    sys.setprofile(original)
    try:
        with pytest.raises(RuntimeError,match='existing'):
            with CallAudit({'a':function}):
                pass
        assert sys.getprofile() is original
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize('failure',['missing','fallback','empty','short'])
def test_sage_fallback_or_unexecuted_cannot_pass(failure):
    counts = {name:{'calls':400,'successful_returns':400,'empty_returns':0} for name in ('core_sage','sage_kernel')}
    counts['pytorch'] = {'calls':0,'successful_returns':0,'empty_returns':0}
    assert require_sage(counts)['status'].startswith('actual_sage')
    if failure == 'missing':
        counts['core_sage']['calls'] = 0
    elif failure == 'fallback':
        counts['pytorch']['calls'] = 1
    elif failure == 'empty':
        counts['sage_kernel']['empty_returns'] = 1
    else:
        counts['sage_kernel']['successful_returns'] = 399
    with pytest.raises(ValueError):
        require_sage(counts)
