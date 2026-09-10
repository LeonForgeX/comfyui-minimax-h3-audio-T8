import numpy as np
import pytest

from tools.prepare_dlss_fi_boundaries import render,CASES


@pytest.mark.parametrize('case',CASES)
def test_deterministic_cpu_sources_without_generating_FI(case):
    frame=render(case,3)
    assert frame.shape==(256,512,3) and frame.dtype==np.uint8
    assert np.array_equal(frame,render(case,3))
    assert not np.array_equal(frame,render(case,4))


def test_hud_stays_fixed_while_texture_moves():
    a,b=render('hud30',3),render('hud30',8)
    assert np.array_equal(a[:38],b[:38]) and not np.array_equal(a[38:],b[38:])


def test_flash_is_not_declared_as_a_hard_cut():
    assert CASES['flash_low_texture30'][2]==()
    assert np.array_equal(render('flash_low_texture30',20),render('flash_low_texture30',21))
    assert np.std(render('flash_low_texture30',20))==0
    assert not np.array_equal(render('flash_low_texture30',19),render('flash_low_texture30',20))
    assert CASES['hard_cut24'][2]==(24,)


@pytest.mark.parametrize('case,index',[('unknown',0),('hud30',60),('hud30',-1)])
def test_bad_source_requests_rejected(case,index):
    with pytest.raises(ValueError):
        render(case,index)
