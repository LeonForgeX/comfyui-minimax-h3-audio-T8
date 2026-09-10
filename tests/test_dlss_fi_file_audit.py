from copy import deepcopy
from fractions import Fraction

import pytest

from tools.audit_dlss_fi_file_probe import check_rows


def records():
    hashes = ['a'*64,'b'*64,'c'*64]
    rows = [{'slot':i,'pts':str(Fraction(2)+Fraction(i,48)),
             'kind':'source' if i%2 == 0 else ('tail_hold' if i==5 else 'generated'),
             'rgb_sha256': (hashes[i//2] if i%2 == 0 or i==5 else 'd'*64)} for i in range(6)]
    return rows,hashes


def test_independent_frame_record_accounting():
    rows,hashes = records()
    assert check_rows(rows,hashes,rate=Fraction(24),origin=Fraction(2)) == {'source':3,'generated':2,'cut_hold':0,'tail_hold':1}


@pytest.mark.parametrize('fault',['short','time','kind','source','copy','tail','hash'])
def test_tampered_record_rejected(fault):
    rows,hashes = records()
    rows = deepcopy(rows)
    if fault == 'short':
        rows.pop()
    elif fault == 'time':
        rows[1]['pts']='0'
    elif fault == 'kind':
        rows[1]['kind']='source'
    elif fault == 'source':
        rows[0]['rgb_sha256']='f'*64
    elif fault == 'copy':
        rows[1]['rgb_sha256']=hashes[0]
    elif fault == 'tail':
        rows[-1]['rgb_sha256']='f'*64
    else:
        rows[1]['rgb_sha256']='bad'
    with pytest.raises(ValueError):
        check_rows(rows,hashes,rate=Fraction(24),origin=Fraction(2))
