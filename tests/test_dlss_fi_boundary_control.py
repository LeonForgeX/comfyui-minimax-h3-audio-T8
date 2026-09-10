from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import pytest

from dlss_fi_contract import TwoXTimeline
from tools import run_dlss_fi_boundaries as control
from tools.audit_dlss_fi_boundaries import check_records


def records(cuts=()):
    plan = TwoXTimeline(5, Fraction(30), Fraction(2), frozenset(cuts))
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5)]
    rows = []
    for slot in plan.slots():
        input_index = slot.source_index+1 if slot.kind in ('generated', 'cut_hold') else slot.source_index
        rows.append({'slot': slot.index, 'pts': str(slot.pts), 'kind': slot.kind, 'input_index': input_index,
            'reset': False if slot.kind == 'tail_hold' else input_index == 0 or input_index in cuts,
            'rgb_sha256': hashlib.sha256(f'g{slot.index}'.encode()).hexdigest() if slot.kind == 'generated' else hashes[slot.source_index]})
    return rows, hashes, plan


@pytest.mark.parametrize('cuts', [(), (2,), (1, 2, 3, 4)])
def test_independent_records_distinguish_generation_cut_and_tail(cuts):
    rows, hashes, plan = records(cuts)
    assert check_records(rows, hashes, plan) == control.expected_counts(plan)


@pytest.mark.parametrize('field,value', [('pts', '2'), ('kind', 'generated'), ('input_index', 1),
                                       ('reset', False), ('rgb_sha256', '0'*64)])
def test_changed_cut_row_rejected(field, value):
    rows, hashes, plan = records((2,))
    rows[3][field] = value
    with pytest.raises(ValueError):
        check_records(rows, hashes, plan)


def test_missing_records_and_endpoint_duplication_rejected():
    rows, hashes, plan = records()
    with pytest.raises(ValueError):
        check_records(rows[:-1], hashes, plan)
    rows[1]['rgb_sha256'] = hashes[0]
    with pytest.raises(ValueError, match='endpoint'):
        check_records(rows, hashes, plan)


def test_plan_never_touches_gpu_or_creates_run_root(tmp_path, monkeypatch):
    _, _, plan = records()
    monkeypatch.setattr(control, 'boundary_source', lambda *args: {'plan': plan})
    monkeypatch.setattr(control, 'fixed_source', lambda *args: ({}, {}))
    def forbidden(*args, **kwargs):
        pytest.fail('CPU plan must not acquire GPU resources')
    monkeypatch.setattr(control, 'NvmlResourceReader', forbidden)
    monkeypatch.setattr(control, 'SerialProbeLease', forbidden)
    target = tmp_path/'output'
    result = control.run(SimpleNamespace(manifest=None, case='hud30', runtime=None, mode='plan', run_root=target))
    assert result['status'] == 'CPU_plan_no_worker'
    assert not target.exists()


def test_manifest_cannot_redirect_fixed_fixture(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'status': 'CPU_synthetic_sources_only_no_FI_execution',
        'cases': {'hud30': {'file': {'path': 'elsewhere'}}}}))
    with pytest.raises((ValueError, FileNotFoundError)):
        control.boundary_source(manifest, 'hud30')
