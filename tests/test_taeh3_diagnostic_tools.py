"""CPU entry guards only; these tests are not pretrained model qualification."""
import sys
from types import SimpleNamespace

import pytest

from tools import diagnose_taeh3_gpu as diagnostic


@pytest.mark.parametrize('mode, expected', [
    ('first_forward_repeat_only', 'same_loaded_three_native_forwards_partial_cancel'),
    ('single_route', 'one_fresh_actual_call'),
    ('residency_diagnostic', 'two_OFF_actual_calls'),
    ('residency_all_casts', 'two_OFF_actual_calls'),
    (None, 'three_actual_calls'),
])
def test_status_counts_completed_calls_without_claiming_partial_clip(mode, expected):
    flags = dict(first_forward_repeat_only=False, single_route=None,
                 residency_diagnostic=False, residency_all_casts=False)
    if mode:
        flags[mode] = 'off1' if mode == 'single_route' else True
    status = diagnostic.completion_status(SimpleNamespace(**flags))
    assert status.startswith(expected)
    assert 'not_' in status and 'qualification' in status


@pytest.mark.parametrize('modes', [
    ['--residency-diagnostic', '--single-route', 'off1'],
    ['--residency-all-casts', '--first-forward-repeat-only'],
    ['--residency-diagnostic', '--residency-all-casts'],
])
def test_diagnostic_modes_cannot_accidentally_combine(monkeypatch, tmp_path, modes):
    output = tmp_path / 'never-created'
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--recipe', 'unused.json',
                                    '--output', str(output), *modes])
    with pytest.raises(SystemExit) as error:
        diagnostic.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('exists', [False, True])
def test_reject_output_outside_owned_artifacts_before_core_or_gpu(monkeypatch, tmp_path, exists):
    output = tmp_path / 'outside'
    if exists:
        output.mkdir()
    before = list(output.iterdir()) if exists else None
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--recipe', 'unused.json',
                                    '--output', str(output), '--residency-all-casts'])
    with pytest.raises(ValueError, match='Fresh owned output required'):
        diagnostic.main()
    assert output.exists() == exists
    assert not exists or list(output.iterdir()) == before
