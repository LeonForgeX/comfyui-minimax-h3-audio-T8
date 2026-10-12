"""Nonportable drafts skip cache-only scans but retain current-run guards."""

import pytest
import torch

from h3_audio_t8_pkg import progressive_first_pass_draft as draft_runtime
from test_progressive_first_pass_draft import SCOPE, draft_arguments, run, same
from test_progressive_sampling_runtime import stub_lifter, tiny_model  # noqa: F401


def opaque_model():
    model = tiny_model()

    def selected(apply, arguments):
        return apply(arguments['input'], arguments['timestep'], **arguments['c'])

    model.set_model_unet_function_wrapper(selected)
    return model


def count_identities(monkeypatch):
    captures = []
    original = draft_runtime.low_model_identity

    def capture(model, sampler=None):
        captures.append(model)
        return original(model, sampler)

    monkeypatch.setattr(draft_runtime, 'low_model_identity', capture)
    return captures


def test_nonportable_generation_keeps_native_result_and_only_initial_final_scans(
        tmp_path, stub_lifter, monkeypatch):  # noqa: F811
    low, high = opaque_model(), tiny_model()
    baseline, _ = run(model=low, high=high)
    captures = count_identities(monkeypatch)
    generated, report = run(tmp_path, model=low, high=high)
    same(generated, baseline)
    assert captures == [low, high, low, high]
    assert report['counts']['actual_forwards'] == {'low': 2, 'high': 2}
    state = report['first_pass_draft']
    assert state['saved_low'] is False and state['reused_low'] is False
    assert state['portable_reuse'] is False
    assert state['save_status'] == 'nonportable_not_saved'
    assert state['nonportable_reason'] == 'model_stack_unverified'
    assert not list(tmp_path.glob('low-boundary-*'))


@pytest.mark.parametrize('changed_stage', ['low', 'high'])
def test_nonportable_skip_save_still_rejects_fresh_final_weight_mutation(
        tmp_path, monkeypatch, changed_stage):
    low, high = opaque_model(), tiny_model()
    values = draft_arguments(low, high)
    captures = count_identities(monkeypatch)
    session = draft_runtime.ProgressiveFirstPassDraftSession(tmp_path, SCOPE, 'save_or_reuse')
    with session.exclusive():
        assert session._bind_and_load_low(**values) is None
        assert captures == [low, high]
        # This no-cache branch owns no tensor or receipt publication at all.
        assert session.save_low({}, {}) == {'status': 'nonportable_not_saved'}
        assert captures == [low, high]
        changed = low if changed_stage == 'low' else high
        with torch.no_grad():
            next(changed.model.parameters()).data.add_(.1)
        with pytest.raises(ValueError, match='execution inputs changed'):
            session.verify()
    assert not list(tmp_path.glob('low-boundary-*'))


def test_nonportable_skip_save_still_requires_locked_unchanged_contract(tmp_path):
    low, high = opaque_model(), tiny_model()
    values = draft_arguments(low, high)
    session = draft_runtime.ProgressiveFirstPassDraftSession(tmp_path, SCOPE, 'save_or_reuse')
    with pytest.raises(RuntimeError, match='bound and exclusively locked'):
        session.save_low({}, {})
    with session.exclusive():
        session._bind_and_load_low(**values)
        session.contract['settings']['seed'] += 1
        with pytest.raises(ValueError, match='execution inputs changed'):
            session.save_low({}, {})


def test_nonportable_reuse_only_never_resamples_low(tmp_path, stub_lifter):  # noqa: F811
    with pytest.raises(ValueError, match='no portable'):
        run(tmp_path, model=opaque_model(), mode='reuse_only')
    assert not stub_lifter
    assert not list(tmp_path.glob('low-boundary-*'))
