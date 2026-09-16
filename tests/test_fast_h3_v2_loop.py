import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2DualModelLongVideoEXPT8
from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8


def _engine(monkeypatch, **kwargs):
    monkeypatch.setattr(v2, '_gates', lambda model: None)
    monkeypatch.setattr(v2, 'capture_fast_h3_v2_owner', lambda model: None)
    settings = dict(contract={}, low_width=256, low_height=384, upscaler_model='learned.safetensors',
        first_shift_video=10., first_shift_audio=3., second_shift_video=10., second_shift_audio=3.,
        fast_h3_v2_profile='trained_vsa_exp')
    settings.update(kwargs)
    return runner.DualModelSegmentRunner(object(), object(), **settings)


@pytest.mark.parametrize('first', [True, False])
def test_split_ladder_and_sampler_stage_binding(monkeypatch, first):
    engine = _engine(monkeypatch)
    model = object()
    latent = {'samples': NestedTensor((torch.zeros(1, 24, 2, 2, 3), torch.zeros(1, 32, 2, 5)))}
    # This test audits exact dispatch/binding; tensor initialization is independently covered.
    class Sampler:
        extra_options = {}
    monkeypatch.setattr(v2, 'build_fast_h3_v2_setup',
        lambda m, lat, profile: (m, Sampler(), v2.dmd_sigmas(), '{}'))
    output_model, sampler, sigmas, report = engine._stage_sampling(model, latent, first)
    start, end = (0, 4) if first else (4, 8)
    assert output_model is model
    assert torch.equal(sigmas, v2.dmd_sigmas()[start:end+1])
    assert sampler.extra_options == {'stage_start': start, 'stage_end': end}
    assert json.loads(report)['first_pass_audio_complete'] is False
    assert json.loads(report)['recipe_nfe'] == 8
    assert json.loads(report)['nfe'] == end-start == 4
    assert engine.audio == ('legacy_policy', 0.)


@pytest.mark.parametrize('settings', [dict(coarse_steps=20), dict(refine_steps=3),
    dict(first_shift_video=12.), dict(prompt_relay_mode='apply_exp'),
    dict(fast_h3_v2_profile='official_comfy_template_exp')])
def test_no_mixed_or_reset_recipe(monkeypatch, settings):
    with pytest.raises(ValueError):
        _engine(monkeypatch, **settings)


def test_dense_relay_is_explicit_and_audio_is_unlocked(monkeypatch):
    engine = _engine(monkeypatch, fast_h3_v2_profile='dense_compat_exp', prompt_relay_mode='apply_exp')
    assert engine.audio_policy['first_pass_complete_trajectory'] is False
    assert engine.audio == ('legacy_policy', 0.)


def test_new_schema_is_separate_and_old_defaults_unchanged():
    old = MiniMaxH3DualModelLongVideoEXPT8.define_schema()
    new = MiniMaxH3FastH3V2DualModelLongVideoEXPT8.define_schema()
    old_defaults = {item.id: getattr(item, 'default', None) for item in old.inputs}
    new_defaults = {item.id: getattr(item, 'default', None) for item in new.inputs}
    assert old_defaults['total_duration_seconds'] == 24.
    assert old_defaults['first_shift_video'] == 12.
    assert new_defaults['total_duration_seconds'] == 8.
    assert new_defaults['profile'] == 'trained_vsa_exp'
    assert 'first_shift_video' not in new_defaults and 'coarse_steps' not in new_defaults
    assert old_defaults['chain_id'] == 'h3_dual_model_long_video'
