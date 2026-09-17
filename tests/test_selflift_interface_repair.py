"""Real tiny Core Euler: repaired optional contracts, not GPU/quality proof."""

import json

import comfy.samplers
import comfy.patcher_extension
import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_progressive_sampling_runtime import conditioning, latent, tiny_model
import test_progressive_sampling_runtime as fixtures

stub_lifter = fixtures.stub_lifter


def run(model, **kwargs):
    return runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
        comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
        upscaler_model='test', seed=19, low_evaluations=4, **kwargs)


def test_configured_same_model_delegates_existing_wrapper_and_matches_legacy(stub_lifter):
    model = tiny_model()
    calls = []
    def prior(original, arguments):
        calls.append(tuple(arguments['input'].shape))
        return original(arguments['input'], arguments['timestep'], **arguments['c'])
    model.set_model_unet_function_wrapper(prior)
    legacy, _ = run(model)
    assert len(calls) == 8
    calls.clear()
    configured, text = run(model, model_hires=model)
    assert len(calls) == 8
    assert model.model_options['model_function_wrapper'] is prior
    for left, right in zip(legacy['samples'].unbind(), configured['samples'].unbind(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert json.loads(text)['counts']['actual_forwards'] == {'low': 4, 'high': 4}


def test_distinct_stage_wrappers_and_stacked_weight_patches_are_preserved(stub_lifter):
    low, high = tiny_model(), tiny_model()
    calls = []
    def owner(phase):
        def wrapper(original, arguments):
            calls.append(phase)
            return original(arguments['input'], arguments['timestep'], **arguments['c'])
        return wrapper
    owners = {'low': owner('low'), 'high': owner('high')}
    low.set_model_unet_function_wrapper(owners['low'])
    high.set_model_unet_function_wrapper(owners['high'])
    key, weight = next(iter(high.model.named_parameters()))
    high.add_patches({key: ('diff', (torch.full_like(weight, .01),))}, .7)
    high.add_patches({key: ('diff', (torch.full_like(weight, .02),))}, .2)
    descriptors = list(high.patches[key])
    output, text = run(low, model_hires=high)
    assert calls == ['low'] * 4 + ['high'] * 4
    assert high.patches[key] == descriptors
    assert low.model_options['model_function_wrapper'] is owners['low']
    assert high.model_options['model_function_wrapper'] is owners['high']
    assert json.loads(text)['stage_models']['high']['weight_patch_entries'] == 2
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())


def test_real_high_wrapper_exception_propagates_and_original_can_be_reused(stub_lifter):
    low, high = tiny_model(), tiny_model()
    calls = []
    def broken(original, arguments):
        calls.append('actual_high_owner')
        raise RuntimeError('actual foreign wrapper failure')
    high.set_model_unet_function_wrapper(broken)
    with pytest.raises(RuntimeError, match='actual foreign wrapper failure'):
        run(low, model_hires=high)
    assert calls == ['actual_high_owner']
    assert high.model_options['model_function_wrapper'] is broken
    assert not low.wrappers and not high.wrappers
    run(low)


def test_configured_route_preserves_user_latent_metadata(stub_lifter):
    model = tiny_model()
    source = latent()
    selected = object()
    source['user_selected_metadata'] = selected
    output, _ = runtime.sample_progressive_h3(model, conditioning(), conditioning(), source,
        comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
        upscaler_model='test', seed=19, low_evaluations=4, model_hires=model)
    assert output['user_selected_metadata'] is selected
    assert source['user_selected_metadata'] is selected


def test_unknown_keyword_is_not_silently_ignored(stub_lifter):
    with pytest.raises(TypeError, match='imaginary_checkpoint'):
        run(tiny_model(), imaginary_checkpoint=object())
    assert not stub_lifter


@pytest.mark.parametrize('tst,eav', [(True, False), (False, True), (True, True)])
def test_user_non_delegating_apply_owner_warns_without_false_feature_evidence(stub_lifter, caplog, tst, eav):
    model = tiny_model()
    calls = []
    def bypass(executor, x, *args, **kwargs):
        calls.append(tuple(x.shape))
        return torch.zeros_like(x)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, 'user_bypass', bypass)
    output, text = run(model, model_hires=model,
        tst_mode='report_only' if tst else 'disabled', eav_mode='report_only' if eav else 'disabled')
    report = json.loads(text)
    assert len(calls) == 8
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 0}
    assert report['counts']['forward_evidence_complete'] is False
    assert model.wrappers[comfy.patcher_extension.WrappersMP.APPLY_MODEL]['user_bypass'] == [bypass]
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    if tst:
        assert all(not report['tst'][phase]['completed'] for phase in ('low', 'high'))
    if eav:
        assert report['eav']['summary']['composition_verified'] is False
    assert 'advisory' in caplog.text


def test_bypassed_low_observer_does_not_publish_false_checkpoint(tmp_path, stub_lifter, caplog):
    from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession
    model = tiny_model()
    def bypass(executor, x, *args, **kwargs):
        return torch.zeros_like(x)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, 'user_bypass', bypass)
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        output, text = run(model, checkpoint=checkpoint, tst_mode='report_only')
    report = json.loads(text)
    assert report['checkpoint']['saved_low'] is False
    assert report['checkpoint']['save_status'] == 'incomplete_forward_evidence_not_cached'
    assert not list(tmp_path.glob('low-boundary-*.json'))
    assert all(torch.isfinite(part).all() for part in output['samples'].unbind())
    assert 'advisory' in caplog.text


def test_job_foreign_delegating_owner_keeps_selection_and_nonportable_identity(stub_lifter):
    from test_progressive_producers import component
    from h3_audio_t8_pkg.progressive_job import NativeProgressiveJob
    model = tiny_model()
    def selected(original, arguments):
        return original(arguments['input'], arguments['timestep'], **arguments['c'])
    model.set_model_unet_function_wrapper(selected)
    value = NativeProgressiveJob(model, model, comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
        **{role: component(role) for role in ('clip', 'video_vae', 'audio_vae')},
        chain_id='foreign_owner', total_duration_seconds=8, width=128, height=64,
        upscaler_model='test', global_prompt='Continue walking.', base_seed=8)
    identity = value.sha256
    assert value.verify() == value.verify() == identity
    assert model.model_options['model_function_wrapper'] is selected
    assert all(m['weights']['portable_cache_reuse'] is False for m in value.identity['models'])
    with torch.no_grad():
        next(model.model.parameters()).add_(.01)
    with pytest.raises(ValueError, match='contract changed'):
        value.verify()
