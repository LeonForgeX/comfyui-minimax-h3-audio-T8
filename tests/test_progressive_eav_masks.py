"""Actual tiny native H3 masked-EAV execution; CPU is not media qualification."""

import copy
import json
from types import SimpleNamespace

import comfy.samplers
import comfy.utils
import pytest
import torch

from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_eav import audit_progressive_eav_stage
from h3_audio_t8_pkg.progressive_eav_masks import NativeProgressiveMaskContract, validate_progressive_eav_mask_scope
from h3_audio_t8_pkg.progressive_masking import normalize_av_masks
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_initialized_sample
from h3_audio_t8_pkg.sampling import native_flow_sigmas
import test_progressive_sampling_runtime as fixtures
from test_progressive_masking import nested, source
from test_progressive_relay import paired

stub_lifter = fixtures.stub_lifter


def masked_source(kind):
    clean = source()
    video, audio = clean.unbind()
    vm, am = torch.ones(1, 1, *video.shape[2:]), torch.ones(1, 1, *audio.shape[2:])
    if kind == 'zero':
        vm.zero_()
        am.zero_()
    elif kind in ('prefix', 'fractional', 'video_only', 'audio_only'):
        if kind != 'audio_only':
            vm[:, :, 0] = 0.
        am[..., :3] = 0.
        if kind == 'fractional':
            vm[..., 1:3] = .123
            am[..., 3:5] = .321
    mask = None if kind == 'none' else vm if kind == 'video_only' else nested((vm, am))
    return {'samples': clean, 'noise_mask': mask}


def execute(model, positive, av, *, partial=False, mode='apply_exp', **kwargs):
    schedule = native_flow_sigmas(8, 12.)[2:] if partial else native_flow_sigmas(8, 12.)
    return runtime.sample_progressive_h3(model, positive, fixtures.conditioning(), av,
        comfy.samplers.ksampler('euler'), schedule, upscaler_model='test', seed=4, low_evaluations=3,
        input_mode='initialized_av_exp', eav_mode=mode, eav_tau=.2, **kwargs)


@pytest.mark.parametrize('kind', ['zero', 'one', 'prefix', 'fractional', 'video_only', 'audio_only', 'none'])
@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('relay', [False, True])
def test_masked_eav_preserves_known_regions_report_only_identity_and_stage_clocks(stub_lifter, kind, partial, relay):
    model = fixtures.tiny_model()
    positive = fixtures.conditioning()
    if relay:
        model, positive, _ = paired(model)
    av = masked_source(kind)
    original = [p.clone() for p in av['samples'].unbind()]
    options = copy.deepcopy(model.model_options)
    baseline, _ = execute(model, positive, av, partial=partial, mode='disabled')
    control, _ = execute(model, positive, av, partial=partial, mode='report_only')
    output, text = execute(model, positive, av, partial=partial)
    for a, b in zip(baseline['samples'].unbind(), control['samples'].unbind()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    if kind != 'zero':
        assert any(not torch.equal(a, b) for a, b in zip(output['samples'].unbind(), control['samples'].unbind()))
    masks = normalize_av_masks(av['noise_mask'], *av['samples'].unbind())
    if masks is not None:
        for actual, expected, mask in zip(output['samples'].unbind(), av['samples'].unbind(), masks.unbind()):
            if bool((mask == 0).any()):
                torch.testing.assert_close(actual[mask == 0], expected[mask == 0], atol=1e-6, rtol=1e-6)
    for a, b in zip(av['samples'].unbind(), original):
        assert torch.equal(a, b)
    assert model.model_options == options
    report = json.loads(text)
    expected_nfe = 6 if partial else 8
    assert report['eav']['summary']['full_schedule_nfe'] == expected_nfe
    for phase, expected in (('low', 3), ('high', expected_nfe - 3)):
        stage = report['eav'][phase]
        assert stage['verified_native_mask_forwards'] == expected
        assert stage['model_forward_count'] == expected
        assert stage['attention_calls_per_active_forward'] == [1] * expected
        assert stage['config']['sampling_profile'] == eav.PROGRESSIVE_INITIALIZED_EAV_PROFILE
        binding = stage['config']['progressive_mask_contract']
        assert all(f['progressive_mask_contract'] == binding for f in stage['forwards'])
        if relay:
            assert report['prompt_relay'][phase]['completed_calls'] == {'forward': expected, 'routed_attention': expected}
    assert report['eav']['high']['forwards'][0]['progress_video'] > report['eav']['low']['forwards'][-1]['progress_video']


@pytest.mark.parametrize('relay', [False, True])
def test_initialized_i2va_eav_and_cancel_reuse(stub_lifter, relay):
    model = fixtures.tiny_model()
    positive = fixtures.conditioning('i2va')
    if relay:
        model, positive, _ = paired(model, 'i2va')
    av = masked_source('prefix')
    original_options = copy.deepcopy(model.model_options)

    def cancel(i, *_):
        if i == 3:
            raise RuntimeError('cancel masked HIGH')

    with pytest.raises(RuntimeError, match='cancel masked HIGH'):
        execute(model, positive, av, task='i2va', callback=cancel)
    assert model.model_options == original_options
    _, text = execute(model, positive, av, task='i2va')
    assert json.loads(text)['eav']['high']['verified_native_mask_forwards'] == 5


def contract_fixture():
    model, av = fixtures.tiny_model(), masked_source('fractional')
    video, audio = av['samples'].unbind()
    mask = normalize_av_masks(av['noise_mask'], video, audio)
    contract = NativeProgressiveMaskContract(model, mask, video.shape, audio.shape)
    expected = model.model._denoise_mask_values(comfy.utils.pack_latents(mask.unbind())[0], (video.shape, audio.shape))
    return contract, (video, audio), expected


@pytest.mark.parametrize('change', ['missing_video', 'missing_audio', 'video', 'audio', 'shape'])
def test_contract_refuses_runtime_mask_drift(change):
    contract, parts, values = contract_fixture()
    if change.startswith('missing'):
        values.pop('denoise_mask' if change == 'missing_video' else 'audio_denoise_mask')
    elif change == 'shape':
        parts = (parts[0][..., :4], parts[1])
    else:
        values['denoise_mask' if change == 'video' else 'audio_denoise_mask'].flatten()[0] = .75
    with pytest.raises(RuntimeError, match='Initialized EAV'):
        contract.validate(parts, values.get('denoise_mask'), values.get('audio_denoise_mask'))


def test_empty_contract_rejects_new_mask_and_reports_are_detached():
    video, audio = source().unbind()
    contract = NativeProgressiveMaskContract(fixtures.tiny_model(), None, video.shape, audio.shape)
    with pytest.raises(RuntimeError, match='unexpected'):
        contract.validate((video, audio), torch.zeros(1, 1, 2, 4, 8), None)
    report = contract.report()
    report['video_shape'][0] = 2
    assert contract.report()['video_shape'][0] == 1


@pytest.mark.parametrize('change', ['profile', 'reference', 'long_video', 'stg', 'task', 'unknown'])
def test_new_contract_does_not_relax_other_composers(change):
    contract, _, _ = contract_fixture()
    kwargs = dict(profile=eav.PROGRESSIVE_INITIALIZED_EAV_PROFILE, allowed_tasks=('T2VA',),
                  reference=False, long_video=None, stg=None)
    if change == 'profile':
        kwargs['profile'] = 'stock20'
    elif change == 'task':
        kwargs['allowed_tasks'] = ('LongVideoMotion',)
    elif change == 'unknown':
        contract = {}
    else:
        kwargs[change] = True
    with pytest.raises(ValueError):
        validate_progressive_eav_mask_scope(contract, **kwargs)


def test_legacy_route_still_refuses_masks_without_explicit_contract():
    video, audio = source().unbind()
    with pytest.raises(RuntimeError, match='rejects video/audio denoise masks'):
        eav._runtime_route(x=(video, audio), timestep=torch.ones(1), context=torch.zeros(1, 2, 8), payload={},
            denoise_mask=torch.zeros(1, 1, 2, 4, 8), audio_denoise_mask=None,
            start_progress=0., end_progress=1.)
    assert eav.EAV_SAMPLING_PROFILES == ('stock20', 'turbo8_alpha8')


@pytest.mark.parametrize('change', ['missing', 'binding', 'spatial'])
def test_stage_audit_refuses_missing_or_wrong_mask_evidence(stub_lifter, change):
    model, av = fixtures.tiny_model(), masked_source('prefix')
    _, text = execute(model, fixtures.conditioning(), av)
    stage = json.loads(text)['eav']['high']
    if change == 'missing':
        stage['forwards'][0].pop('progressive_mask_contract')
    elif change == 'binding':
        stage['forwards'][0]['progressive_mask_contract']['binding_sha256'] = 'wrong'
    else:
        stage['config']['progressive_mask_contract']['video_shape'][-1] *= 2
    plan = plan_progressive_initialized_sample(*av['samples'].unbind(), native_flow_sigmas(8, 12.), low_evaluations=3)
    with pytest.raises(RuntimeError, match='mask'):
        audit_progressive_eav_stage(SimpleNamespace(snapshot=lambda **kwargs: stage), plan, 'high', model)
