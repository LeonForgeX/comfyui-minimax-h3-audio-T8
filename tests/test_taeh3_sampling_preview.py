from types import SimpleNamespace
import base64
from io import BytesIO
import json

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.utils
import pytest
import torch

from h3_audio_t8_pkg import taeh3_sampling_preview as preview
from h3_audio_t8_pkg.preview_execution_context import CONTEXT, preview_scope
from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
import test_progressive_sampling_runtime as fixtures


def settings(**kwargs):
    return preview.PreviewSettings(model_path='missing-test-file', min_interval_ms=0, **kwargs)


@pytest.fixture
def environment(monkeypatch):
    events = []
    context = SimpleNamespace(prompt_id='actual-prompt', node_id='sampler-node')
    server = SimpleNamespace(send_sync=lambda event, value, client: events.append((event, value, client)))
    monkeypatch.setattr(preview, '_environment', lambda: (context, server, 'only-this-client'))
    monkeypatch.setattr(preview, '_check', lambda: None)
    return events


def decoder_stub(monkeypatch):
    class Stub:
        def __init__(self, *args):
            pass
        def decode(self, value):
            return torch.zeros(5, 32, 32, 3)
    monkeypatch.setattr(preview, 'TinyDecoder', Stub)


def test_disabled_and_no_client_allocate_nothing(monkeypatch):
    model = fixtures.tiny_model()
    assert preview.attach_preview(model, None, 'p', False) is model
    monkeypatch.setattr(preview, '_environment', lambda: None)
    wrapper = preview.TAEH3PreviewWrapper(settings(), 'p')
    assert wrapper(lambda value: value, model) is model


def test_packed_and_nested_x0_choose_only_video():
    av = fixtures.latent()['samples']
    assert preview.video_prediction(av) is av.unbind()[0]
    packed, shapes = comfy.utils.pack_latents(av.unbind())
    assert torch.equal(preview.video_prediction(packed, shapes), av.unbind()[0])
    with pytest.raises(ValueError):
        preview.video_prediction(av.unbind()[1])


@pytest.mark.parametrize('prefix', [2, 7, 12])
def test_prefix_is_bounded_detached_and_input_unchanged(prefix):
    source = torch.ones(1, 24, 37, 48, 32, requires_grad=True)
    before = source.detach().clone()
    value, info = preview.bounded_prefix(source, settings(latent_prefix=prefix, max_resolution=64))
    assert value.shape == (1, 24, prefix, 4, 2)
    value.zero_()
    assert torch.equal(source, before) and not value.requires_grad
    assert info['approximate'] and info['spatially_resampled']


def test_actual_callbacks_preserved_and_private_events(environment, monkeypatch):
    decoder_stub(monkeypatch)
    source = fixtures.latent()['samples']
    before = [part.clone() for part in source.unbind()]
    rng = torch.get_rng_state().clone()
    calls = []
    def run(*args, callback=None):
        for step in range(4):
            callback(step, source, source, 4)
        return source
    with preview_scope(phase='low', global_offset=3, global_total=8):
        result = preview.TAEH3PreviewWrapper(settings(), 'p')(run, callback=lambda *args: calls.append(args))
    assert result is source and len(calls) == 4
    assert all(call[1] is source and call[2] is source for call in calls)
    assert all(torch.equal(a, b) for a, b in zip(source.unbind(), before))
    assert torch.equal(rng, torch.get_rng_state()) and CONTEXT.get() is None
    assert [row[1]['kind'] for row in environment] == ['start', 'frames', 'frames', 'frames', 'end']
    assert all(row[2] == 'only-this-client' for row in environment)
    assert environment[-2][1]['global_step'] == 7


def test_failure_disables_observer_not_sampling(environment):
    wrapper = preview.TAEH3PreviewWrapper(settings(), 'p')
    source = fixtures.latent()['samples']
    calls = []
    def run(*args, callback=None):
        for step in range(3):
            callback(step, source, source, 3)
        return source
    assert wrapper(run, callback=lambda *args: calls.append(args)) is source
    assert len(calls) == 3
    assert [row[1]['kind'] for row in environment] == ['start', 'unavailable', 'end']
    assert environment[-1][1]['unavailable']


def test_cancel_propagates_then_same_wrapper_recovers(environment, monkeypatch):
    decoder_stub(monkeypatch)
    wrapper = preview.TAEH3PreviewWrapper(settings(), 'p')
    source = fixtures.latent()['samples']
    def run(*args, callback=None):
        callback(0, source, source, 1)
        return source
    def cancel(*args):
        raise mm.InterruptProcessingException()
    with pytest.raises(mm.InterruptProcessingException):
        wrapper(run, callback=cancel)
    assert environment[-1][1]['status'] == 'cancelled'
    assert wrapper(run) is source
    assert environment[-1][1]['status'] == 'finished'
    assert environment[0][1]['run_id'] != environment[-1][1]['run_id']


def test_high_filter_does_not_call_decoder_or_fake_callbacks(environment):
    with preview_scope(phase='high', global_offset=4):
        assert preview.TAEH3PreviewWrapper(settings(), 'p')(lambda: 'same') == 'same'
    assert environment == []


def test_exact_observer_has_no_cache_identity_effect_foreign_owner_kept():
    model = fixtures.tiny_model()
    original = stage_model_identity(model)
    clone = preview.attach_preview(model, settings(), 'panel')
    assert stage_model_identity(clone) == original
    assert not model.wrappers and clone.wrappers
    def fake(*args):
        return None
    from comfy.patcher_extension import WrappersMP
    clone.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'foreign', fake)
    projected = preview.cache_projection(clone)
    assert projected.wrappers[WrappersMP.OUTER_SAMPLE]['foreign'] == [fake]
    assert stage_model_identity(projected)['schema'].endswith('user_stack_v1')


def test_real_installed_native_taeh3_decoder_cpu_no_rng():
    import folder_paths
    path = folder_paths.get_full_path('vae_approx', 'taeh3.safetensors')
    assert path, 'This qualification requires the already installed tiny checkpoint'
    rng = torch.get_rng_state().clone()
    decoder = preview.TinyDecoder(path, torch.device('cpu'))
    images = decoder.decode(torch.zeros(1, 24, 2, 4, 4))
    assert images.shape == (5, 64, 64, 3) and torch.isfinite(images).all()
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('phase', ['low', 'all'])
def test_actual_native_Euler_preview_is_bit_identical_and_stage_labelled(environment, monkeypatch, phase, stub_lifter):
    import comfy.samplers
    from h3_audio_t8_pkg.progressive_sampling_runtime import sample_progressive_h3
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    decoder_stub(monkeypatch)
    model = fixtures.tiny_model()
    args = (fixtures.conditioning(), fixtures.conditioning(), fixtures.latent(),
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.))
    options = dict(upscaler_model='test', seed=123, cfg=1., low_evaluations=4)
    before, first_report = sample_progressive_h3(model, *args, **options)
    observed = preview.attach_preview(model, settings(phase=phase), 'panel')
    after, next_report = sample_progressive_h3(observed, *args, **options)
    assert all(torch.equal(a, b) for a, b in zip(before['samples'].unbind(), after['samples'].unbind()))
    assert json.loads(first_report)['counts'] == json.loads(next_report)['counts']
    frames = [item[1] for item in environment if item[1]['kind'] == 'frames']
    assert {item['phase'] for item in frames} == ({'low'} if phase == 'low' else {'low', 'high'})
    assert [item['global_step'] for item in frames] == ([1, 3, 4] if phase == 'low' else [1, 3, 4, 5, 7, 8])
    assert not model.wrappers and CONTEXT.get() is None


def test_foreign_callable_cannot_hide_under_our_cache_key():
    from comfy.patcher_extension import WrappersMP
    model = fixtures.tiny_model()
    def foreign(*args):
        return None
    model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, preview.OWNER + ':panel', foreign)
    assert preview.cache_projection(model) is model
    assert stage_model_identity(model)['schema'].endswith('user_stack_v1')


stub_lifter = fixtures.stub_lifter


@pytest.mark.parametrize('source_size, expected', [((16, 24), (85, 128)), ((24, 16), (128, 85)), ((54, 74), (93, 128))])
def test_jpeg_restores_source_aspect_only_not_sampler_or_decoded_images(monkeypatch, source_size, expected):
    from PIL import Image
    monkeypatch.setattr(preview, '_check', lambda: None)
    images = torch.zeros(5, 128, 64, 3)
    before = images.clone()
    frames, indices = preview._jpeg_frames(images, settings(max_resolution=128), source_size)
    assert len(frames) == len(indices) == 5 and torch.equal(images, before)
    for frame in frames:
        with Image.open(BytesIO(base64.b64decode(frame.split(',', 1)[1]))) as picture:
            assert picture.size == expected and max(picture.size) <= 128
