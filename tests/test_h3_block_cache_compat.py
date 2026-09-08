"""Cache finalization call contract, independently of an installed cache plugin."""
from types import SimpleNamespace
import inspect

import pytest
import torch
from comfy.model_management import InterruptProcessingException

from h3_audio_t8_pkg.h3_block_cache_compat import cache_finalization_executor


class Executor:
    def __init__(self, model):
        self.class_obj = model
        self.wrappers = [object()]
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 'delegate-result'


@pytest.mark.parametrize('new_core', [False, True])
def test_finalization_adapts_only_call_interface_without_mutating_shared_model(new_core):
    calls = []
    if new_core:
        def layer(x, t, video, audio, sigma, sample_sigmas, shifts):
            calls.append((x, t, video, audio, sigma, sample_sigmas, shifts))
            return 'rows'
    else:
        def layer(x, t, video, audio):
            calls.append((x, t, video, audio))
            return 'rows'
    model = SimpleNamespace(final_layer=layer, sigma_shift_video=12., sigma_shift_audio=3., blocks=object())
    executor = Executor(model)
    schedule = torch.tensor([1., .5, 0.])
    options = {'sample_sigmas': schedule, 'minimax_h3_sigma_shift_video': 9., 'minimax_h3_sigma_shift_audio': 2.}
    view = cache_finalization_executor(executor, torch.tensor([500.]), options)
    assert view('input', flag=True) == 'delegate-result'
    assert executor.calls == [(('input',), {'flag': True})]
    assert view.wrappers is executor.wrappers
    assert view.class_obj.blocks is model.blocks
    assert view.class_obj.final_layer(1, 2, 3, 4) == 'rows'
    assert calls[0][:4] == (1, 2, 3, 4)
    if new_core:
        assert float(calls[0][4]) == .5
        assert calls[0][5] is schedule
        assert calls[0][6] == (9., 2.)
    assert model.final_layer is layer
    assert executor.class_obj is model
    assert options['sample_sigmas'] is schedule


def test_already_updated_external_finalizer_arguments_pass_through_exactly():
    calls = []
    def layer(*args, **kwargs):
        calls.append((args, kwargs))
        return 'rows'
    model = SimpleNamespace(final_layer=layer)
    executor = Executor(model)
    view = cache_finalization_executor(executor, None, {})
    sigma, schedule, shifts = object(), object(), object()
    assert view.class_obj.final_layer(1, 2, 3, 4, sigma, schedule, shifts) == 'rows'
    assert calls == [((1, 2, 3, 4, sigma, schedule, shifts), {})]
    assert model.final_layer is layer


@pytest.mark.parametrize('exception', [RuntimeError, InterruptProcessingException, KeyboardInterrupt])
def test_error_and_cancel_propagate_without_patching_shared_layer(exception):
    def layer(x, t, video, audio):
        raise exception('controlled-finalization-stop')
    model = SimpleNamespace(final_layer=layer, sigma_shift_video=12., sigma_shift_audio=3.)
    executor = Executor(model)
    view = cache_finalization_executor(executor, torch.tensor([500.]), {})
    with pytest.raises(exception, match='controlled-finalization-stop'):
        view.class_obj.final_layer(1, 2, 3, 4)
    assert model.final_layer is layer and executor.class_obj is model


def test_actual_core_pdd_head_bank_receives_exact_schedule_and_audio_shift():
    import comfy.ops
    from comfy.ldm.minimax.model import FinalLayer
    if 'sample_sigmas' not in inspect.signature(FinalLayer.forward).parameters:
        pytest.skip('this historical Core predates schedule-dependent PDD heads')
    torch.manual_seed(839)
    layer = FinalLayer(8, 4, 96, 32, 1e-6, dtype=torch.float32, device='cpu',
                       operations=comfy.ops.disable_weight_init)
    for parameter in layer.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    for head in (layer.video_out, layer.audio_out):
        head.weight = torch.nn.Parameter(torch.cat((head.weight.detach(), .3 * head.weight.detach())))
        head.bias = torch.nn.Parameter(torch.cat((head.bias.detach(), .3 * head.bias.detach())))
    layer.requires_grad_(False)
    model = SimpleNamespace(final_layer=layer, sigma_shift_video=12., sigma_shift_audio=3.)
    schedule = torch.tensor([1., .75, .25, 0.])
    options = {'sample_sigmas': schedule, 'minimax_h3_sigma_shift_video': 1., 'minimax_h3_sigma_shift_audio': 3.}
    view = cache_finalization_executor(Executor(model), torch.tensor([750.]), options)
    x, t = torch.randn(8, 8), torch.randn(2, 4)
    video, audio = (4, 8, 0), (2, 4, 1)
    with torch.no_grad():
        expected = layer(x, t, video, audio, torch.tensor(.75), schedule, (1., 3.))
        actual = view.class_obj.final_layer(x, t, video, audio)
        wrong_audio_shift = layer(x, t, video, audio, torch.tensor(.75), schedule, (1., 1.))
    for reference, result in zip(expected, actual):
        torch.testing.assert_close(reference, result, rtol=0, atol=0)
    assert not torch.allclose(expected[1], wrong_audio_shift[1])
    assert model.final_layer is layer
