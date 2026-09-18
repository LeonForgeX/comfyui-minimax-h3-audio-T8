"""Known 2D tiny graph and private events, without changing sampler math."""
from types import SimpleNamespace

import comfy.model_management as mm
import pytest
from safetensors.torch import save_file
import torch
from torch import nn

from h3_audio_t8_pkg import taeh3_sampling_preview as preview
from h3_audio_t8_pkg.taeh3_2d_preview import build_decoder, decoder_state, decode_frames


def checkpoint_state():
    # Independent expected-shape fixture; initialization uses no global RNG.
    state = {'1.weight': torch.full((96, 24, 3, 3), .00001),
             '1.bias': torch.full((96,), .001),
             '23.weight': torch.full((3, 64, 3, 3), .00001),
             '23.bias': torch.full((3,), .001)}
    for index, width in ((7, 96), (12, 96), (17, 64), (21, 64)):
        state[f'{index}.weight'] = torch.full((width, width, 3, 3), .00001)
    for index in (3, 4, 5, 8, 9, 10, 13, 14, 15, 18, 19, 22):
        width = 96 if index <= 10 else 64
        n_in = 96 if index == 13 else width
        for layer in (0, 2, 4):
            state[f'{index}.conv.{layer}.weight'] = torch.full(
                (width, n_in if layer == 0 else width, 3, 3), .00001)
            state[f'{index}.conv.{layer}.bias'] = torch.full((width,), .001)
        if n_in != width:
            state[f'{index}.skip.weight'] = torch.full((width, n_in, 1, 1), .00001)
    return state


@pytest.mark.parametrize('prefix', ['', 'decoder.', 'taesd_decoder.'])
def test_known_complete_layout_loads_without_rng_or_input_mutation(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(preview, '_check', lambda: None)
    state = checkpoint_state()
    file = tmp_path/'taeh3_2d.safetensors'
    save_file({prefix+k: v for k, v in state.items()}, str(file))
    rng = torch.get_rng_state().clone()
    decoder = preview.TinyDecoder(file, torch.device('cpu'))
    assert decoder.family == 'independent_2d_latent_frames'
    value = torch.linspace(-.2, .2, 1*24*2*2*4).reshape(1, 24, 2, 2, 4)
    before = value.clone()
    images = decoder.decode(value)
    assert images.shape == (2, 32, 64, 3) and torch.isfinite(images).all()
    direct = build_decoder(state)
    expected = torch.cat([direct(value[:, :, i]).movedim(1, -1) for i in range(2)])
    torch.testing.assert_close(images, expected, rtol=0, atol=0)
    assert torch.equal(value, before) and torch.equal(rng, torch.get_rng_state())
    assert sum(isinstance(m, nn.Upsample) for m in direct) == 4
    assert direct[13].conv[0].weight.shape[:2] == (64, 96)


@pytest.mark.parametrize('change', ['missing', 'extra', 'input_channels', 'output_channels', 'nonfinite', 'integer'])
def test_mismatched_decoder_cannot_be_guessed(change):
    state = checkpoint_state()
    if change == 'missing':
        state.pop('21.weight')
    elif change == 'extra':
        state['99.weight'] = torch.zeros(1)
    elif change == 'input_channels':
        state['1.weight'] = torch.zeros(96, 32, 3, 3)
    elif change == 'output_channels':
        state['23.bias'] = torch.zeros(12)
    elif change == 'nonfinite':
        state['1.bias'][0] = float('nan')
    else:
        state['1.bias'] = state['1.bias'].long()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='H3 2D tiny decoder'):
        build_decoder(state)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('state', [{}, {'encoder.1.weight': torch.zeros(1)},
                                  {'decoder.1.weight': torch.zeros(1), '1.bias': torch.zeros(1)}])
def test_unknown_mixed_or_empty_prefixes_rejected(state):
    with pytest.raises(ValueError):
        decoder_state(state)


def test_serial_frame_cancel_and_recovery_keep_input_unchanged():
    model = build_decoder(checkpoint_state())
    seen = []
    hook = model.register_forward_pre_hook(lambda _m, args: seen.append(tuple(args[0].shape)))
    value = torch.zeros(1, 24, 7, 2, 2)
    before = value.clone()
    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 3:
            raise mm.InterruptProcessingException()

    try:
        with pytest.raises(mm.InterruptProcessingException):
            decode_frames(model, value, torch.device('cpu'), torch.float32, cancel)
        assert seen == [(1, 24, 2, 2)]
        images = decode_frames(model, value, torch.device('cpu'), torch.float32, lambda: None)
        assert images.shape == (7, 32, 32, 3)
        assert seen == [(1, 24, 2, 2)]*8 and torch.equal(value, before)
    finally:
        hook.remove()


@pytest.mark.parametrize('shape', [(1, 24, 13, 2, 2), (2, 24, 2, 2, 2),
                                   (1, 32, 2, 2, 2), (1, 24, 2, 33, 2)])
def test_unbounded_input_refused_before_decoder(shape):
    with pytest.raises(ValueError, match='bounded'):
        decode_frames(lambda _: pytest.fail('unbounded decode'), torch.zeros(shape),
                      torch.device('cpu'), torch.float32, lambda: None)


def test_private_event_2d_indices_do_not_claim_24fps(monkeypatch):
    events = []
    context = SimpleNamespace(prompt_id='only-this-prompt', node_id='sampler')
    server = SimpleNamespace(send_sync=lambda event, value, client: events.append((value, client)))
    monkeypatch.setattr(preview, '_environment', lambda: (context, server, 'private-client'))
    monkeypatch.setattr(preview, '_check', lambda: None)

    class Stub:
        family = 'independent_2d_latent_frames'

        def __init__(self, *_):
            pass

        def decode(self, _):
            return torch.zeros(7, 32, 32, 3)

    monkeypatch.setattr(preview, 'TinyDecoder', Stub)
    value = torch.zeros(1, 24, 7, 2, 2)
    original = []

    def sample(callback=None):
        callback(0, value, value, 1)
        return value

    wrapper = preview.TAEH3PreviewWrapper(preview.PreviewSettings(model_path='test', frames=3), 'panel')
    assert wrapper(sample, callback=lambda *args: original.append(args)) is value
    frame = events[1][0]
    assert frame['decoder_family'] == Stub.family and frame['source_prefix_fps'] is None
    assert frame['decoded_latent_frame_indices'] == frame['decoded_frame_indices'] == [0, 3, 6]
    assert 'not reconstructed pixel-frame timing' in frame['coverage']
    assert len(original) == 1 and original[0][1] is value
    assert all(client == 'private-client' for _, client in events)
