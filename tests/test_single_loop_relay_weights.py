"""Real ModelPatcher regressions for internal Relay + deferred LoRA composition."""
import uuid

import pytest
import torch

from h3_audio_t8_pkg import long_video_in_node_loop_effects_advanced as loop
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from test_prompt_relay_core_compat import bound_layout, model_fixture


def weighted_model():
    model = model_fixture()
    model.patches = {"x": [(1., (torch.ones(2),), 1., None, None)]}
    return model


def test_internal_binding_restores_weights_without_mutating_source(monkeypatch):
    source = weighted_model()
    binding, _ = bound_layout("joint_av_exp")
    positive, latent, receipt = object(), object(), object()
    original_uuid = source.patches_uuid

    def builder(model, semantic_bridge):
        assert not model.patches
        assert semantic_bridge is receipt
        bound, _ = relay.patch_prompt_relay_model(model, binding, 32)
        return bound, positive, latent, receipt

    monkeypatch.setattr(loop, "build_prompt_relay_long_video_conditioning", builder)
    bound, out_positive, out_latent, out_receipt = loop._bind_loop_relay_conditioning(
        source, semantic_bridge=receipt)
    assert bound is not source and bound.model is source.model
    assert (out_positive, out_latent, out_receipt) == (positive, latent, receipt)
    assert bound.patches_uuid == original_uuid == source.patches_uuid
    assert bound.patches == source.patches
    assert bound.patches is not source.patches
    assert bound.patches["x"] is not source.patches["x"]
    assert relay.prompt_relay_model_contract(bound)["binding_hash"] == binding["binding_hash"]
    assert not source.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)


@pytest.mark.parametrize("source", [None, "unweighted"])
def test_original_unweighted_route_keeps_object_identity(monkeypatch, source):
    source = object() if source is None else model_fixture()
    expected = (object(), object())

    def builder(model, **kwargs):
        assert model is source
        assert kwargs == {"length": 124}
        return expected

    monkeypatch.setattr(loop, "build_prompt_relay_long_video_conditioning", builder)
    assert loop._bind_loop_relay_conditioning(source, length=124) is expected


@pytest.mark.parametrize("failure", ["exception", "base", "weights", "mutation"])
def test_binding_does_not_hide_incompatible_builders(monkeypatch, failure):
    source = weighted_model()
    original_patches = source.patches

    def builder(model):
        if failure == "exception":
            raise ValueError("builder failed")
        if failure == "base":
            return (model_fixture(),)
        if failure == "weights":
            model.patches = {"unexpected": []}
        if failure == "mutation":
            source.patches_uuid = uuid.uuid4()
        return (model,)

    monkeypatch.setattr(loop, "build_prompt_relay_long_video_conditioning", builder)
    with pytest.raises((ValueError, RuntimeError)):
        loop._bind_loop_relay_conditioning(source)
    assert source.patches is original_patches


def test_public_relay_warns_and_preserves_prebound_weights(caplog):
    source = weighted_model()
    original_patches = source.patches
    original_uuid = source.patches_uuid
    relay._assert_core_contract(source)
    assert source.patches is original_patches
    assert source.patches_uuid == original_uuid
    assert "weight patches" in caplog.text
    assert "continuing" in caplog.text
