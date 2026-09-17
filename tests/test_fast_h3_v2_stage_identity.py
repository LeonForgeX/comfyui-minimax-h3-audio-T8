"""V2 cache ownership identities, not generation/quality/performance proof."""
from types import SimpleNamespace

import pytest
import torch

from comfy.ldm.modules import attention
from comfy.patcher_extension import CallbacksMP
from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg import h3_memory_advanced as memory
from h3_audio_t8_pkg import long_video_dual_identity as identity
from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
from test_fast_h3_v2_memory_bridge import _gated_model
from test_long_video_dual_model_stages import latent


def _setup(source=None, profile="trained_vsa_exp"):
    model = _gated_model() if source is None else source
    model.model.model_config = SimpleNamespace(sampling_settings={}, unet_config={"tiny": True})
    return v2.build_fast_h3_v2_setup(model, latent(0.0), profile=profile)[0]


@pytest.mark.parametrize("profile", v2.PROFILES)
def test_actual_setup_sampling_profiles_are_content_bound_and_clone_stable(profile):
    model = _setup(profile=profile)
    first = identity.stage_model_identity(model)
    assert first["schema"] == "t8.h3.dual_stage_model/v2"
    assert first["sha256"] == identity.stage_model_identity(model.clone())["sha256"]
    contract = first["fast_h3_v2"]
    assert contract["profile"] == profile
    assert contract["rungs"] == list(v2.RUNG_STEPS)
    assert contract["model_sampling"]["configuration"]["shift"] == 10.0
    assert contract["model_sampling"]["configuration"]["noise_scale"] == 1.0
    if profile == "official_comfy_template_exp":
        assert contract["model_sampling"]["configuration"]["audio_shift"] == 3.0
        assert contract["model_sampling"]["protocol"] == "native_av_carrier"
    else:
        assert contract["model_sampling"]["protocol"] == "t8_raw_audio_flow"
    assert model.get_attachment(v2.KEY) is not None
    assert "model_sampling" in model.object_patches


def test_runtime_counts_and_pooled_state_do_not_change_cache_identity():
    model = _setup()
    before = identity.stage_model_identity(model)["sha256"]
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    runtime.counts["vsa"] = 123
    runtime.dense_reasons["short_sequence"] = 99
    runtime.steps["0.9"] = {"dense": 1}
    runtime.closed = False
    runtime.patch.pooled[(0, 64, ())] = (torch.ones(3, 128), torch.zeros(3, 128))
    assert identity.stage_model_identity(model)["sha256"] == before


def test_actual_v2_memory_configuration_and_original_plain_backend_are_preserved():
    source = _gated_model()
    set_h3_attention_backend(source, attention.attention_pytorch)
    source, _ = memory.configure_low_vram_attention(source, 2)
    source, _ = memory.configure_chunk_feed_forward(source, 2, 256)
    model = _setup(source)
    first = identity.stage_model_identity(model)
    assert first["backend"]["kind"] == "core_plain"
    assert first["backend"]["name"] == "pytorch"
    assert first["memory"]["head_chunks"] == first["fast_h3_v2"]["head_chunks"] == 2
    assert first["memory"]["ffn_settings"] == [2, 256]
    assert model.get_wrappers("diffusion_model", v2.KEY)
    assert len(model.object_patches) == 4


def test_profiles_rungs_sampling_buffers_and_unsampled_weight_bytes_invalidate_identity(monkeypatch):
    source = _gated_model()
    a, b = _setup(source), _setup(source, "dense_compat_exp")
    first = identity.stage_model_identity(a)["sha256"]
    assert identity.stage_model_identity(b)["sha256"] != first
    sampler = a.object_patches["model_sampling"]
    sampler.noise_scale = 1.2
    changed = identity.stage_model_identity(a)["sha256"]
    assert changed != first
    sampler.sigmas[57] += 0.001
    assert identity.stage_model_identity(a)["sha256"] != changed
    changed = identity.stage_model_identity(a)["sha256"]
    with torch.no_grad():
        a.model.diffusion_model.blocks[0].attn.to_gate_compress.weight.view(-1)[57] += 0.125
    assert identity.stage_model_identity(a)["sha256"] != changed
    changed = identity.stage_model_identity(a)["sha256"]
    monkeypatch.setattr(v2, "RUNG_STEPS", (998, *v2.RUNG_STEPS[1:]))
    assert identity.stage_model_identity(a)["sha256"] != changed


@pytest.mark.parametrize("kind", ["attachment", "callback", "wrapper", "object_patch"])
def test_v2_does_not_wash_unknown_owners(kind):
    model = _setup()
    def foreign(*a, **k):
        return None
    if kind == "attachment":
        model.set_attachments("foreign_attachment", {"claimed_safe": True})
    elif kind == "callback":
        model.add_callback_with_key(CallbacksMP.ON_CLEANUP, "foreign_callback", foreign)
    elif kind == "wrapper":
        model.add_wrapper_with_key("diffusion_model", "foreign_wrapper", foreign)
    else:
        model.object_patches["diffusion_model.blocks.0.attn.forward"] = foreign
    result = identity.stage_model_identity(model)
    assert result["portable_cache_reuse"] is False
    assert result["sha256"] != identity.stage_model_identity(model)["sha256"]


def test_copied_marker_is_not_a_v2_owner():
    model = _gated_model()
    model.set_attachments(v2.KEY, {"schema": v2.SCHEMA, "profile": "trained_vsa_exp"})
    with pytest.raises(RuntimeError, match="foreign or malformed"):
        identity.stage_model_identity(model)


def test_receipt_and_hook_identity_coordinated_spoof_cannot_hide_foreign_factory():
    model = _setup()
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    def foreign(*a, **k):
        return None
    runtime.guard = foreign
    model.wrappers["diffusion_model"][v2.KEY] = [foreign]
    # Superficially same function identities now pass the owner's mutable
    # reference checks. The identity adapter independently checks its factory.
    with pytest.raises(ValueError, match="foreign execution factory"):
        identity.stage_model_identity(model)


def test_native_dense_receipt_does_not_authorize_an_unknown_dit():
    source = _gated_model()
    source.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): lambda *a: None}}
    model = v2._install_runtime(source, "dense_compat_exp", 0)
    result = identity.stage_model_identity(model)
    assert result["portable_cache_reuse"] is False
    assert result["sha256"] != identity.stage_model_identity(model)["sha256"]


@pytest.mark.parametrize("mutation", ["field", "method", "hook", "foreign_object"])
def test_model_sampling_requires_precise_inert_native_schema(mutation):
    model = _setup()
    sampler = model.object_patches["model_sampling"]
    if mutation == "field":
        sampler.unknown_runtime_state = 1
    elif mutation == "method":
        sampler.noise_scaling = lambda *a: None
    elif mutation == "hook":
        sampler.register_forward_pre_hook(lambda *a: None)
    else:
        model.object_patches["model_sampling"] = SimpleNamespace(shift=10, noise_scale=1)
    result = identity.stage_model_identity(model)
    assert result["portable_cache_reuse"] is False
    assert result["sha256"] != identity.stage_model_identity(model)["sha256"]


def test_loaded_sampling_backup_uses_nonportable_identity_without_shared_unpatch():
    model = _setup()
    original = model.model.model_sampling
    model.object_patches_backup["model_sampling"] = original
    assert identity.stage_model_identity(model)["portable_cache_reuse"] is False
    assert model.model.model_sampling is original


@pytest.mark.parametrize("profile", v2.PROFILES)
@pytest.mark.parametrize("kind", ["override", "dit", "extra_dit"])
def test_later_selected_owners_are_not_washed_into_portable_v2_identity(profile, kind):
    model = _setup(profile=profile)
    live = model.model_options["transformer_options"]
    calls = []
    def foreign(*args, **kwargs):
        calls.append(True)
        return None
    if kind == "override":
        live["optimized_attention_override"] = foreign
    else:
        key = ("double_block", 0 if kind == "dit" else 999)
        live.setdefault("patches_replace", {}).setdefault("dit", {})[key] = foreign
    first = identity.stage_model_identity(model)
    second = identity.stage_model_identity(model)
    assert first["portable_cache_reuse"] is second["portable_cache_reuse"] is False
    assert first["sha256"] != second["sha256"]
    assert not calls
    assert (live["optimized_attention_override"] if kind == "override" else
            live["patches_replace"]["dit"][key]) is foreign


def test_existing_bare_identity_schema_is_unchanged():
    model = _gated_model()
    result = identity.stage_model_identity(model)
    assert result["schema"] == "t8.h3.dual_stage_model/v1"
    assert "fast_h3_v2" not in result
