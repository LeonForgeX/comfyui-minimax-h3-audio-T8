from collections import namedtuple

import pytest
import torch

from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity, content_identity
from test_relay_kj_memory import small_model, patched_source, memory_nodes  # noqa: F401
from test_relay_kj_backend import kj  # noqa: F401


def test_same_base_different_lora_strength_or_order_changes_identity():
    a = small_model()
    tensor = torch.ones(3)
    a.patches = {"diffusion_model.blocks.0.attn.qkv_proj.weight": [(1., ("lora", (tensor,)), 1., None, None)]}
    b = a.clone()
    first = stage_model_identity(a)
    assert stage_model_identity(b)["sha256"] == first["sha256"]
    b.patches[next(iter(b.patches))][0] = (.5, ("lora", (tensor,)), 1., None, None)
    assert stage_model_identity(b)["sha256"] != first["sha256"]
    assert len(a.patches[next(iter(a.patches))]) == 1


def test_unsampled_weight_bytes_are_hashed_not_just_64_positions():
    model = small_model()
    before = stage_model_identity(model)["sha256"]
    with torch.no_grad():
        model.model.diffusion_model.blocks[0].attn.qkv_proj.weight.view(-1)[57] += .001
    assert stage_model_identity(model)["sha256"] != before


def test_pre_lora_backup_provides_stable_base_identity():
    model = small_model()
    before = stage_model_identity(model)["sha256"]
    name = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = model.model_state_dict()[name]
    backup = namedtuple("Backup", "weight inplace_update")
    model.backup[name] = backup(weight.clone(), False)
    with torch.no_grad():
        weight.add_(1.)
    assert stage_model_identity(model)["sha256"] == before


def test_kj_memory_identity_is_verified_without_losing_configuration(request):
    memory_fixture = request.getfixturevalue("memory_nodes")
    model = patched_source(memory_fixture, "sage_lowmem_ffn", 2)
    first = stage_model_identity(model)
    assert first["memory"]["head_chunks"] == 2
    other = patched_source(memory_fixture, "sage_lowmem_ffn", 3)
    assert stage_model_identity(other)["sha256"] != first["sha256"]


def test_unknown_live_hook_cannot_share_resume_identity():
    model = small_model()
    model.model.diffusion_model.blocks[0].register_forward_pre_hook(lambda *args: None)
    with pytest.raises(ValueError, match="shared live hooks"):
        stage_model_identity(model)


def test_media_reordering_and_content_replacement_changes_identity():
    a, b = torch.zeros(100), torch.ones(100)
    before = content_identity([a, b])
    assert content_identity([b, a]) != before
    a[57] = .001
    assert content_identity([a, b]) != before


def test_actual_loader_metadata_is_hashed_but_not_an_execution_patch_bypass():
    model = small_model()
    model.set_attachments("t8_h3_lora_metadata", {"format": "comfy", "training": "EMA B"})
    first = stage_model_identity(model)["sha256"]
    model.set_attachments("t8_h3_lora_metadata", {"format": "comfy", "training": "other"})
    assert stage_model_identity(model)["sha256"] != first
    model.set_attachments("foreign_runtime", {"claimed_safe": True})
    with pytest.raises(ValueError, match="unknown attachments"):
        stage_model_identity(model)


def test_loader_metadata_cannot_hide_a_callback():
    model = small_model()
    model.set_attachments("t8_h3_lora_metadata", {"callback": lambda: None})
    with pytest.raises(ValueError, match="plain safetensors string map"):
        stage_model_identity(model)


def test_current_core_lora_adapter_and_legacy_tuples_are_both_content_bound():
    from comfy.weight_adapter.lora import LoRAAdapter
    weights = (torch.ones(2, 1), torch.ones(1, 3), 1., None, None, None)
    adapter = LoRAAdapter({'up', 'down'}, weights)
    before = content_identity({'patch': [(1., adapter, 1., None, None)]})
    weights[0][0, 0] = 2.
    assert content_identity({'patch': [(1., adapter, 1., None, None)]}) != before
    assert content_identity(('lora', weights))['type'] == 'tuple'
    adapter.h = lambda *args: None
    with pytest.raises(ValueError, match='runtime mutations'):
        content_identity(adapter)


def test_float8_raw_identity_is_bounded_and_finite_on_cpu():
    import hashlib
    value = torch.ones(1024 * 1024 + 3).to(torch.float8_e4m3fn)
    identity = content_identity(value)
    assert identity['tensor_sha256'] == hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert identity['dtype'] == 'torch.float8_e4m3fn'
    value.view(torch.uint8)[-1] = 127
    with pytest.raises(ValueError, match='nonfinite'):
        content_identity(value)
