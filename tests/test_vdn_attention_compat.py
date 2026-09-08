from types import SimpleNamespace

import pytest
import torch
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.modules import attention
from comfy.model_patcher import ModelPatcher

from h3_audio_t8_pkg import vdn_h3_advanced as vdn
from h3_audio_t8_pkg.vdn_attention_compat import prepare_vdn_attention_model


def model_fixture():
    diffusion = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    diffusion.blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for block in diffusion.blocks:
        block.attn = torch.nn.Identity()
    model = torch.nn.Module()
    model.diffusion_model = diffusion
    model.model_sampling = SimpleNamespace(percent_to_sigma=lambda p: 1 - p)
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def sparse(model):
    core = pytest.importorskip("comfy_extras.nodes_sparse_attention")
    return core.apply_block_sparse_attention(
        model, tau=1.3, topk_ratio=0, vsa=False, start_percent=0, end_percent=1,
        min_tokens=4096, dense_blocks=set(), sink_conditioning="off", extra_tokens=0, verbose=False,
    )


def test_sparse_before_vdn_is_locally_detached_with_backend_and_user_callbacks_preserved():
    pytest.importorskip("comfy_extras.nodes_sparse_attention")
    model = model_fixture()
    model.set_model_optimized_attention(attention.attention_pytorch)
    incoming = sparse(model)
    incoming.add_callback_with_key("test", "user", lambda *_: None)
    original_hook = incoming.model_options["transformer_options"]["optimized_attention_override"]
    adapted, removed = prepare_vdn_attention_model(incoming)
    assert removed == 3
    assert adapted is not incoming
    assert vdn._attention_conflicts(adapted) == []
    assert incoming.model_options["transformer_options"]["optimized_attention_override"] is original_hook
    assert adapted.callbacks["test"]["user"] == incoming.callbacks["test"]["user"]
    assert any(values for groups in incoming.callbacks.values() for key, values in groups.items() if key == "block_sparse_attention")
    assert not any(values for groups in adapted.callbacks.values() for key, values in groups.items() if key == "block_sparse_attention")


def test_sparse_after_vdn_restores_real_hook_identity_on_each_prepare():
    model = model_fixture()
    hooks = (lambda *args: None, lambda *args: None)
    model.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = hooks
    for index, hook in enumerate(hooks):
        model.set_model_patch_replace(hook, "dit", "double_block", index)
    patched = sparse(model)
    options = patched.model_options["transformer_options"]
    for _ in range(2):
        patched.prepare_state(torch.tensor(0.5), patched.model_options)
        vdn.validate_vdn_runtime_options(options)
        assert "optimized_attention_override" not in options
        assert all(options["patches_replace"]["dit"][("double_block", i)] is h for i, h in enumerate(hooks))


def test_unknown_replacement_is_not_erased_alongside_known_sparse_node():
    model = model_fixture()
    hooks = (lambda *args: None, lambda *args: None)
    model.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = hooks
    patched = sparse(model)
    def alien(*args):
        return None
    patched.set_model_patch_replace(alien, "dit", "double_block", 1)
    options = patched.model_options["transformer_options"]
    with pytest.raises(RuntimeError, match="indices.*1"):
        vdn.validate_vdn_runtime_options(options)
    assert options["patches_replace"]["dit"][("double_block", 1)] is alien


def test_actual_extension_loader_alias_is_recognized(monkeypatch):
    import importlib.util
    import sys
    core = pytest.importorskip("comfy_extras.nodes_sparse_attention")
    spec = importlib.util.spec_from_file_location("nodes_sparse_attention", core.__file__)
    alias = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, alias)
    spec.loader.exec_module(alias)
    model = model_fixture()
    patched = alias.apply_block_sparse_attention(
        model, tau=1.3, topk_ratio=0, vsa=False, start_percent=0, end_percent=1,
        min_tokens=4096, dense_blocks=set(), sink_conditioning="off", extra_tokens=0, verbose=False,
    )
    adapted, removed = prepare_vdn_attention_model(patched)
    assert removed == 3
    assert vdn._attention_conflicts(adapted) == []
    monkeypatch.setattr(alias, "__file__", str(__file__))
    unchanged, removed = prepare_vdn_attention_model(patched)
    assert removed == 0
    assert unchanged is patched
