from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_guidance import build_outpaint_guidance
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_regional import (
    _runtime_route,
    make_outpaint_regional_bias,
    patch_outpaint_regional_model,
    regional_model_contract,
    route_outpaint_regional_attention,
)
from h3_audio_t8_pkg.video_outpaint_regional_conditioning import prepare_outpaint_regional_conditioning


class FakeClip:
    def tokenize(self, prompt):
        return prompt

    def encode_from_tokens_scheduled(self, prompt):
        n = len(prompt.split()) + 1
        return [[torch.zeros(1, n, 8), {
            "minimax_token_tags": torch.ones(n, dtype=torch.long), "pooled_output": None}]]


def _plan():
    return build_outpaint_plan(
        source_sha256="c" * 64, width=320, height=240, frame_count=22,
        aspect="custom", left=64, top=64, right=64, bottom=64,
        generation_megapixels=0.5,
    )


def _provider(tmp_path):
    plan = _plan()
    guidance = build_outpaint_guidance(
        plan, [{"shot": 0, "region": "top", "prompt": "blue sky"}], [])
    return prepare_outpaint_regional_conditioning(FakeClip(), "subject", guidance, plan, tmp_path)


class FakePatcher:
    def __init__(self):
        from comfy.model_base import MiniMaxH3
        from comfy.patcher_extension import WrappersMP
        self.model = MiniMaxH3.__new__(MiniMaxH3)
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.wrappers = WrappersMP.init_wrappers()
        self.attachments = {}
        for name in ("patches", "weight_wrapper_patches", "additional_models", "callbacks",
                     "injections", "hook_patches", "forced_hooks", "current_hooks"):
            setattr(self, name, {})

    def clone(self):
        return copy.deepcopy(self)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def set_model_optimized_attention(self, optimized_attention):
        def override(_, *args, **kwargs):
            return optimized_attention(*args, **kwargs)
        self.model_options["transformer_options"]["optimized_attention_override"] = override

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def get_attachment(self, key):
        return self.attachments.get(key)


def test_bias_blocks_each_region_span_only_outside_allowed_rows():
    allowed = torch.tensor([True, False, False, True])
    rows = torch.tensor([0, 1, 6, 7], dtype=torch.long)
    bias = make_outpaint_regional_bias(rows, 9, 4, [{
        "text_key_start": 2, "text_key_end": 4, "allowed": allowed,
    }], dtype=torch.float32)
    assert torch.equal(bias[[0, 3], 2:4], torch.zeros(2, 2))
    assert torch.all(bias[[1, 2], 2:4] == torch.finfo(torch.float32).min)
    assert torch.count_nonzero(bias[:, :2]) == 0


def test_runtime_route_binds_native_layout_and_spatial_grid(tmp_path):
    provider = _provider(tmp_path)
    grid = provider.binding["sampling_grid"]
    text = provider.binding["shots"][0]["text_len"]
    latent_h, latent_w, latent_t, audio_t = grid["rows"] * 2, grid["columns"] * 2, 2, 37
    video_start = text + audio_t * 2
    layout = SimpleNamespace(
        signature=(text, latent_t, latent_h, latent_w, audio_t),
        seq_len=video_start + latent_t * grid["rows"] * grid["columns"],
        segments=[(0, text, "text"), (text, video_start, "audio"),
                  (video_start, video_start + latent_t * grid["rows"] * grid["columns"], "video")],
    )
    route = _runtime_route(layout, provider.binding, 0, torch.device("cpu"))
    assert route["video_start"] == video_start
    assert route["frame_rows"] == grid["rows"] * grid["columns"]
    assert route["regions"][0]["allowed"].dtype == torch.bool


def test_router_leaves_nonvideo_queries_native_and_bounds_each_mask(monkeypatch):
    import h3_audio_t8_pkg.video_outpaint_regional as module

    masks = []

    def native(q, _k, _v, heads, **_kwargs):
        return torch.zeros(q.shape[0], q.shape[2], heads * q.shape[3])

    def masked(q, _k, _v, heads, mask, **_kwargs):
        masks.append(tuple(mask.shape))
        return torch.ones(q.shape[0], q.shape[2], heads * q.shape[3])

    monkeypatch.setattr(module.attention_module, "optimized_attention", native)
    monkeypatch.setattr(module.attention_module, "attention_pytorch", masked)
    q = torch.zeros(1, 2, 8, 4)
    route = {"seq_len": 8, "video_start": 4, "video_end": 8, "frame_rows": 2,
             "regions": [{"text_key_start": 1, "text_key_end": 3,
                           "allowed": torch.tensor([True, False])}]}
    result = route_outpaint_regional_attention(
        q, q, q, 2, skip_reshape=True,
        transformer_options={module.REGIONAL_RUNTIME_KEY: route}, query_chunk_rows=2)
    assert result.shape == (1, 8, 8)
    assert torch.count_nonzero(result[:, :4]) == 0
    assert torch.all(result[:, 4:] == 1)
    assert masks == [(2, 8), (2, 8)]


def test_model_composer_is_exact_and_rejects_tampering(tmp_path):
    provider = _provider(tmp_path)
    patched, contract = patch_outpaint_regional_model(FakePatcher(), provider, 128)
    assert contract["region_count"] == 1
    assert contract["query_chunk_rows"] == 128
    assert regional_model_contract(patched) == contract
    patched.model_options["transformer_options"]["optimized_attention_override"].\
        _t8_outpaint_regional_binding_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="owner"):
        regional_model_contract(patched)
