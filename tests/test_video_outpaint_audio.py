from contextlib import closing

import pytest
import torch
from torch.nn import functional as F

from comfy.ldm.minimax import audio_vae as native
from h3_audio_t8_pkg.video_outpaint_audio import (
    causal_prefix_attention, iter_encode_outpaint_audio, _encoder_halo,
    iter_encode_outpaint_audio_vae, loaded_audio_vae_identity,
)


class TinyAudio(torch.nn.Module):
    """Native operations with small random weights, NOT a learned H3 checkpoint."""
    encode = native.MiniMaxH3AudioVAE.encode

    def __init__(self):
        super().__init__()
        self.sample_rate, self.hop_length = 32000, 800
        self.encoder = native.Encoder(d_model=2, d_latent=16)
        self.pre_block = native.AttnProjection(16, 32, 8)
        self.mean_proj = torch.nn.Conv1d(32, 32, 1)
        self.register_buffer("latents_mean", torch.linspace(-0.1, 0.1, 32))
        self.register_buffer("latents_std", torch.linspace(0.8, 1.2, 32))
        generator = torch.Generator().manual_seed(143)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator)*0.08)
                if "norm" in name and name.endswith("weight") or name.endswith("alpha"):
                    parameter.fill_(1)
            self.pre_block.attn.zero_k_bias.zero_()


@pytest.mark.parametrize("length,block", [(1, 1), (17, 4), (33, 16), (129, 64)])
def test_prefix_attention_matches_native_causal_sdpa(length, block):
    gen = torch.Generator().manual_seed(21)
    q, k, v = [torch.randn(2, 3, length, 8, generator=gen)*3 for _ in range(3)]
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    chunks = []
    for start in range(0, length, block):
        reads = []

        def reader(a, b):
            reads.append((a, b))
            return k[..., a:b, :], v[..., a:b, :]

        chunks.append(causal_prefix_attention(q[..., start:start+block, :], start, reader, block_tokens=block))
        assert max(b-a for a, b in reads) <= block
    torch.testing.assert_close(torch.cat(chunks, dim=-2), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("samples,block", [(1, 4), (801, 4), (17*800+13, 8), (69*800-19, 16)])
def test_bounded_audio_matches_whole_native_encoder(tmp_path, samples, block):
    stage = TinyAudio().eval()
    assert _encoder_halo(stage) == 13
    wave = torch.randn(1, 2, samples, generator=torch.Generator().manual_seed(31))*0.15
    reads = []

    def reader(start, stop):
        reads.append((start, stop))
        return wave[..., start:stop].clone()

    with torch.inference_mode():
        expected = stage.encode(wave)
    actual = torch.cat([chunk for _, chunk in iter_encode_outpaint_audio(
        stage, reader, samples, block_tokens=block, scratch_parent=tmp_path)], dim=-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert max(b-a for a, b in reads) <= (block+26)*800
    assert not list(tmp_path.iterdir())


def test_cancel_and_bad_architecture_do_not_leave_scratch(tmp_path):
    stage = TinyAudio().eval()
    wave = torch.zeros(1, 2, 800*20)
    with closing(iter_encode_outpaint_audio(stage, lambda a, b: wave[..., a:b], wave.shape[-1],
                                            block_tokens=8, scratch_parent=tmp_path)) as chunks:
        assert next(chunks)[0] == 0
        assert list(tmp_path.iterdir())
    assert not list(tmp_path.iterdir())
    stage.encoder.block[0].padding = (4,)
    with pytest.raises(ValueError, match="receptive-field"):
        list(iter_encode_outpaint_audio(stage, lambda a, b: wave[..., a:b], wave.shape[-1], scratch_parent=tmp_path))
    assert not list(tmp_path.iterdir())


def test_attention_invalid_kv_and_cancel_are_explicit():
    query = torch.ones(1, 2, 4, 3)
    with pytest.raises(ValueError, match="shape"):
        causal_prefix_attention(query, 0, lambda a, b: (torch.zeros(1), torch.zeros(1)))

    def cancel():
        raise RuntimeError("cancel requested")

    with pytest.raises(RuntimeError, match="cancel"):
        causal_prefix_attention(query, 0, lambda a, b: None, interrupt_check=cancel)


class ManagedAudio:
    def __init__(self):
        from types import SimpleNamespace
        self.first_stage_model = TinyAudio().eval()
        self.vae_dtype = torch.float32
        self.device = torch.device("cpu")
        self.patcher = SimpleNamespace(patches={}, object_patches={})
        self.disable_offload = False
        self.memory_shapes = []

    def throw_exception_if_invalid(self):
        pass

    def memory_used_encode(self, shape, dtype):
        self.memory_shapes.append((shape, dtype))
        return shape[-1]*900*4


def test_managed_audio_uses_bounded_memory_and_native_model_lifecycle(tmp_path, monkeypatch):
    import comfy.model_management as management
    calls = []
    monkeypatch.setattr(management, "load_models_gpu", lambda *a, **kw: calls.append((a, kw)))
    vae = ManagedAudio()
    samples = 800*100
    with closing(iter_encode_outpaint_audio_vae(
        vae, lambda a, b: torch.zeros(1, 2, b-a), samples, block_tokens=8, scratch_parent=tmp_path,
    )) as chunks:
        assert next(chunks)[0] == 0
    assert vae.memory_shapes == [((1, 2, (8+26)*800), torch.float32)]
    assert len(calls) == 1 and calls[0][0] == ([vae.patcher],)
    assert calls[0][1]["force_full_load"] is False
    assert not list(tmp_path.iterdir())


def test_audio_identity_detects_actual_weights_buffers_and_pending_patches():
    vae = ManagedAudio()
    first = loaded_audio_vae_identity(vae)
    assert first == loaded_audio_vae_identity(vae)
    with torch.no_grad():
        vae.first_stage_model.pre_block.attn.zero_k_bias[0] += 1
    assert loaded_audio_vae_identity(vae)["sha256"] != first["sha256"]
    vae.patcher.patches["weight"] = "pending"
    with pytest.raises(ValueError, match="patches"):
        loaded_audio_vae_identity(vae)
