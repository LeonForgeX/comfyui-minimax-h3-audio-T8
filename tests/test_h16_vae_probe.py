from types import SimpleNamespace
import subprocess

import pytest
import torch

from tools.run_h16_vae_decode_probe import audit_preview, expected_shape, preview, tensor_sha
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE


@pytest.mark.parametrize("t,hw", [(1, (30, 52)), (2, (30, 52)), (7, (30, 52)),
                                  (22, (30, 52)), (22, (29, 51))])
def test_probe_shape_contract_matches_native_core_without_weights(t, hw):
    vae = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.vae_ratio, vae.vae_ratio_t = 16, 4
    vae.clip_length = 17
    vae.token_drop, vae.tokens_chunk_size, vae.token_overlap, vae.frame_pre_padding = 3, 5, 2, 3
    vae.decoder = SimpleNamespace(out_channels=3)
    value = torch.empty((1, 24, t, *hw), device="meta")
    b, c, f, h, w = vae.decode_output_shape(value.shape)
    assert b == 1 and expected_shape(value) == (f, h, w, c)


def test_tensor_digest_covers_noncontiguous_values():
    tensor = torch.arange(24).reshape(4, 6).T
    assert tensor_sha(tensor) == tensor_sha(tensor.contiguous())
    assert tensor_sha(tensor) != tensor_sha(tensor + 1)


def test_preview_h264_full_audio_copy_and_strict_audit(tmp_path):
    source, output = tmp_path / "source.m4a", tmp_path / "preview.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=0.25",
                    "-c:a", "aac", str(source)], check=True)
    images = torch.zeros(6, 32, 48, 3)
    images[:, :, :, 1] = .25
    command = preview(images, source, output)
    assert "-shortest" not in command and "copy" in command
    report = audit_preview(output, source, 6, 48, 32)
    assert report["source_audio_payloads_identical"] and report["audio_packet_count"] > 0
    with pytest.raises(RuntimeError, match="canvas"):
        audit_preview(output, source, 7, 48, 32)
