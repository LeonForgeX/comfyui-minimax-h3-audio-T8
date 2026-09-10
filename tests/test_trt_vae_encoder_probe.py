from fractions import Fraction

import numpy as np
from PIL import Image
import pytest
import torch

from tools.trt_vae_encoder_probe_worker import native_encoder_shell, source_tiles, probe_geometry
from tools.audit_trt_vae_encoder_probe import tensor_metrics


def test_metrics_include_all_elements_and_zero_reference_is_not_fake_psnr():
    a = torch.zeros(4)
    b = torch.tensor([0., 0., 0., 2.])
    report = tensor_metrics(a, b)
    assert report["rmse"] == 1 and report["max_error"] == 2
    assert report["relative_rmse"] is None and not report["exact_tensor_equal"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "shape", "integer"])
def test_invalid_metrics_rejected(bad):
    a = torch.zeros(2)
    b = torch.zeros(3) if bad == "shape" else torch.zeros(2, dtype=torch.int64) if bad == "integer" else torch.tensor([0., bad])
    with pytest.raises(ValueError):
        tensor_metrics(a, b)


def test_native_encoder_uses_actual_core_and_fp32_normalization():
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, LATENTS_MEAN
    core = native_encoder_shell()
    assert type(core) is MiniMaxH3VideoVAE
    assert core.encode.__func__ is MiniMaxH3VideoVAE.encode
    assert not hasattr(core, "decoder") and not hasattr(core, "post_quant_conv")
    assert all(p.is_meta for p in core.parameters())
    assert all(not b.is_meta for b in core.buffers())
    assert torch.equal(core.latents_mean, torch.tensor(LATENTS_MEAN))
    assert all(b.dtype == torch.float32 for b in core.buffers())


@pytest.mark.parametrize("frames", [16, 17])
def test_real_source_crop_has_exact_pixels_and_no_frame_repetition(tmp_path, frames):
    import av
    still = np.random.default_rng(3).integers(0, 256, (288, 320, 3), dtype=np.uint8)
    image = tmp_path / "source.png"
    Image.fromarray(still).save(image)
    video = tmp_path / "source.mkv"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 320, 288, "bgr0"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((288, 320, 3), i, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if frames == 16:
        with pytest.raises(ValueError, match="17 genuine"):
            source_tiles(image, video)
    else:
        inputs, crops = source_tiles(image, video)
        assert inputs["image"].shape == (1, 3, 1, 256, 256)
        assert inputs["video"].shape == (1, 3, 17, 256, 256)
        np.testing.assert_array_equal(inputs["image"][0, :, 0].permute(1, 2, 0).numpy(), still[16:272, 32:288])
        assert inputs["video"][0, 0, :, 0, 0].tolist() == list(range(17))
        assert crops["image"]["xywh"] == [32, 16, 256, 256]


def test_small_image_not_silently_upscaled(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (255, 256)).save(path)
    with pytest.raises(ValueError, match="without scaling"):
        source_tiles(path, tmp_path / "unused.mp4")


def test_complete_scope_is_real_short_video_not_relabelled_tile():
    assert probe_geometry({}) == (False, 256, 256, 17)
    assert probe_geometry({"scope": "full_0p5mp"}) == (True, 1024, 512, 73)
    with pytest.raises(ValueError):
        probe_geometry({"scope": "long32"})
    with pytest.raises(ValueError, match="bounded"):
        source_tiles("unused", "unused", frame_count=768)
