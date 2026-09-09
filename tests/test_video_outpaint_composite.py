from __future__ import annotations

import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
from h3_audio_t8_pkg.video_outpaint_composite import composite_outpaint_frames


def _fixture(channels=3, dtype=torch.float32):
    plan = build_outpaint_plan(source_sha256="a" * 64, width=43, height=37,
                              frame_count=89, aspect="custom", left=11, right=8, top=3, bottom=6)
    source = torch.arange(2 * 37 * 43 * channels).reshape(2, 37, 43, channels).remainder(256).to(dtype)
    candidate = torch.zeros((2, 46, 62, channels), dtype=dtype)
    return source, candidate, plan


@pytest.mark.parametrize("channels", [3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.uint8])
def test_exact_source_and_auxiliary_channels_with_no_input_mutation(channels, dtype):
    source, candidate, plan = _fixture(channels, dtype)
    original = source.clone()
    result, report = composite_outpaint_frames(source, candidate, plan, start_frame=23)
    assert torch.equal(result[:, 3:40, 11:54], source)
    assert torch.equal(source, original)
    assert torch.count_nonzero(candidate) == 0
    result[:, 3:40, 11:54] = 0
    assert torch.count_nonzero(result) == 0
    assert report["source_exact_before_encoding"] is True
    assert not report["lossy_encoded_pixel_equality_claimed"]
    assert not report["color_match_applied"]


def test_chunk_concatenation_matches_whole_composite():
    source, candidate, plan = _fixture()
    whole, _ = composite_outpaint_frames(source, candidate, plan)
    parts = [composite_outpaint_frames(source[i:i+1], candidate[i:i+1], plan, start_frame=i)[0] for i in range(2)]
    assert torch.equal(whole, torch.cat(parts))


def test_padding_frames_and_bad_shapes_cannot_enter_delivery():
    source, candidate, plan = _fixture()
    with pytest.raises(ValueError, match="frame range"):
        composite_outpaint_frames(source, candidate, plan, start_frame=88)
    with pytest.raises(ValueError, match="geometry"):
        composite_outpaint_frames(source[:, :-1], candidate, plan)
    with pytest.raises(ValueError, match="dtype"):
        composite_outpaint_frames(source.half(), candidate, plan)
    candidate[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        composite_outpaint_frames(source, candidate, plan)
