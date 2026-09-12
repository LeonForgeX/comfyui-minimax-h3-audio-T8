"""Compare local split motion-guide coordinates with current native H3 grids."""
import pytest
import torch

from comfy.ldm.minimax.model import PackedLayout
from h3_audio_t8_pkg.long_video import (
    CONTEXT_FRAME_STEPS, MOTION_FRAME_INDEX, step_offsets,
    pixel_frames_from_latent_t, repair_long_video_layout,
)


@pytest.mark.parametrize('frames', [22, 39])
@pytest.mark.parametrize('height,width', [(14, 28), (28, 56)])
def test_saved_tail_phase_and_guide_coordinates_match_native(frames, height, width):
    steps, target_t = CONTEXT_FRAME_STEPS[frames], 37
    offsets = step_offsets(target_t)
    start = target_t - steps
    # The saved tail starts at exactly the global decoded frame that the next
    # window subtracts from its accepted-start time; no hidden off-by-one here.
    assert offsets[start] == pixel_frames_from_latent_t(target_t) - frames
    assert [v - offsets[start] for v in offsets[start:]] == step_offsets(steps)
    tail = torch.zeros(1, 24, steps, height, width)
    split = [{'resolved_frame_index': 0, MOTION_FRAME_INDEX: offset,
              'latent': tail[:, :, index:index + 1]}
             for index, offset in enumerate(step_offsets(steps))]
    # Keep a reference timeline span in both layouts, so matching the target
    # origin is tested as well as the relative motion-guide offsets.
    refs = [{'kind': 'audio', 'ref_audio_t': 37,
             'audio_latent': torch.zeros(1, 32, 2, 37)}]
    native = PackedLayout(11, target_t, height, width, 207,
                          keyframes=[{'resolved_frame_index': 0, 'latent': tail}], refs=refs)
    local = PackedLayout(11, target_t, height, width, 207, keyframes=split, refs=refs)
    repair_long_video_layout(local, split, refs, 124)
    torch.testing.assert_close(local.position_ids, native.position_ids, rtol=0, atol=1e-10)
