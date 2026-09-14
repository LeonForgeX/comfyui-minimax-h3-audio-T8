"""Public window settings must agree with the actual native chain planner."""
import pytest

from h3_audio_t8_pkg.nodes_progressive_long_video import MiniMaxH3ProgressiveLongVideoEXPT8 as Long
from h3_audio_t8_pkg.long_video_orchestration import build_long_video_chain_plan
from h3_audio_t8_pkg.core import MIN_TRAINED_FRAMES, MAX_TRAINED_FRAMES


def test_window_schema_matches_runtime_bounds():
    window = next(item for item in Long.define_schema().inputs if item.id == 'render_window_frames')
    assert (window.min, window.max, window.step) == (MIN_TRAINED_FRAMES, MAX_TRAINED_FRAMES, 17)
    for frames in range(window.min, window.max + 1, window.step):
        segment, = build_long_video_chain_plan('window_contract', 3., frames)
        assert segment.plan.render_frames == frames
        assert segment.plan.final_frame_count == 72


@pytest.mark.parametrize('frames', [39, 73, 123, 125, 379, 1025])
def test_invalid_window_rejected_without_models(frames):
    with pytest.raises(ValueError, match='range|grid'):
        build_long_video_chain_plan('window_contract', 3., frames)
