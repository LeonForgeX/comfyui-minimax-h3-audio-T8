"""H3's packed conditioning/audio rows must not be approximated as video."""
from types import SimpleNamespace

import pytest
from h3_audio_t8_pkg import relay_sol_backend as sol


def test_non_video_prefix_rounds_outward_to_cover_mixed_boundary_block():
    layout = SimpleNamespace(seq_len=513, segments=[(0, 70, 'text'),
        (70, 100, 'cond'), (100, 145, 'audio'), (145, 513, 'video')])
    assert sol.h3_exact_prefix(layout, 513) == (0, 3)


@pytest.mark.parametrize('segments', [
    [(0, 64, 'text'), (64, 513, 'video'), (513, 514, 'audio')],
    [(0, 64, 'text'), (65, 128, 'audio'), (128, 513, 'video')],
    [(0, 64, 'text'), (64, 513, 'video')],
])
def test_unknown_or_noncontiguous_layout_is_not_guessed(segments):
    assert sol.h3_exact_prefix(SimpleNamespace(seq_len=513, segments=segments), 513) is None


def test_absent_layout_is_not_assumed_to_be_video_only():
    assert sol.h3_exact_prefix(None, 513) is None
