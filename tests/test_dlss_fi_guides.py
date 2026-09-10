import numpy as np
import pytest

from tools.dlss_fi_guides import MotionGuide


def picture():
    # Deterministic textured plane translated horizontally, not a generated video.
    rng = np.random.default_rng(21)
    rgba = rng.integers(0, 256, (128, 256, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def test_static_and_explicit_cut_are_zero_guides_not_generated_images():
    guide, frame = MotionGuide(256, 128), picture()
    field, reset = guide.process(frame)
    assert reset and field.shape == (128, 256, 2) and not field.any()
    field, reset = guide.process(frame.copy())
    assert not reset and not field.any()
    field, reset = guide.process(np.roll(frame, 8, axis=1), reset=True)
    assert reset and not field.any()


def test_current_to_previous_pixel_direction_for_translation():
    guide, frame = MotionGuide(256, 128), picture()
    guide.process(frame)
    field, reset = guide.process(np.roll(frame, 8, axis=1))
    assert not reset and field.dtype == np.float16 and np.isfinite(field).all()
    center = field[32:-32, 32:-32]
    assert abs(float(np.median(center[..., 0])) + 8) < 1
    assert abs(float(np.median(center[..., 1]))) < 1


def test_nonfinite_or_overflow_flow_is_rejected_not_replaced(monkeypatch):
    class BadFlow:
        def calc(self, *args):
            return np.full((128, 256, 2), np.inf, dtype=np.float32)
    guide, frame = MotionGuide(256, 128), picture()
    guide.process(frame)
    guide.estimator = BadFlow()
    with pytest.raises(ValueError, match='Invalid optical'):
        guide.process(np.roll(frame, 8, axis=1))


@pytest.mark.parametrize('width,height', [(0,128),(256,True),(8192,8192)])
def test_invalid_geometry_rejected(width, height):
    with pytest.raises(ValueError):
        MotionGuide(width, height)


def test_dtype_and_shape_not_silently_converted():
    with pytest.raises(ValueError):
        MotionGuide(256,128).process(picture().astype(np.float32))
