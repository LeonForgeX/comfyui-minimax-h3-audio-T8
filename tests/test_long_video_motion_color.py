import builtins

import numpy as np
import pytest
import torch

from h3_audio_t8_pkg import long_video_motion_color as mc


def textured(count=16):
    rng = np.random.default_rng(44)
    rgb = .35 + rng.random((64, 64, 3)).astype(np.float32) * .25
    frames = torch.from_numpy(np.repeat(rgb[None], count, axis=0))
    return frames


def test_local_transient_color_is_reduced_without_geometry_or_tail_changes():
    reference = textured(5)
    source = textured()
    source[1, 12:52, 8:30, 0] += .02
    source[1, 12:52, 34:56, 0] -= .02
    original = source.clone()
    output, report = mc.stabilize_local_motion_color(source, reference)
    before = (source[1, 16:48, 12:26, 0] - reference[-1, 16:48, 12:26, 0]).mean().abs()
    after = (output[1, 16:48, 12:26, 0] - reference[-1, 16:48, 12:26, 0]).mean().abs()
    assert after < before
    assert torch.equal(source, original)
    assert torch.equal(source[12:], output[12:])
    assert report['maximum_applied_rgb_delta'] <= .008001
    assert report['maximum_applied_luma_delta'] <= .006001
    assert report['maximum_applied_chroma_delta'] <= .004001
    assert report['maximum_global_rgb_mean_delta'] < .0001
    assert report['geometry_warped'] is report['audio_touched'] is report['latent_touched'] is False


def test_static_source_and_auxiliary_channels_are_preserved():
    source = torch.cat((textured(), torch.full((16, 64, 64, 1), .7)), -1)
    output, report = mc.stabilize_local_motion_color(source, source[-5:])
    assert torch.equal(output, source)
    assert report['applied'] is False


def test_monotonic_lighting_is_not_flattened():
    reference = textured(5)
    source = textured()
    for index in range(16):
        source[index] += .001 * (index + 1)
    output, _ = mc.stabilize_local_motion_color(source, reference)
    assert float((output - source).abs().max()) < .0001


def test_uniform_flash_retains_global_rgb_mean():
    source = textured()
    source[1] += .015
    output, report = mc.stabilize_local_motion_color(source, textured(5))
    assert float((output.mean((1, 2)) - source.mean((1, 2))).abs().max()) < .0001
    assert report['maximum_applied_rgb_delta'] <= .008001


def test_large_cut_abstains():
    source = torch.full((16, 64, 64, 3), .8)
    output, report = mc.stabilize_local_motion_color(source, torch.full((5, 64, 64, 3), .1))
    assert output is source
    assert report['status'] == 'abstain_large_cut_or_zero_strength'


@pytest.mark.parametrize('fault', ['nan', 'range', 'canvas', 'offset', 'window'])
def test_invalid_input_fails_closed(fault):
    source, reference, settings = textured(), textured(5), {}
    if fault == 'nan':
        source[0, 0, 0, 0] = float('nan')
    elif fault == 'range':
        source[0, 0, 0, 0] = 1.1
    elif fault == 'canvas':
        reference = reference[:, :32]
    elif fault == 'offset':
        settings['maximum_rgb_offset'] = .5
    else:
        settings['affected_frames'] = 100
    with pytest.raises(ValueError):
        mc.stabilize_local_motion_color(source, reference, **settings)


def test_missing_opencv_has_clear_optional_dependency_error(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == 'cv2':
            raise ImportError('missing optional OpenCV')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', unavailable)
    with pytest.raises(RuntimeError, match='other Color Match modes do not'):
        mc.runtime_identity()


def test_same_source_is_deterministic():
    source = textured()
    source[2, 8:56, 8:28] += .015
    first, report = mc.stabilize_local_motion_color(source, textured(5))
    second, repeated = mc.stabilize_local_motion_color(source, textured(5))
    assert torch.equal(first, second)
    assert report == repeated


def test_untrusted_correspondences_cannot_change_output(monkeypatch):
    def rejected(cv2, target, neighbor, target_low, neighbor_low):
        h, w = target.shape[:2]
        return mc._opponent(neighbor_low), np.zeros((h, w), bool), np.zeros((h, w), np.float32)

    monkeypatch.setattr(mc, '_align_neighbor', rejected)
    source = textured()
    source[1, :32] += .02
    output, report = mc.stabilize_local_motion_color(source, textured(5))
    assert output is source
    assert not report['applied']


def test_gamut_boundaries_and_auxiliary_channel_are_safe():
    source = textured()
    source[:, :8] = 0
    source[:, -8:] = 1
    source[1, 16:48, :32, 2] += .012
    source = torch.cat((source, torch.full((16, 64, 64, 1), .123)), -1)
    reference = source[4:9].clone()
    output, report = mc.stabilize_local_motion_color(source, reference)
    assert float(output[..., :3].min()) >= 0 and float(output[..., :3].max()) <= 1
    assert torch.equal(output[..., 3:], source[..., 3:])
    assert report['maximum_applied_luma_delta'] <= .006001
    assert report['maximum_applied_chroma_delta'] <= .004001
