import numpy as np
import pytest
import torch

from h3_audio_t8_pkg.video_outpaint_alignment import _mesh, register_outpaint_candidate
from h3_audio_t8_pkg.video_outpaint_composite import composite_outpaint_frames
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan


def _fixture(dtype=torch.float32, frames=2):
    import cv2
    plan = build_outpaint_plan(source_sha256='a'*64, width=128, height=128,
                              frame_count=frames, aspect='custom', left=64, top=64, right=64, bottom=64)
    random = np.random.default_rng(314).random((256,256,3)).astype(np.float32)
    truth = cv2.GaussianBlur(random, (5,5), 1)
    candidate = cv2.warpAffine(truth, np.float32([[1,0,3],[0,1,-2]]), (256,256), borderMode=cv2.BORDER_REFLECT_101)
    source = truth[64:192,64:192].copy()
    def tensor(x):
        t = torch.from_numpy(x.copy())[None].repeat(frames,1,1,1)
        return (t*255).round().to(dtype) if dtype == torch.uint8 else t.to(dtype)
    return tensor(source), tensor(candidate), plan, tensor(truth)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.uint8])
def test_real_correspondences_correct_translation_without_touching_source_or_far_pixels(dtype):
    source, candidate, plan, truth = _fixture(dtype)
    before = candidate.clone()
    registered, report = register_outpaint_candidate(source, candidate, plan, band_pixels=32)
    assert torch.equal(candidate, before)
    assert torch.equal(registered[:,64:192,64:192], candidate[:,64:192,64:192])
    assert torch.equal(registered[:,:32], candidate[:,:32])
    assert torch.equal(registered[:,224:], candidate[:,224:])
    composed, preservation = composite_outpaint_frames(source, registered, plan)
    assert torch.equal(composed[:,64:192,64:192], source)
    assert preservation['source_exact_before_encoding']
    # Four sides independently, away from corner blending. Texture is known,
    # unlike a boundary RGB jump which can improve merely by blurring.
    for ys, xs in [(slice(61,64),slice(76,180)),(slice(192,195),slice(76,180)),
                   (slice(76,180),slice(61,64)),(slice(76,180),slice(192,195))]:
        old = (candidate[:,ys,xs].float()-truth[:,ys,xs].float()).abs().mean()
        new = (registered[:,ys,xs].float()-truth[:,ys,xs].float()).abs().mean()
        assert new < old*.45, (old,new)
    assert not report['perceptual_acceptance'] and not report['color_correction']


def test_frame_chunking_auxiliary_channels_and_disabled_path():
    source, candidate, plan, _ = _fixture()
    source = torch.cat((source, torch.full_like(source[...,:1], .9)), dim=-1)
    candidate = torch.cat((candidate, torch.full_like(candidate[...,:1], .3)), dim=-1)
    whole, _ = register_outpaint_candidate(source, candidate, plan)
    pieces = [register_outpaint_candidate(source[i:i+1],candidate[i:i+1],plan,start_frame=i)[0] for i in range(2)]
    assert torch.equal(whole, torch.cat(pieces))
    assert torch.equal(whole[...,3:], candidate[...,3:])
    disabled, _ = register_outpaint_candidate(source,candidate,plan,max_displacement=0)
    assert torch.equal(disabled,candidate)


def test_uncertain_correspondences_do_not_move_pixels():
    flow = np.ones((64,64,2), np.float32)*4
    uncertain = np.full((64,64), 100, np.float32)
    mesh, sides, _ = _mesh(flow,uncertain,[32,32,96,96],128,128,32,8)
    assert not np.any(mesh)
    assert all(not side['applied'] for side in sides)


@pytest.mark.parametrize('settings', [{'band_pixels':0},{'max_displacement':float('nan')},{'max_displacement':17}])
def test_invalid_settings_rejected(settings):
    source,candidate,plan,_ = _fixture()
    with pytest.raises(ValueError):
        register_outpaint_candidate(source,candidate,plan,**settings)
