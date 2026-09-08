import json
from types import SimpleNamespace

import pytest
import torch
import comfy.nested_tensor

from h3_audio_t8_pkg import vdn_two_pass as twopass
from h3_audio_t8_pkg.sampling import native_flow_sigmas


def latent(h=2, w=2, t=7):
    return {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, t, h, w), torch.zeros(1, 32, 2, 39)))}


@pytest.mark.parametrize("stage,steps,refine", [("stage_dmd_8nfe", 8, 4), ("stage_dmd_8nfe", 8, 3), ("stage_b_50nfe", 50, 5)])
def test_refine_uses_own_stage_grid_not_lightx2v(monkeypatch, stage, steps, refine):
    model = SimpleNamespace(get_attachment=lambda _: {"status": "configured", "stage": stage})
    full = native_flow_sigmas(steps, 12.)
    monkeypatch.setattr(twopass, "setup_vdn_execution", lambda m, a: (m, "euler", full, "{}"))
    result = twopass.setup_vdn_refine(model, latent(4, 4), latent(), refine)
    torch.testing.assert_close(result[2], full[-refine-1:])
    report = json.loads(result[3])
    assert report["total_nfe"] == steps + refine
    assert report["audio_matches_first_pass"]
    assert report["fresh_restart_noise_required"]
    assert report["additional_turbo_lora"] is False


@pytest.mark.parametrize("refine", [0, 8, 50])
def test_invalid_dmd_restart_rejected_before_sampling(refine):
    model = SimpleNamespace(get_attachment=lambda _: {"status": "configured", "stage": "stage_dmd_8nfe"})
    with pytest.raises(ValueError, match="refine_steps"):
        twopass.setup_vdn_refine(model, latent(4, 4), latent(), refine)


def test_changed_frame_count_rejected():
    model = SimpleNamespace(get_attachment=lambda _: {"status": "configured", "stage": "stage_dmd_8nfe"})
    with pytest.raises(ValueError, match="frame count"):
        twopass.setup_vdn_refine(model, latent(4, 4, 8), latent(), 4)
