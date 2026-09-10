import pytest
import torch
from tools.audit_trt_vae_t1_probe import metrics


def test_encoder_metrics_compute_relative_error_not_visual_acceptance():
    value = torch.ones(2, 3)
    result = metrics(value, value * 1.01)
    assert result["relative_rmse_percent"] == pytest.approx(1, abs=1e-4)
    assert result["rmse"] == pytest.approx(.01, abs=1e-6)
    assert "passed" not in result


@pytest.mark.parametrize("case", ["shape", "nan", "inf", "integer", "zero"])
def test_encoder_metrics_reject_invalid_evidence(case):
    reference = torch.ones(2, 3)
    candidate = reference.clone()
    if case == "shape":
        candidate = candidate[:, :1]
    elif case in ("nan", "inf"):
        candidate[0, 0] = float(case)
    elif case == "integer":
        reference = reference.to(torch.uint8)
    else:
        reference.zero_()
    with pytest.raises(ValueError):
        metrics(reference, candidate)


def test_full_api_normalization_audit_distinguishes_buffer_precision_from_seam_errors():
    from tools.audit_trt_vae_full_encoder import normalization_residual
    mean, std = [.12345]*24, [1.2345]*24
    old = torch.linspace(-2,2,24*8).reshape(1,24,1,2,4)
    m,s = [torch.tensor(v).view(1,24,1,1,1) for v in (mean,std)]
    actual = (old*s+m-m.half().float())/s.half().float()
    receipt = normalization_residual(old,actual,mean,std)
    assert receipt["all_elements_within_affine_roundoff"] and not receipt["old_trt_bit_equal"]
    actual[...,0,0] += .0001
    with pytest.raises(ValueError,match="not explained"):
        normalization_residual(old,actual,mean,std)
