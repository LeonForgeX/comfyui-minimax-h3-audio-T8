import math
import pytest
import torch
from tools.audit_h16_vae_comparison import metrics


def test_equal_and_fixed_delta():
    first = torch.full((2, 3, 4, 3), .25)
    result = metrics(first, first)
    assert result["identical"] and result["mae"] == 0 and result["psnr_db"] is None
    result = metrics(first, first + .125)
    assert result["mae"] == .125 and result["mse"] == .125**2
    assert result["psnr_db"] == pytest.approx(-10 * math.log10(.125**2))
    assert result["frames"][1]["mean_rgb_delta"] == [.125] * 3


def test_mismatched_or_nonfinite_rejected():
    first = torch.zeros(2, 3, 4, 3)
    with pytest.raises(ValueError):
        metrics(first, first[:1])
    bad = first.clone()
    bad[1, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        metrics(first, bad)
