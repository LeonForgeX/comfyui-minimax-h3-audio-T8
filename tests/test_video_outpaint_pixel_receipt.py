import copy
import hashlib

import pytest

from h3_audio_t8_pkg.video_outpaint_pixel_receipt import validate_pixel_receipt
from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan, canonical


def fixture(mode):
    plan = build_outpaint_plan(
        source_sha256="a" * 64,
        width=128,
        height=128,
        frame_count=2,
        aspect="custom",
        top=64,
        bottom=64,
    )
    exact = mode == "preserve_source"
    receipt = {
        "schema": "t8.h3.video_outpaint.pre_encode_pixels/v1"
        if exact
        else "t8.h3.video_outpaint.reconstructed_pixels/v1",
        "plan_sha256": plan["plan_sha256"],
        "source_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "frame_count": 2,
        "width": 128,
        "height": 256,
        "source_exact_before_encoding": exact,
        "lossy_encoded_equality_claimed": False,
        "source_rgb_sha256": "c" * 64,
    }
    if exact:
        receipt["pasted_rgb_sha256"] = "c" * 64
    else:
        receipt.update(
            source_mode=mode,
            source_reconstructed=True,
            reconstructed_source_rgb_sha256="d" * 64,
        )
    return plan, seal(receipt)


def seal(receipt):
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    return receipt


@pytest.mark.parametrize("mode", ["preserve_source", "joint_decode"])
def test_distinct_receipt_contracts_cannot_be_interchanged(mode):
    plan, receipt = fixture(mode)
    kwargs = {"plan": plan, "source_sha256": "a" * 64, "candidate_sha256": "b" * 64}
    validate_pixel_receipt(receipt, source_mode=mode, **kwargs)
    other = "joint_decode" if mode == "preserve_source" else "preserve_source"
    with pytest.raises(ValueError):
        validate_pixel_receipt(receipt, source_mode=other, **kwargs)
    for key, value in [
        ("source_exact_before_encoding", not (mode == "preserve_source")),
        ("lossy_encoded_equality_claimed", True),
        ("source_rgb_sha256", "bad"),
        ("candidate_sha256", "e" * 64),
    ]:
        tampered = copy.deepcopy(receipt)
        tampered[key] = value
        with pytest.raises(ValueError):
            validate_pixel_receipt(seal(tampered), source_mode=mode, **kwargs)


def test_joint_receipt_never_accepts_pasteback_claim_even_with_equal_hashes():
    plan, receipt = fixture("joint_decode")
    receipt["reconstructed_source_rgb_sha256"] = receipt["source_rgb_sha256"]
    kwargs = {
        "plan": plan,
        "source_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "source_mode": "joint_decode",
    }
    validate_pixel_receipt(
        seal(receipt), **kwargs
    )  # Coincidental equality does not change the mode contract.
    receipt["pasted_rgb_sha256"] = receipt["source_rgb_sha256"]
    with pytest.raises(ValueError):
        validate_pixel_receipt(seal(receipt), **kwargs)
