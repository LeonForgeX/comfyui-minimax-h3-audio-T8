from copy import deepcopy
import hashlib
import json

import pytest

from tools.audit_trt_encoder_condition_pair import verify_reference_conditioning


def condition(latent, text="text-sha"):
    blocks = [[{"sha256": text}, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": latent}]}]]
    raw = json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode()
    return {"conditioning": blocks, "conditioning_sha256": hashlib.sha256(raw).hexdigest(),
            "initial_av": ["same-video", "same-audio"]}


def test_condition_comparison_removes_only_verified_reference():
    a = condition({"sha256": "native"})
    b = condition({"sha256": "trt"})
    original = deepcopy(a)
    assert verify_reference_conditioning(a, {"sha256": "native"}) == verify_reference_conditioning(b, {"sha256": "trt"})
    assert a == original
    c = condition({"sha256": "trt"}, text="changed-text")
    assert verify_reference_conditioning(a, {"sha256": "native"}) != verify_reference_conditioning(c, {"sha256": "trt"})


def test_condition_comparison_refuses_unused_or_substituted_encoder():
    with pytest.raises(ValueError, match="not the saved"):
        verify_reference_conditioning(condition({"sha256": "native"}), {"sha256": "trt"})


def test_condition_comparison_refuses_modified_history_digest():
    value = condition({"sha256": "native"})
    value["conditioning"][0][1]["minimax_keyframes"][0]["resolved_frame_index"] = 5
    with pytest.raises(ValueError, match="digest"):
        verify_reference_conditioning(value, {"sha256": "native"})
