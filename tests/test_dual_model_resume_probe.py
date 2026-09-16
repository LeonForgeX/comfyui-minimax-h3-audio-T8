from copy import deepcopy

import pytest

from tools.run_dual_model_resume_probe import validate_resume


def receipt():
    return {
        "output_before": {"manifest.json": {"bytes": 1, "sha256": "a"}},
        "final_sha256": "f" * 64,
        "chain_id": "chain",
        "manifest_revision": 2,
        "segment_count": 2,
        "runtime_source_count": 12,
    }


def report():
    return {
        "resume_action": "returned_verified_existing_final",
        "final_video_sha256": "f" * 64,
        "segment_audits": [{}, {}],
    }


def test_cache_only_resume_requires_every_output_hash_to_remain_unchanged():
    result = validate_resume(receipt(), report(), receipt()["output_before"])
    assert result["status"] == "cache_only_resume_pass"
    assert result["segment_count"] == 2
    assert result["sampling_reused"] is False
    assert result["all_output_hashes_unchanged"] is True


@pytest.mark.parametrize("fault", ["output", "sha", "audits"])
def test_cache_only_resume_rejects_any_unproven_state(fault):
    source = receipt()
    resumed = report()
    after = deepcopy(source["output_before"])
    if fault == "output":
        after["manifest.json"]["sha256"] = "changed"
    elif fault == "sha":
        resumed["final_video_sha256"] = "different"
    else:
        resumed["segment_audits"].pop()
    with pytest.raises(ValueError):
        validate_resume(source, resumed, after)
