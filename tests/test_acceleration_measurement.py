from copy import deepcopy

import pytest

from h3_audio_t8_pkg.acceleration_measurement import Measurement, compare_equal_workload


def workload():
    return dict(model_sha256="a" * 64, conditioning_sha256="b" * 64, recipe_sha256="c" * 64,
                video_vae_sha256="d" * 64, audio_vae_sha256="e" * 64, seed=8,
                width=1024, height=512, frames=73, fps_num=24, fps_den=1,
                nfe=8, cfg=1., core_revision="test-core")


class Clock:
    value = 0.

    def __call__(self):
        return self.value


def measured(seconds=10.):
    clock = Clock()
    record = Measurement(workload(), cache_state="warm", clock=clock)
    with record.stage("sampling"):
        clock.value = seconds
    return record.finish()


def test_total_includes_unassigned_time_and_does_not_claim_peak():
    clock = Clock()
    source = workload()
    record = Measurement(source, cache_state="cold", clock=clock)
    source["nfe"] = 4
    clock.value = 1
    with record.stage("model_loading"):
        clock.value = 4
    clock.value = 10
    record.observe_resources({"device_used_bytes": 1234, "torch_allocated_bytes": 800})
    report = record.finish()
    assert report["elapsed_seconds"] == 10
    assert report["stage_seconds"] == 3
    assert report["unassigned_seconds"] == 7
    assert report["workload"]["nfe"] == 8
    assert report["resource_scope"] == "observed_samples_not_continuous_peaks"
    assert report["quality"] == "not_evaluated"


def test_exception_is_not_a_success_and_stage_releases():
    clock = Clock()
    record = Measurement(workload(), cache_state="warm", clock=clock)
    with pytest.raises(ValueError, match="model failed"):
        with record.stage("sampling"):
            clock.value = 4
            raise ValueError("model failed")
    assert not record.active
    failed = record.finish()
    assert failed["status"] == "failed"
    with pytest.raises(ValueError, match="complete"):
        compare_equal_workload(failed, measured())


def test_no_overlapping_stages_or_finish_while_active():
    clock = Clock()
    record = Measurement(workload(), cache_state="warm", clock=clock)
    with record.stage("outer"):
        with pytest.raises(RuntimeError, match="Nested"):
            with record.stage("inner"):
                pass
        with pytest.raises(RuntimeError, match="during"):
            record.finish()
        clock.value = 3
    record.finish()
    with pytest.raises(RuntimeError, match="closed"):
        record.finish()


def test_cache_hit_rejected_from_speed_comparison():
    baseline = measured()
    candidate = measured(1.)
    candidate["cache_hit"] = True
    with pytest.raises(ValueError, match="Cached"):
        compare_equal_workload(baseline, candidate)


@pytest.mark.parametrize("field,value", [("nfe", 12), ("model_sha256", "d" * 64),
    ("frames", 124), ("cfg", 5.), ("width", 512), ("recipe_sha256", "e" * 64),
    ("seed", 9), ("fps_num", 48), ("video_vae_sha256", "f" * 64)])
def test_different_recipe_not_equal_workload(field, value):
    baseline = measured()
    candidate = measured()
    candidate["workload"][field] = value
    with pytest.raises(ValueError, match="Workload mismatch"):
        compare_equal_workload(baseline, candidate)


def test_cold_warm_split():
    baseline, candidate = measured(), measured()
    candidate["cache_state"] = "cold"
    with pytest.raises(ValueError, match="cold with warm"):
        compare_equal_workload(baseline, candidate)


def test_slower_is_reported_honestly():
    result = compare_equal_workload(measured(10), measured(12))
    assert result["saved_fraction"] == pytest.approx(-.2)
    assert result["speedup"] == pytest.approx(10 / 12)
    assert "not_accepted" in result["scope"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1., True])
def test_invalid_elapsed(value):
    baseline = measured()
    broken = deepcopy(baseline)
    broken["elapsed_seconds"] = value
    with pytest.raises(ValueError, match="elapsed"):
        compare_equal_workload(baseline, broken)


@pytest.mark.parametrize("observation", [{"used": 3}, {"used_bytes": -1}, {"used_bytes": True}, {}])
def test_resource_units_and_invalid(observation):
    record = Measurement(workload(), cache_state="warm", clock=Clock())
    with pytest.raises(ValueError):
        record.observe_resources(observation)
