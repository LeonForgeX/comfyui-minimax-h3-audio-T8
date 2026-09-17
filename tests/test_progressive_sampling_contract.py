import copy

import pytest
import torch

from h3_audio_t8_pkg import progressive_sampling_contract as contract


def inputs(width=128, height=64, t=7):
    frames = 5 + ((t - 2) // 5) * 17
    return (torch.zeros(1, 24, t, height // 16, width // 16),
            torch.zeros(1, 32, 2, round(frames * 40 / 24)))


def plan(steps=8, low=6, **kwargs):
    return contract.plan_progressive_first_sample(*inputs(), torch.linspace(1, 0, steps + 1),
                                                  low_evaluations=low, **kwargs)


@pytest.mark.parametrize("steps", [2, 4, 8, 20])
def test_every_valid_transition_has_exact_budget(steps):
    for low in range(1, steps):
        result = plan(steps, low)
        assert result.total_evaluations == steps
        assert result.high_evaluations == steps - low
        assert result.prediction_sigma > result.resume_sigma > 0
        assert result.report()["status"] == "planned_not_executed"


@pytest.mark.parametrize("width,height", [(1024, 512), (736, 608), (608, 736)])
def test_actual_rounding_and_identity(width, height):
    values = inputs(width, height)
    before = tuple(x.clone() for x in values)
    result = contract.plan_progressive_first_sample(*values, torch.linspace(1, 0, 9), low_evaluations=6)
    assert result.low_width % 32 == result.low_height % 32 == 0
    assert result.low_width < width and result.low_height < height
    assert all(torch.equal(a, b) for a, b in zip(values, before))
    assert len(result.report()["plan_sha256"]) == 64
    assert result.report()["plan_sha256"] == copy.deepcopy(result).report()["plan_sha256"]


@pytest.mark.parametrize("field,value", [("low_scale", 1), ("low_scale", 0.1),
    ("low_scale", float("nan")), ("low_scale", True), ("task", "vdn_refine"),
    ("noise_mask", torch.zeros(1))])
def test_unsupported_inputs_fail_before_model_loading(field, value):
    with pytest.raises(ValueError):
        plan(**{field: value})


@pytest.mark.parametrize("low", [0, 8, 9, True, 2.5])
def test_transition_invalid(low):
    with pytest.raises(ValueError):
        plan(low=low)


@pytest.mark.parametrize("values", [[1., 0.], [0.9, 0.5, 0], [1., 0.5, 0.1],
    [1., 0.5, 0.5, 0.], [1., float("nan"), 0.], [1., float("inf"), 0.],
    [1., -0.1, 0.], [1., 1.1, 0.]])
def test_bad_schedules(values):
    with pytest.raises(ValueError):
        contract.plan_progressive_first_sample(*inputs(), torch.tensor(values), low_evaluations=1)


@pytest.mark.parametrize("part", [0, 1])
@pytest.mark.parametrize("value", [0.1, float("nan"), float("inf")])
def test_initial_samples_cannot_be_completed_or_invalid(part, value):
    tensors = list(inputs())
    tensors[part].flatten()[0] = value
    with pytest.raises(ValueError, match="all-zero"):
        contract.plan_progressive_first_sample(*tensors, torch.linspace(1, 0, 9), low_evaluations=6)


@pytest.mark.parametrize("t", [2, 7, 12, 22])
def test_temporal_contract(t):
    result = contract.plan_progressive_first_sample(*inputs(t=t), torch.linspace(1, 0, 9), low_evaluations=6)
    assert result.video_shape[2] == t


def test_mismatched_audio_length_rejected():
    video, audio = inputs()
    with pytest.raises(ValueError, match="audio time"):
        contract.plan_progressive_first_sample(video, audio[..., :-1], torch.linspace(1, 0, 9), low_evaluations=6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("noise_scale", [0.5, 1., 1.7])
@pytest.mark.parametrize("sigmas", [(1., .8), (.8, .4), (.2, 0.)])
def test_reused_euler_boundary_equals_next_flow_marginal(dtype, noise_scale, sigmas):
    generator = torch.Generator(device="cpu").manual_seed(8)
    clean = torch.randn(1, 24, 2, 4, 8, generator=generator, dtype=dtype)
    noise = torch.randn(clean.shape, generator=generator, dtype=dtype)
    at_prediction = contract.rectified_flow_state(clean, noise, sigmas[0], noise_scale=noise_scale)
    rebuilt = contract.euler_sampler_space_step(at_prediction, clean, *sigmas)
    oracle = contract.rectified_flow_state(clean, noise, sigmas[1], noise_scale=noise_scale)
    torch.testing.assert_close(rebuilt, oracle)


def test_audio_updates_in_native_scaled_space():
    raw_clean = torch.arange(24., dtype=torch.float64).reshape(1, 3, 2, 4)
    noise = torch.ones_like(raw_clean)
    audio_scale = 4.
    state = contract.rectified_flow_state(raw_clean * audio_scale, noise, .75)
    result = contract.euler_sampler_space_step(state, raw_clean * audio_scale, .75, .25)
    oracle = contract.rectified_flow_state(raw_clean * audio_scale, noise, .25)
    torch.testing.assert_close(result, oracle)
    # This is a sampler-space oracle, not permission to substitute raw AV state.
    incorrect = contract.euler_sampler_space_step(state, raw_clean, .75, .25)
    assert not torch.allclose(result, incorrect)


@pytest.mark.parametrize("sigma", [0., .2, .7, 1.])
@pytest.mark.parametrize("noise_scale", [.5, 1., 1.7])
def test_boundary_oracle_matches_installed_core_const(sigma, noise_scale):
    import comfy.model_sampling
    native = comfy.model_sampling.CONST()
    native.noise_scale = noise_scale
    clean = torch.arange(24., dtype=torch.float64).reshape(1, 3, 2, 4)
    noise = torch.ones_like(clean)
    actual = native.noise_scaling(torch.tensor(sigma, dtype=torch.float64), noise, clean)
    expected = contract.rectified_flow_state(clean, noise, sigma, noise_scale=noise_scale)
    torch.testing.assert_close(actual, expected)


def test_native_av_scaling_and_restart_roundtrip():
    import comfy.model_sampling
    class NativeAV(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass
    native = NativeAV()
    native.set_parameters(shift=12., audio_shift=3.)
    clean = torch.arange(32., dtype=torch.float64).reshape(1, 4, 2, 4) * native.audio_scale
    noise = torch.ones_like(clean)
    sigma = torch.tensor(.7, dtype=torch.float64)
    state = native.noise_scaling(sigma, noise, clean)
    encoded_restart = native.inverse_noise_scaling(sigma, state)
    resumed = native.noise_scaling(sigma, torch.zeros_like(state), encoded_restart)
    torch.testing.assert_close(resumed, state)
    assert native.audio_scale == 4.


def test_no_mutation_or_dtype_conversion_at_boundary():
    clean = torch.arange(12.).reshape(3, 4)
    noise = torch.zeros_like(clean)
    before = clean.clone()
    contract.rectified_flow_state(clean, noise, .5)
    assert torch.equal(clean, before)
    with pytest.raises(ValueError, match="dtype"):
        contract.rectified_flow_state(clean, noise.double(), .5)


def test_ledger_distinguishes_cfg_forwards_and_callbacks():
    ledger = contract.EvaluationLedger(plan())
    for stage, count in (("low", 6), ("high", 2)):
        for _ in range(count):
            ledger.record(stage, forward=True)
            ledger.record(stage, forward=True)
            ledger.record(stage)
    report = ledger.finish()
    assert sum(report["callbacks"].values()) == 8
    assert sum(report["actual_forwards"].values()) == 16


def test_ledger_rejects_false_success():
    ledger = contract.EvaluationLedger(plan())
    with pytest.raises(RuntimeError, match="before"):
        ledger.record("high", forward=True)
    with pytest.raises(RuntimeError, match="counts"):
        ledger.finish()
    for stage, count in (("low", 6), ("high", 2)):
        for _ in range(count):
            ledger.record(stage)
    with pytest.raises(RuntimeError, match="evidence"):
        ledger.finish()


def test_runtime_can_report_bypassed_observer_without_faking_forward_evidence(caplog):
    ledger = contract.EvaluationLedger(plan())
    with pytest.raises(RuntimeError, match='counts'):
        ledger.finish(allow_incomplete_evidence=True)
    for stage, count in (('low', 6), ('high', 2)):
        for _ in range(count):
            ledger.record(stage)
    report = ledger.finish(allow_incomplete_evidence=True)
    assert report['actual_forwards'] == {'low': 0, 'high': 0}
    assert report['callbacks'] == {'low': 6, 'high': 2}
    assert report['forward_evidence_complete'] is False
    assert 'continuing' in caplog.text
    with pytest.raises(RuntimeError, match='evidence'):
        ledger.finish()  # Strict persisted completion evidence remains strict.
