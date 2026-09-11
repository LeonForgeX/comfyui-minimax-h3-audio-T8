import json
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import long_video_dual_model_runner as runner
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import _write_effects_audit


def av(width, height, value=0.):
    video = torch.full((1, 24, 7, height // 16, width // 16), value)
    audio = torch.full((1, 32, 2, 20), value)
    return {"samples": NestedTensor((video, audio))}


def test_partial_four_default_does_not_freeze_unfinished_audio(rig):
    engine, run, *_ = rig
    result = run()
    assert engine.audio == ('legacy_policy', 0.)
    mask = result['sampled'].get('noise_mask')
    assert mask is None or torch.all(mask.unbind()[1] == 1)


def test_execution_timings_do_not_count_cached_sampling_as_new_work(rig):
    _, run, *_ = rig
    fresh = run()['sampling_report']['dual_model']['execution_timings']
    assert [event['phase'] for event in fresh['events']] == [
        'first_conditioning', 'first_sampling', 'second_conditioning',
        'learned_upscale', 'reconcile', 'second_sampling']
    assert all(event['seconds'] >= 0 and event['status'] == 'completed' for event in fresh['events'])
    cached = run()['sampling_report']['dual_model']['execution_timings']
    assert [event['phase'] for event in cached['events']] == ['second_conditioning']
    assert cached['historical_cache_time_counted_as_current'] is False


@pytest.mark.parametrize('resume_prepared', [False, True])
def test_audio_context_locks_only_prefix_without_erasing_coarse_new_audio(
        rig, monkeypatch, resume_prepared):
    engine, run, _, fail, _, second = rig
    original = engine._conditions

    def conditions(model, context, inputs, projected):
        values = list(original(model, context, inputs, projected))
        if model is second:
            latent = values[2]
            video, audio = latent['samples'].unbind()
            mask = torch.ones_like(audio)
            mask[..., :4] = 0
            latent['samples'] = NestedTensor((video, torch.full_like(audio, 99)))
            latent['noise_mask'] = NestedTensor((torch.ones_like(video), mask))
        return tuple(values)

    monkeypatch.setattr(engine, '_conditions', conditions)
    if resume_prepared:
        fail['high'] = True
        with pytest.raises(RuntimeError, match='second pass failure'):
            run()
        fail['high'] = False
    result = run()
    audio = result['sampled']['samples'].unbind()[1]
    mask = result['sampled']['noise_mask'].unbind()[1]
    assert torch.all(audio[..., :4] == 99)
    assert torch.all(audio[..., 4:] == 1)
    assert torch.all(mask[..., :4] == 0)
    assert torch.all(mask[..., 4:] == 1)


def test_partial_four_old_zero_lock_request_is_migrated():
    engine = runner.DualModelSegmentRunner(object(), object(), contract={},
        low_width=64, low_height=32, upscaler_model='fixture', coarse_steps=4,
        second_audio_source='first_pass', second_audio_strength=0.)
    assert engine.audio == ('legacy_policy', 0.)
    assert engine.audio_policy['migration'] == 'partial_first_pass_zero_lock_to_joint_continuation'


def test_stock20_auto_keeps_complete_first_pass_audio():
    engine = runner.DualModelSegmentRunner(object(), object(), contract={},
        low_width=64, low_height=32, upscaler_model='fixture', coarse_steps=20)
    assert engine.audio == ('first_pass', 0.)
    assert engine.audio_policy['first_pass_complete_trajectory'] is True


def test_failed_second_pass_does_not_publish_partial_audio_context(rig, tmp_path):
    _, run, _, fail, *_ = rig
    fail['high'] = True
    with pytest.raises(RuntimeError, match='second pass failure'):
        run()
    assert not list(tmp_path.rglob('low.context.safetensors'))


def test_audio_policy_change_does_not_reuse_previous_stage_cache(rig):
    engine, run, calls, *_ = rig
    run()
    engine.audio_policy = {**engine.audio_policy, 'test_new_contract': True}
    before = len(calls)
    result = run()
    assert [x[0] for x in calls[before:]].count('sample') == 2
    assert result['sampling_report']['dual_model']['high_reused'] is False


def test_low_context_uses_finished_second_pass_audio(rig, monkeypatch, tmp_path):
    _, run, _, _, _, second = rig
    original = runner.sample_model_stage
    def finish_audio(model, *args, **kwargs):
        result, report = original(model, *args, **kwargs)
        if model is second:
            video, audio = result['samples'].unbind()
            result['samples'] = NestedTensor((video, audio + 7))
        return result, report
    monkeypatch.setattr(runner, 'sample_model_stage', finish_audio)
    result = run()
    record = result['sampling_report']['dual_model']['low_context']
    context, _ = runner._load_accepted_context_file(tmp_path / record['path'], 'test', 0, 1)
    assert torch.all(context['audio_tail'] == 8)
    assert torch.all(context['video_tail'] == 2)
    assert record['audio_source'] == 'completed_second_pass_output'
    receipt = json.loads(next(tmp_path.rglob('high_output-*.json')).read_text())
    assert receipt['contract']['audio_policy_version'] == 2


@pytest.fixture
def rig(monkeypatch, tmp_path):
    calls = []
    first, second = object(), object()
    def condition(self, model, context, inputs, projected):
        calls.append(("condition", model, inputs["width"], context))
        latent = av(inputs["width"], inputs["height"])
        positive = [[torch.zeros(1), {"minimax_keyframes": [{"latent": latent["samples"].unbind()[0]}]}]]
        return model, positive, latent, None, "prompt", "{}", {"status": "disabled"}
    monkeypatch.setattr(runner.DualModelSegmentRunner, "_conditions", condition)
    monkeypatch.setattr(runner, "release_stage_residency", lambda *components: {'fixture': 'release'})
    monkeypatch.setattr(runner, "setup_dual_clock_sampling", lambda model, latent, *a: (model, object(), torch.linspace(1, 0, 21)))
    monkeypatch.setattr(runner, "build_learned_two_pass_parity_plan", lambda *a: (
        torch.linspace(1, .5, 5), torch.linspace(.9, 0, 5), "{}"))
    fail = {"high": False}
    def sample(model, positive, latent, **kwargs):
        calls.append(("sample", model, kwargs["output_kind"] if "output_kind" in kwargs else "denoised_x0"))
        if model is second and fail["high"]:
            raise RuntimeError("simulated second pass failure")
        result = dict(latent)
        result["samples"] = NestedTensor((latent["samples"].unbind()[0] + 2, latent["samples"].unbind()[1] + (1 if model is first else 0)))
        return result, {"fixture": "sample"}
    monkeypatch.setattr(runner, "sample_model_stage", sample)
    def upscale(latent, name, size_mode, scale, mp, width, height, policy, anisotropy, precision, release):
        calls.append(("upscale", name, width, height))
        assert policy == "honor_dimensions_exp" and release == "offload_after"
        output = av(width, height, 2.)
        output["samples"] = NestedTensor((output["samples"].unbind()[0], latent["samples"].unbind()[1]))
        return output, width, height, "{}"
    monkeypatch.setattr(runner, "learned_upscale_h3_av_latent", upscale)
    engine = runner.DualModelSegmentRunner(first, second, contract={"first_model": {"sha256": "first"}},
        low_width=64, low_height=32, upscaler_model="actual-test-fixture-name")
    def run(index=0, candidate="candidate0", parent="", context=None, context_frames=22):
        return engine.run(root=tmp_path, chain_id="test", job_sha256="job", segment=SimpleNamespace(
            index=index, seed=7 + index, plan=SimpleNamespace(save_context=True, context_frames=context_frames)),
            candidate_id=candidate, base_candidate_id=candidate, high_context=context or {"empty": True},
            parent_candidate_id=parent, parent_revision=index, projected_plan=None,
            inputs={"width": 128, "height": 64, "clip": object(), "video_vae": object(), "audio_vae": object()})
    return engine, run, calls, fail, first, second


@pytest.mark.parametrize('resume_high_cache', [False, True])
def test_next_segment_receives_completed_audio_after_normal_or_cached_output(
        rig, monkeypatch, tmp_path, resume_high_cache):
    _, run, calls, _, first, second = rig
    original = runner.sample_model_stage

    def finish_audio(model, *args, **kwargs):
        result, report = original(model, *args, **kwargs)
        if model is second:
            video, audio = result['samples'].unbind()
            result['samples'] = NestedTensor((video, audio + 7))
        return result, report

    monkeypatch.setattr(runner, 'sample_model_stage', finish_audio)
    result = run()
    if resume_high_cache:
        before = len(calls)
        result = run()
        assert not any(item[0] == 'sample' for item in calls[before:])
        assert result['sampling_report']['dual_model']['high_reused'] is True
    candidate = tmp_path / 'candidates/segment_00000/candidate0/candidate.json'
    _write_effects_audit(str(candidate), {'contract_sha256': 'job', 'segment_index': 0,
        'candidate_id': 'candidate0', 'sampling_plan': result['sampling_report']})
    high_context = {'empty': False,
        'video_tail': result['sampled']['samples'].unbind()[0],
        'audio_tail': result['sampled']['samples'].unbind()[1]}
    before = len(calls)
    run(1, 'candidate1', 'candidate0', high_context)
    conditions = [item for item in calls[before:] if item[0] == 'condition']
    assert [item[1] for item in conditions] == [first, second]
    low_context = conditions[0][3]
    assert torch.all(low_context['audio_tail'] == 8)
    assert torch.all(low_context['video_tail'] == 2)
    assert torch.equal(low_context['audio_tail'], high_context['audio_tail'])
    assert conditions[1][3] is high_context


def test_stages_are_serial_and_rebuild_high_conditioning(rig):
    _, run, calls, _, first, second = rig
    result = run()
    assert [item[0] for item in calls] == ["condition", "sample", "condition", "upscale", "sample"]
    assert calls[0][1:3] == (first, 64) and calls[2][1:3] == (second, 128)
    assert calls[1][2] == "denoised_x0" and calls[-1][2] == "zero_sigma_output"
    assert result["sampling_report"]["dual_model"]["low_context"]["video_shape"][-2:] == [2, 4]
    assert list(result["sampled"]["samples"].unbind()[0].shape[-2:]) == [4, 8]
    assert 'noise_mask' not in result['sampled']
    assert torch.all(result["sampled"]["samples"].unbind()[1] == 1)


def test_second_failure_resumes_saved_x0_and_upscale_without_resampling(rig):
    _, run, calls, fail, first, second = rig
    fail["high"] = True
    with pytest.raises(RuntimeError, match="second pass failure"):
        run()
    before = len(calls)
    fail["high"] = False
    result = run()
    later = calls[before:]
    assert [item[0] for item in later] == ["condition", "sample"]
    assert later[0][1] is later[1][1] is second
    assert result["sampling_report"]["dual_model"]["low_reused"] is True
    before = len(calls)
    result = run()
    assert [item[0] for item in calls[before:]] == ["condition"]
    assert result["sampling_report"]["dual_model"]["high_reused"] is True


def test_next_segment_uses_separate_low_and_high_contexts(rig, tmp_path):
    _, run, calls, _, first, second = rig
    result = run()
    path = tmp_path / "candidates/segment_00000/candidate0/candidate.json"
    _write_effects_audit(str(path), {"contract_sha256": "job", "segment_index": 0,
        "candidate_id": "candidate0", "sampling_plan": result["sampling_report"]})
    high_context = {"empty": False, "video_tail": torch.full((1, 24, 7, 4, 8), 9.)}
    before = len(calls)
    run(1, "candidate1", "candidate0", high_context)
    conditions = [item for item in calls[before:] if item[0] == "condition"]
    assert conditions[0][1] is first and conditions[1][1] is second
    assert conditions[0][3]["video_tail"].shape[-2:] == (2, 4)
    assert torch.all(conditions[0][3]["video_tail"] == 2)
    assert conditions[1][3] is high_context
    receipt = json.loads(path.with_name("effects_audit.json").read_text())
    assert receipt["sampling_plan"]["dual_model"]["low_context"]


def test_eav_cannot_masquerade_as_four_step_stock20():
    with pytest.raises(ValueError, match="EAV requires full Stock20"):
        runner.DualModelSegmentRunner(object(), object(), contract={}, low_width=64, low_height=32,
                                     upscaler_model="x", coarse_steps=4, eav_config={"mode": "apply_exp"})


@pytest.mark.parametrize('abort', [False, True])
def test_eav_backend_report_requires_successful_finalization(rig, monkeypatch, tmp_path, abort):
    engine, run, _, _, first, _ = rig
    engine.steps = (20, 4)
    engine.eav_config = {'mode': 'apply_exp'}
    token = object()
    backend = {'kind': 'fixture', 'completed_calls': {'sage:unbiased': 1000}}
    monkeypatch.setattr(runner, 'build_eav_long_video_model', lambda *a, **k: (first, token, '{}'))
    def finalize(latent, runtime):
        assert runtime is token
        if abort:
            raise RuntimeError('EAV audit rejected')
        return latent, json.dumps({'config': {'composed_attention_backend': backend}})
    monkeypatch.setattr(runner, 'finalize_eav_runtime', finalize)
    if abort:
        with pytest.raises(RuntimeError, match='EAV audit rejected'):
            run()
        assert not list(tmp_path.rglob('low_x0-*.json'))
    else:
        report = run()['sampling_report']['dual_model']['first_pass']
        assert report['backend'] == backend
        assert 'after successful EAV finalization' in report['backend_observation']


def test_relay_conditions_return_the_same_report_type_as_plain(monkeypatch):
    model = object()
    monkeypatch.setattr(runner, 'patch_long_video_model', lambda value: value)
    report = {'long_video_report': {'task': 't2va', 'width': 128}, 'status': 'applied_exp'}
    def bind(incoming, builder, **kwargs):
        assert incoming is model and builder is runner.build_prompt_relay_long_video_conditioning
        assert 'prompt' not in kwargs and kwargs['prompt_relay_plan'] == {'plan_hash': 'test'}
        return model, [], av(128, 64), None, 'prompt', [], json.dumps(report)
    monkeypatch.setattr(runner, 'bind_stage_conditioning', bind)
    engine = runner.DualModelSegmentRunner(model, model, contract={}, low_width=64, low_height=32,
        upscaler_model='fixture', prompt_relay_mode='apply_exp')
    result = engine._conditions(model, {'empty': True}, {'prompt': 'old'}, {'plan_hash': 'test'})
    assert json.loads(result[5]) == report['long_video_report']
    assert result[6] == report


@pytest.mark.parametrize('delta,allowed', [(1e-7, True), (.01, False)])
def test_audio_delivery_restores_only_expected_normalization_roundoff(rig, monkeypatch, delta, allowed):
    engine, run, _, _, first, second = rig
    # Exact restoration is only valid after a complete first trajectory.
    engine.steps = (20, 4)
    engine.audio = ('first_pass', 0.)
    original = runner.sample_model_stage
    def sample(model, *args, **kwargs):
        result, report = original(model, *args, **kwargs)
        if model is second:
            video, audio = result['samples'].unbind()
            result['samples'] = NestedTensor((video, audio + delta))
        return result, report
    monkeypatch.setattr(runner, 'sample_model_stage', sample)
    if not allowed:
        with pytest.raises(RuntimeError, match='normalization roundtrip'):
            run()
    else:
        result = run()
        assert torch.all(result['sampled']['samples'].unbind()[1] == 1.)
        assert result['sampling_report']['dual_model']['second_pass']['audio_delivery']['source'] == 'first_pass_exact_latent'
