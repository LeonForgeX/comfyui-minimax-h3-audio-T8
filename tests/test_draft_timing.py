"""Exclusive wall-clock buckets include draft work; no GPU timing claim."""
from h3_audio_t8_pkg import progressive_sampling_composed as composed
from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_first_pass_draft import ProgressiveFirstPassDraftSession
from test_progressive_first_pass_draft import run
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def test_draft_work_has_exclusive_buckets_and_final_verify_is_in_total(tmp_path, stub_lifter, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(composed.time, 'perf_counter', lambda: now[0])
    bind = ProgressiveFirstPassDraftSession._bind_and_load_low
    save = ProgressiveFirstPassDraftSession.save_low
    verify = ProgressiveFirstPassDraftSession.verify
    stage = runtime._native_stage

    def bind_with_cost(self, *args, **kwargs):
        result = bind(self, *args, **kwargs)
        now[0] += 11
        return result

    def save_with_cost(self, *args, **kwargs):
        result = save(self, *args, **kwargs)
        now[0] += 5
        return result

    def verify_with_cost(self, *args, **kwargs):
        result = verify(self, *args, **kwargs)
        now[0] += 7
        return result

    def stage_with_cost(*args, **kwargs):
        result = stage(*args, **kwargs)
        now[0] += 3
        return result

    monkeypatch.setattr(ProgressiveFirstPassDraftSession, '_bind_and_load_low', bind_with_cost)
    monkeypatch.setattr(ProgressiveFirstPassDraftSession, 'save_low', save_with_cost)
    monkeypatch.setattr(ProgressiveFirstPassDraftSession, 'verify', verify_with_cost)
    monkeypatch.setattr(runtime, '_native_stage', stage_with_cost)
    _, report = run(tmp_path)
    times = report['timings']
    assert times['draft_bind_and_load_seconds'] >= 11
    assert times['draft_save_and_verify_seconds'] == 19
    assert times['draft_final_verify_seconds'] == 7
    assert times['low_sampling_including_model_prepare'] == 3
    assert times['high_sampling_including_model_prepare'] == 3
    assert report['sampler_seconds'] == sum(times.values())
    assert report['first_pass_draft']['saved_low'] is True
    _, replay = run(tmp_path, mode='reuse_only')
    assert 'draft_save_and_verify_seconds' not in replay['timings']
    assert replay['timings']['draft_final_verify_seconds'] == 7
    assert replay['counts']['actual_forwards']['low'] == 0
