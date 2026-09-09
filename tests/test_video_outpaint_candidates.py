from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest
import torch
import comfy.nested_tensor

from h3_audio_t8_pkg.video_outpaint_candidates import (
    candidate_receipt, continue_selected_candidate, sample_first_candidate, select_candidate,
)
from h3_audio_t8_pkg.video_outpaint_sampling_runtime import sample_prepared_outpaint_windows
from h3_audio_t8_pkg.video_outpaint_window_store import OutpaintWindowStore
from test_video_outpaint_sampling_runtime import _execution


def _noise_backend(*args, **kwargs):
    video, audio = args[1].unbind()
    return comfy.nested_tensor.NestedTensor((video.clone(), audio.clone()))


def _fixture(path, *, cuts=(), algorithm="t8.outpaint.native_cpu_noise/v1"):
    inputs = _execution(path, cuts=cuts, noise_algorithm=algorithm)
    inputs["sample_function"] = _noise_backend
    original = inputs["audio_for_window"]

    def audio(shot, local):
        samples = original(shot, local)
        return {"samples": samples, "noise_mask": torch.ones((1, 1, 2, samples.shape[-1]))}

    inputs["audio_for_window"] = audio
    return inputs


@pytest.mark.parametrize("algorithm", ["t8.outpaint.coordinate_noise/v1", "t8.outpaint.native_cpu_noise/v1"])
def test_candidate_pause_selection_and_continuation_equal_uninterrupted_av(tmp_path, algorithm):
    control = _fixture(tmp_path / "control", algorithm=algorithm)
    sample_prepared_outpaint_windows(**control)
    inputs = _fixture(tmp_path / "candidate", algorithm=algorithm)
    calls = []
    candidate, report = sample_first_candidate(**inputs, progress=calls.append)
    store = inputs["window_store"]
    assert len(calls) == 1
    assert report["windows_sampled_this_call"] == 1 and report["awaiting_explicit_selection"]
    assert not report["all_windows_sampled"] and not report["generated_video_complete"]
    assert store.snapshot()["status"] == "paused"
    assert len(store.snapshot()["committed"]) == 1
    before = store.load(0)
    with pytest.raises(ValueError, match="explicit resume"):
        sample_prepared_outpaint_windows(**inputs)
    selection = select_candidate(candidate, store)
    result = continue_selected_candidate(selection=selection, **inputs)
    assert result["resume_from"] == 1 and result["windows_sampled_this_call"] == 1
    assert result["all_windows_sampled"] and result["selected_prefix_reused"]
    assert not result["perceptual_acceptance"]
    assert all(torch.equal(a, b) for a, b in zip(before, store.load(0)))
    for index in range(2):
        assert all(torch.equal(a, b) for a, b in zip(control["window_store"].load(index), store.load(index)))
    assert continue_selected_candidate(selection=selection, **inputs)["windows_sampled_this_call"] == 0


def test_candidate_noop_retry_and_descriptor_do_not_choose_or_resample(tmp_path):
    inputs = _fixture(tmp_path)
    candidate, _ = sample_first_candidate(**inputs)
    before = inputs["window_store"].path.read_bytes()
    repeated, report = sample_first_candidate(**inputs, resume=True)
    assert repeated == candidate and report["windows_sampled_this_call"] == 0
    assert inputs["window_store"].path.read_bytes() == before
    assert "selection" not in candidate
    with pytest.raises(ValueError, match="explicit candidate selection"):
        continue_selected_candidate(selection=None, **inputs)


@pytest.mark.parametrize("field", ["sha256", "identity", "prefix", "first_frame_index", "preview_scope", "extra"])
def test_altered_candidate_cannot_be_selected(tmp_path, field):
    inputs = _fixture(tmp_path)
    candidate, _ = sample_first_candidate(**inputs)
    candidate[field] = "altered"
    with pytest.raises(ValueError, match="candidate"):
        select_candidate(candidate, inputs["window_store"])


def test_other_seed_cannot_use_selected_candidate(tmp_path):
    inputs = _fixture(tmp_path / "a")
    candidate, _ = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    required = {k: v for k, v in store.identity.items() if k not in {"implementation_sha256", "plan_sha256"}}
    required["seed"] += 1
    other = OutpaintWindowStore(tmp_path / "b", store.plan, execution_identity=required)
    with pytest.raises(ValueError, match="identity"):
        select_candidate(candidate, other)


def test_corrupted_selected_asset_is_rejected_without_sampling(tmp_path):
    inputs = _fixture(tmp_path)
    candidate, _ = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    selection = select_candidate(candidate, store)
    asset = store.root / f"window-{candidate['prefix'][0]['sha256']}.safetensors"
    asset.write_bytes(b"bad")
    with pytest.raises(ValueError, match="truncated"):
        continue_selected_candidate(selection=selection, **inputs)
    assert len(store.snapshot()["committed"]) == 1


def test_failed_candidate_can_retry_but_is_not_selectable_until_success(tmp_path):
    inputs = _fixture(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("sampler failed")

    inputs["sample_function"] = fail
    with pytest.raises(RuntimeError, match="sampler failed"):
        sample_first_candidate(**inputs)
    store = inputs["window_store"]
    assert store.snapshot()["status"] == "interrupted"
    with pytest.raises(ValueError, match="successfully paused"):
        candidate_receipt(store)
    inputs["sample_function"] = _noise_backend
    candidate, _ = sample_first_candidate(**inputs, resume=True)
    assert select_candidate(candidate, store)


def test_cancel_after_candidate_commit_is_not_a_successful_pause(tmp_path):
    inputs = _fixture(tmp_path)

    def cancel(report):
        raise RuntimeError("cancel after commit")

    with pytest.raises(RuntimeError, match="cancel"):
        sample_first_candidate(**inputs, progress=cancel)
    store = inputs["window_store"]
    assert store.snapshot()["status"] == "interrupted" and len(store.snapshot()["committed"]) == 1
    with pytest.raises(ValueError, match="successfully paused"):
        candidate_receipt(store)
    _, result = sample_first_candidate(**inputs, resume=True)
    assert result["windows_sampled_this_call"] == 0 and store.snapshot()["status"] == "paused"


def test_cut_after_selected_window_does_not_carry_context_to_new_shot(tmp_path):
    inputs = _fixture(tmp_path, cuts=(45,))
    candidate, _ = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    assert store.context_for(1) is None
    seen = []

    def backend(*args, **kwargs):
        seen.append(args[6][0][1])
        return _noise_backend(*args, **kwargs)

    inputs["sample_function"] = backend
    continue_selected_candidate(selection=select_candidate(candidate, store), **inputs)
    assert len(seen) == 1 and "minimax_keyframes" not in seen[0]


def test_selection_and_live_identity_are_rechecked(tmp_path):
    inputs = _fixture(tmp_path)
    candidate, _ = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    selection = select_candidate(candidate, store)
    changed = deepcopy(selection)
    changed["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="selection integrity"):
        continue_selected_candidate(selection=changed, **inputs)
    inputs["verify_execution"] = lambda: {}
    with pytest.raises(ValueError, match="identity"):
        continue_selected_candidate(selection=selection, **inputs)
    assert len(store.snapshot()["committed"]) == 1


def test_old_completed_store_is_byte_unchanged_and_reopens(tmp_path):
    inputs = _fixture(tmp_path)
    store = inputs["window_store"]
    sample_prepared_outpaint_windows(**inputs)
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    reopened = _fixture(tmp_path)["window_store"]
    assert reopened.identity == store.identity and reopened.snapshot()["status"] == "sampled"
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="at most the first"):
        sample_first_candidate(**inputs)
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before


def test_empty_store_cannot_pause_or_be_selected(tmp_path):
    store = _fixture(tmp_path)["window_store"]
    with pytest.raises(ValueError, match="running committed"):
        store.pause()
    with pytest.raises(ValueError, match="successfully paused"):
        candidate_receipt(store)


def test_single_window_candidate_still_requires_selection_without_false_incomplete_state(tmp_path, monkeypatch):
    import test_video_outpaint_sampling_runtime as fixture_module
    from h3_audio_t8_pkg.video_outpaint_plan import build_outpaint_plan
    original_plan = fixture_module._plan
    monkeypatch.setattr(fixture_module, "_plan", lambda cuts: build_outpaint_plan(
        **dict(original_plan(cuts)["request"], window_frames=107)))
    inputs = _fixture(tmp_path)
    candidate, report = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    assert store.snapshot()["status"] == "sampled" and len(store.windows) == 1
    assert report["all_windows_sampled"] and report["awaiting_explicit_selection"]
    assert not report["generated_video_complete"]
    selected = select_candidate(candidate, store)
    assert continue_selected_candidate(selection=selected, **inputs)["windows_sampled_this_call"] == 0


def test_native_provider_candidate_chain_binds_real_model_and_reuses_selected_prefix(tmp_path):
    from test_video_outpaint_execution import _chain
    from h3_audio_t8_pkg.video_outpaint_candidate_execution import (
        sample_verified_first_candidate, continue_verified_candidate,
    )
    inputs = _chain(tmp_path)
    store, candidate, report = sample_verified_first_candidate(**inputs)
    assert not report["caller_supplied_identity_trusted"] and store.snapshot()["status"] == "paused"
    selected = select_candidate(candidate, store)
    before = store.load(0)
    finished, result = continue_verified_candidate(selection=selected, **inputs)
    assert result["resume_from"] == 1 and result["all_windows_sampled"]
    assert all(torch.equal(a, b) for a, b in zip(before, finished.load(0)))


def test_native_candidate_rejects_model_changed_after_selection(tmp_path):
    from test_video_outpaint_execution import _chain
    from h3_audio_t8_pkg.video_outpaint_candidate_execution import (
        sample_verified_first_candidate, continue_verified_candidate,
    )
    inputs = _chain(tmp_path)
    store, candidate, _ = sample_verified_first_candidate(**inputs)
    selected = select_candidate(candidate, store)
    with torch.no_grad():
        inputs["model"].model.diffusion_model.weight += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        continue_verified_candidate(selection=selected, **inputs)
    assert store.snapshot()["status"] == "paused"


def test_continuation_conditions_on_selected_generated_av_not_fresh_noise(tmp_path):
    inputs = _fixture(tmp_path)

    def distinct_candidate(*args, **kwargs):
        video, audio = args[1].unbind()
        return comfy.nested_tensor.NestedTensor((video + 37, audio + 51))

    inputs["sample_function"] = distinct_candidate
    candidate, _ = sample_first_candidate(**inputs)
    store = inputs["window_store"]
    expected = store.context_for(1)
    captured = []

    def continuation(*args, **kwargs):
        video, audio = args[8].unbind()
        vt, at = expected["video_tail"].shape[2], expected["audio_tail"].shape[-1]
        assert torch.equal(video[:, :, :vt], expected["video_tail"])
        assert torch.equal(audio[..., :at], expected["audio_tail"])
        keyframe = args[6][0][1]["minimax_keyframes"][0]
        assert torch.equal(keyframe["latent"], expected["video_tail"])
        assert torch.equal(keyframe["audio_latent"], expected["audio_tail"])
        captured.append(True)
        return _noise_backend(*args, **kwargs)

    inputs["sample_function"] = continuation
    continue_selected_candidate(selection=select_candidate(candidate, store), **inputs)
    assert captured == [True]
    video, audio = store.load(1)
    assert torch.equal(video[:, :, :expected["video_tail"].shape[2]], expected["video_tail"])
    assert torch.equal(audio[..., :expected["audio_tail"].shape[-1]], expected["audio_tail"])
