import json
from pathlib import Path

import pytest
import torch

from tools.progressive_probe_extension import ConditionAudit, ModelAudit, NativeBaseline, TimedDecode, value_identity
from test_progressive_sampling_runtime import tiny_model, latent, conditioning
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from h3_audio_t8_pkg.audio_ops import decode_av_latent
from helpers import FakeVideoVAE, FakeAudioVAE


@pytest.fixture(autouse=True)
def registered_project(monkeypatch):
    import tools.progressive_probe_extension as extension
    monkeypatch.setattr(extension, "_allocator_session", None)
    import folder_paths
    monkeypatch.syspath_prepend(str(Path(folder_paths.__file__).parent))
    import nodes
    from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3ProgressiveSamplerEXPT8", MiniMaxH3ProgressiveSamplerEXPT8)


def test_encoded_identity_binds_values_without_changing_conditioning():
    source, positive = latent(), conditioning()
    output = ConditionAudit.execute(positive, source).result
    assert output[0] is positive and output[1] is source
    before = json.loads(output[2])["conditioning_sha256"]
    positive[0][0][0, 0, 0] = 1
    after = json.loads(ConditionAudit.execute(positive, source).result[2])["conditioning_sha256"]
    assert before != after
    assert value_identity(torch.ones(2, dtype=torch.bfloat16))["dtype"] == "torch.bfloat16"


@pytest.mark.parametrize("missed", [0, 1])
def test_lora_audit_checks_actual_model_before_sampling(missed):
    from types import SimpleNamespace
    model = SimpleNamespace(patches={str(i): object() for i in range(259)})
    report = {"status": "applied", "selected_strength": 1., "applied_patch_count": 259,
              "patch_target_count": 259, "missed_patch_target_count": missed}
    if missed:
        with pytest.raises(RuntimeError, match="sampling is blocked"):
            ModelAudit.execute(model, json.dumps(report), 259)
    else:
        assert ModelAudit.execute(model, json.dumps(report), 259).result[0] is model
        model.patches.pop("0")
        with pytest.raises(RuntimeError, match="sampling is blocked"):
            ModelAudit.execute(model, json.dumps(report), 259)


def test_baseline_observes_real_core_nodes_without_changing_original():
    source = latent()
    model, sampler, sigmas = setup_dual_clock_sampling(tiny_model(), source, 8, 12., 3., "euler")
    output, raw = NativeBaseline.execute(model, conditioning(), source, sampler, sigmas, 1).result
    assert output["samples"].is_nested
    assert json.loads(raw)["actual_network_forwards"] == 8
    assert not model.wrappers


def test_baseline_failure_does_not_leave_an_original_model_wrapper(monkeypatch):
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
    def failed(*args):
        raise RuntimeError("test sampler failure")
    monkeypatch.setattr(SamplerCustomAdvanced, "execute", failed)
    source = latent()
    model, sampler, sigmas = setup_dual_clock_sampling(tiny_model(), source, 8, 12., 3., "euler")
    with pytest.raises(RuntimeError, match="test sampler failure"):
        NativeBaseline.execute(model, conditioning(), source, sampler, sigmas, 1)
    assert not model.wrappers


def test_decode_instrumentation_is_numerically_identical_to_existing_helper():
    source, video_vae, audio_vae = latent(), FakeVideoVAE(), FakeAudioVAE()
    expected_frames, expected_audio, *_ = decode_av_latent(source, video_vae, audio_vae)
    frames, audio, raw = TimedDecode.execute(source, video_vae, audio_vae).result
    assert torch.equal(frames, expected_frames)
    assert torch.equal(audio["waveform"], expected_audio["waveform"])
    assert [row["stage"] for row in json.loads(raw)["timings"]] == ["video_vae_decode", "audio_vae_decode"]


def test_allocator_node_roundtrip_in_real_cpu_core_never_claims_gpu():
    import os
    from tools.progressive_probe_extension import AllocatorAudit
    from tools.progressive_memory_metrics import validate_completed_interval
    begin = json.loads(AllocatorAudit.execute("cpu-unit", "begin", "cpu", os.getpid()).result[0])
    assert begin["status"] == "unavailable" and not torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="still open"):
        AllocatorAudit.execute("other", "begin", "cpu", os.getpid())
    with pytest.raises(RuntimeError, match="identity does not match"):
        AllocatorAudit.execute("other", "finish", "cpu", os.getpid())
    report = json.loads(AllocatorAudit.execute("cpu-unit", "finish", "cpu", os.getpid()).result[0])
    validate_completed_interval(report, run_id="cpu-unit", pid=os.getpid(), device_type="cpu")
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("action, mode, pid", [("begin", "cpu", -1), ("begin", "cuda", None), ("unknown", "cpu", None)])
def test_allocator_node_refuses_wrong_owner_mode_and_action(action, mode, pid):
    import os
    from tools.progressive_probe_extension import AllocatorAudit
    with pytest.raises((RuntimeError, ValueError)):
        AllocatorAudit.execute("invalid", action, mode, os.getpid() if pid is None else pid)


def test_allocator_abort_retains_failure_not_complete():
    import os
    from tools.progressive_probe_extension import AllocatorAudit
    AllocatorAudit.execute("abort", "begin", "cpu", os.getpid())
    report = json.loads(AllocatorAudit.execute("abort", "abort", "cpu", os.getpid()).result[0])
    assert report["outcome"] == "failed" and report["qualified_interval_counters"] is None


def test_decoder_boundaries_are_observed_without_changing_frame_or_audio(monkeypatch):
    from types import SimpleNamespace
    import tools.progressive_probe_extension as extension
    boundaries = []
    monkeypatch.setattr(extension, "_allocator_session", SimpleNamespace(state="active", observe=boundaries.append))
    source, video_vae, audio_vae = latent(), FakeVideoVAE(), FakeAudioVAE()
    expected_frames, expected_audio, *_ = decode_av_latent(source, video_vae, audio_vae)
    frames, audio, _ = TimedDecode.execute(source, video_vae, audio_vae).result
    assert torch.equal(frames, expected_frames) and torch.equal(audio["waveform"], expected_audio["waveform"])
    assert boundaries == ["sampling_output_ready_before_decode", "before_video_vae_decode",
                           "before_audio_vae_decode", "both_decoders_complete"]
