"""Real tiny Core H3/Euler plumbing; learned weights are a labelled test double.

These CPU tests do not qualify trained-model quality, CUDA memory or speed.
"""

import copy
import json

import pytest
import torch
import torch.nn.functional as F

import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.ops
import comfy.samplers
import comfy.supported_models_base

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_sampling_contract import plan_progressive_first_sample
from h3_audio_t8_pkg.sampling import native_flow_sigmas, setup_dual_clock_sampling


def tiny_model():
    class Config(comfy.supported_models_base.BASE):
        latent_format = comfy.latent_formats.MiniMaxH3AV
        unet_extra_config = {}
        sampling_settings = {"shift": 12., "audio_shift": 3.}
        custom_operations = comfy.ops.disable_weight_init

    config = Config(dict(hidden_size=8, num_layers=1, token_refiner_num_layers=0,
                         num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=16,
                         text_dim=8, timestep_input_dim=4, time_embed_hidden_size=8,
                         time_embed_dim=4, rope_inv_freq_len=1, dtype=torch.float32))
    base = comfy.model_base.MiniMaxH3(config, device=torch.device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(43)
    for parameter in base.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .03)
    base.diffusion_model.rope.inv_freq.fill_(1.)
    return comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))


def latent():
    return {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 8), torch.zeros(1, 32, 2, 8)))}


def conditioning(task="t2va"):
    metadata = {"minimax_token_tags": torch.ones(2, dtype=torch.long)}
    if task == "i2va":
        metadata.update(minimax_keyframes=[{"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 4, 8) * .2}],
                        minimax_frame_count=5)
        metadata["minimax_token_tags"][0] = 0
    return [[torch.zeros(1, 2, 8), metadata]]


@pytest.fixture
def stub_lifter(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "_lifter_identity", lambda name, precision:
                        {"name": name, "precision": precision, "path": "test-double", "sha256": "a" * 64})

    def lift(video, audio, plan, identity):
        calls.append((video.shape, audio.shape, identity))
        result = F.interpolate(video.float(), size=(video.shape[2], plan.target_height // 16, plan.target_width // 16),
                               mode="trilinear", align_corners=False)
        return result, {"test_double": True, "trained_lifter_executed": False}

    monkeypatch.setattr(runtime, "_lift_video", lift)
    return calls


@pytest.mark.parametrize("task", ["t2va", "i2va"])
@pytest.mark.parametrize("steps,low", [(2, 1), (4, 3), (8, 6)])
def test_real_tiny_core_euler_runs_both_stages_and_preserves_input(stub_lifter, task, steps, low):
    model = tiny_model()
    positive = conditioning(task)
    metadata_before = copy.deepcopy(positive[0][1])
    options_before = copy.deepcopy(model.model_options)
    source = latent()
    notifications = []
    output, text = runtime.sample_progressive_h3(
        model, positive, conditioning(), source, comfy.samplers.ksampler("euler"), native_flow_sigmas(steps, 12.),
        upscaler_model="test", seed=1, low_evaluations=low, task=task,
        callback=lambda step, _prediction, _state, total: notifications.append((step, total)))
    assert [tuple(v.shape) for v in output["samples"].unbind()] == [(1, 24, 2, 4, 8), (1, 32, 2, 8)]
    assert all(bool(torch.isfinite(value).all()) for value in output["samples"].unbind())
    assert all(not bool(torch.count_nonzero(value)) for value in source["samples"].unbind())
    assert notifications == [(i, steps) for i in range(steps)]
    report = json.loads(text)
    assert report["counts"]["callbacks"] == {"low": low, "high": steps - low}
    assert report["counts"]["actual_forwards"] == {"low": low, "high": steps - low}
    assert report["counts"]["apply_model_calls"] == report["counts"]["actual_forwards"]
    assert report["network_input_shapes"]["low"] == [[1, 24, 2, 2, 4], [1, 32, 2, 8]]
    assert report["network_input_shapes"]["high"] == [[1, 24, 2, 4, 8], [1, 32, 2, 8]]
    assert report["noise"]["high_seed"] == 2
    assert report["pixel_anchor"] is report["highres_tiling"] is False
    assert report["status"] == "sampled_quality_unverified"
    assert report["lift_report"]["trained_lifter_executed"] is False
    assert len(stub_lifter) == 1
    assert model.model_options == options_before
    assert not model.wrappers
    assert set(positive[0][1]) == set(metadata_before)
    if task == "i2va":
        assert torch.equal(positive[0][1]["minimax_keyframes"][0]["latent"],
                           metadata_before["minimax_keyframes"][0]["latent"])


def test_cfg_forward_count_is_not_assumed_from_nfe(stub_lifter):
    _, report = runtime.sample_progressive_h3(tiny_model(), conditioning(), conditioning(), latent(),
        comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.),
        upscaler_model="test", seed=1, cfg=5., low_evaluations=1)
    report = json.loads(report)
    assert sum(report["counts"]["callbacks"].values()) == 2
    assert sum(report["cfg_branch_evaluations"].values()) == 4
    assert sum(report["counts"]["actual_forwards"].values()) >= 2


def test_callback_cancellation_leaves_original_model_reusable(stub_lifter):
    model = tiny_model()
    options = copy.deepcopy(model.model_options)

    def cancel(*args):
        raise RuntimeError("test cancellation")

    kwargs = dict(upscaler_model="test", seed=1, low_evaluations=1)
    with pytest.raises(RuntimeError, match="test cancellation"):
        runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), callback=cancel, **kwargs)
    assert model.model_options == options
    assert not stub_lifter
    output, _ = runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
        comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), **kwargs)
    assert output["samples"].is_nested


def test_lift_failure_does_not_start_high_stage_or_touch_original(monkeypatch, stub_lifter):
    model = tiny_model()
    observed = []

    def broken(*args):
        raise RuntimeError("test learned failure")

    monkeypatch.setattr(runtime, "_lift_video", broken)
    with pytest.raises(RuntimeError, match="test learned failure"):
        runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler("euler"), native_flow_sigmas(4, 12.),
            upscaler_model="test", seed=1, low_evaluations=2,
            callback=lambda step, *args: observed.append(step))
    assert observed == [0, 1]
    assert "model_function_wrapper" not in model.model_options


@pytest.mark.parametrize("option", ["sampler_post_cfg_function", "model_function_wrapper"])
def test_model_wrapper_rejected_before_loading(option):
    model = tiny_model()
    model.model_options[option] = lambda *args: None
    with pytest.raises(ValueError, match="already owns"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


@pytest.mark.parametrize("name", ["t8_minimax_h3_openvdn_contract_v2", "t8_fast_h3_vsa_gate_contract_v1"])
def test_existing_vdn_and_fast_workflows_not_repurposed(name):
    model = tiny_model()
    model.set_attachments(name, {"status": "configured"})
    with pytest.raises(ValueError, match="independent"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


def test_custom_euler_or_noise_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="Euler"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("heun"))
    model.model.model_sampling.noise_scaling = lambda *args: None
    with pytest.raises(ValueError, match="noise scaling"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


def test_missing_lifter_never_becomes_nearest():
    with pytest.raises(ValueError, match="no nearest"):
        runtime._lifter_identity("none", "fp16")


def get_plan(task="i2va"):
    return plan_progressive_first_sample(*latent()["samples"].unbind(), native_flow_sigmas(4, 12.),
                                         low_evaluations=2, task=task)


def test_first_reference_resizes_latent_not_original_metadata():
    source = conditioning("i2va")
    reference = source[0][1]["minimax_keyframes"][0]["latent"]
    low, high = runtime.prepare_stage_conditioning(source, get_plan(), positive=True)
    assert low[0][1]["minimax_keyframes"][0]["latent"].shape == (1, 24, 1, 2, 4)
    assert high[0][1]["minimax_keyframes"][0]["latent"] is reference
    assert high[0][1] is not source[0][1]
    assert high[0][1]["minimax_keyframes"][0] is not source[0][1]["minimax_keyframes"][0]


@pytest.mark.parametrize("key", ["minimax_refs", "area", "control", "hooks", "mask"])
def test_unsupported_conditioning_is_explicit(key):
    source = conditioning()
    source[0][1][key] = object()
    with pytest.raises(ValueError, match="not yet qualified"):
        runtime.prepare_stage_conditioning(source, get_plan("t2va"), positive=True)


@pytest.mark.parametrize("batched", [False, True])
def test_native_qwen_modality_tags_are_preserved_in_both_stages(batched):
    from comfy.text_encoders.minimax import token_tags_from_embeds_info
    source = conditioning("i2va")
    # Core's tag producer is exercised without loading the large text model.
    tags = token_tags_from_embeds_info(2, [{"type": "image", "index": 0, "size": 0}])
    if batched:
        tags = tags.unsqueeze(0)
    source[0][1]["minimax_token_tags"] = tags
    low, high = runtime.prepare_stage_conditioning(source, get_plan(), positive=True)
    assert low[0][1]["minimax_token_tags"] is tags
    assert high[0][1]["minimax_token_tags"] is tags
    assert tags.dtype == torch.long and tags.reshape(-1).tolist() == [0, 1]


@pytest.mark.parametrize("tags", [torch.ones(3, dtype=torch.long), torch.ones(2), torch.ones(2, dtype=torch.bool),
                                  torch.tensor([1, 2]), torch.tensor([-1, 1]), torch.ones(2, 2, dtype=torch.long)])
def test_invalid_modality_tags_are_rejected_before_sampling(tags):
    source = conditioning()
    source[0][1]["minimax_token_tags"] = tags
    with pytest.raises(ValueError, match="minimax_token_tags"):
        runtime.prepare_stage_conditioning(source, get_plan("t2va"), positive=True)


def test_i2va_requires_first_reference():
    with pytest.raises(ValueError, match="requires a first-frame"):
        runtime.prepare_stage_conditioning(conditioning(), get_plan(), positive=True)
    source = conditioning("i2va")
    source[0][1]["minimax_keyframes"][0]["resolved_frame_index"] = 4
    with pytest.raises(ValueError, match="Only first-frame"):
        runtime.prepare_stage_conditioning(source, get_plan(), positive=True)


@pytest.mark.parametrize("noise_scale", [.7, 1., 1.3])
def test_existing_native_setup_and_actual_restart_states(stub_lifter, noise_scale):
    source = latent()
    original = tiny_model()
    original.model.model_sampling.noise_scale = noise_scale
    model, sampler, schedule = setup_dual_clock_sampling(original, source, 8, 8., 2., "euler")
    low_state = {}
    first_high = {}

    def observe(step, prediction, state, total):
        if step == 5:
            low_state["x"] = [v.clone() for v in state.unbind()]
            low_state["x0"] = [v.clone() for v in prediction.unbind()]
        elif step == 6:
            first_high["x"] = [v.clone() for v in state.unbind()]

    _, text = runtime.sample_progressive_h3(model, conditioning(), conditioning(), source, sampler, schedule,
        upscaler_model="test", seed=1, low_evaluations=6, callback=observe)
    # Independent oracle: continue native sampler-space audio through the interval.
    predicted_audio = low_state["x0"][1]
    audio = low_state["x"][1]
    expected_audio = audio + ((audio - predicted_audio) / schedule[5]) * (schedule[6] - schedule[5])
    torch.testing.assert_close(first_high["x"][1], expected_audio, rtol=2e-5, atol=2e-6)
    # The labelled interpolation lifter replaces only the trained resizer here.
    lifted = F.interpolate(low_state["x0"][0], size=(2, 4, 8), mode="trilinear", align_corners=False)
    noise = torch.randn(lifted.shape, generator=torch.Generator(device="cpu").manual_seed(2))
    expected_video = lifted * (1 - schedule[6]) + noise * schedule[6] * noise_scale
    torch.testing.assert_close(first_high["x"][0], expected_video, rtol=2e-5, atol=2e-6)
    assert json.loads(text)["counts"]["actual_forwards"] == {"low": 6, "high": 2}
    assert model.model_options["transformer_options"]["minimax_h3_sigma_shift_video"] == 8.
    assert not original.object_patches
    assert not original.wrappers and not model.wrappers


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_invalid_noise_scale_is_rejected(value):
    model = tiny_model()
    model.model.model_sampling.noise_scale = value
    with pytest.raises(ValueError, match="noise scale"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


def test_network_object_patch_and_unknown_tensor_option_are_rejected():
    model = tiny_model()
    model.add_object_patch("diffusion_model.forward", lambda *args: None)
    with pytest.raises(ValueError, match="object patches"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))
    model = tiny_model()
    model.model_options["transformer_options"]["unqualified"] = torch.ones(2)
    with pytest.raises(ValueError, match="Unqualified transformer option"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


def test_inconsistent_shift_is_rejected():
    model = tiny_model()
    model.model_options["transformer_options"]["minimax_h3_sigma_shift_audio"] = 2.
    with pytest.raises(ValueError, match="shift mismatch"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


@pytest.mark.parametrize("source", [None, [], {}])
def test_invalid_latent_reports_a_useful_error(source):
    with pytest.raises(ValueError, match="LATENT dictionary"):
        runtime.sample_progressive_h3(tiny_model(), conditioning(), conditioning(), source,
            comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), upscaler_model="test", seed=1)


@pytest.mark.parametrize("task", ["t2va", "i2va"])
def test_registered_node_executes_real_core_with_existing_conditioner(stub_lifter, task):
    from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
    from h3_audio_t8_pkg.conditioning import build_conditioning
    from h3_audio_t8_pkg.nodes_progressive_sampling import MiniMaxH3ProgressiveSamplerEXPT8

    class SmallTextEncoder(FakeClip):
        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 2, 8), {}]]

    positive, source, *_ = build_conditioning(
        SmallTextEncoder(), FakeVideoVAE(), FakeAudioVAE(), "A woman in a quiet room.", 128, 64, 5,
        task_type=task, audio_mode="native", add_source_as_reference=False,
        first_frame=torch.zeros(1, 64, 128, 3) if task == "i2va" else None)
    model, sampler, sigmas = setup_dual_clock_sampling(tiny_model(), source, 8, 12., 3., "euler")
    result = MiniMaxH3ProgressiveSamplerEXPT8.execute(
        model=model, positive=positive, negative=conditioning(), av_latent=source, sampler=sampler, sigmas=sigmas,
        upscaler_model="test", seed=1, low_evaluations=6, task=task)
    output, text = result.result
    assert output["samples"].is_nested
    assert json.loads(text)["counts"]["actual_forwards"] == {"low": 6, "high": 2}


def test_invalid_lift_geometry_rejected_before_sampling(stub_lifter):
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 6), torch.zeros(1, 32, 2, 8)))}
    with pytest.raises(ValueError, match="anisotropic"):
        runtime.sample_progressive_h3(tiny_model(), conditioning(), conditioning(), source,
            comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), upscaler_model="test", seed=1, low_evaluations=1)
    assert not stub_lifter


def test_non_native_timestep_multiplier_is_rejected():
    model = tiny_model()
    model.model.model_sampling.multiplier = 500
    with pytest.raises(ValueError, match="multiplier"):
        runtime.validate_native_model(model, comfy.samplers.ksampler("euler"))


def test_high_stage_cancellation_does_not_leave_wrappers(stub_lifter):
    model = tiny_model()
    options = copy.deepcopy(model.model_options)

    def cancel_high(step, *args):
        if step == 1:
            raise RuntimeError("cancel during high stage")

    with pytest.raises(RuntimeError, match="cancel during high"):
        runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), upscaler_model="test", seed=1,
            low_evaluations=1, callback=cancel_high)
    assert len(stub_lifter) == 1
    assert model.model_options == options
    assert not model.wrappers


@pytest.mark.parametrize("malformed", [False, True])
def test_wrong_checkpoint_rejected_before_low_stage_or_full_hash(tmp_path, monkeypatch, malformed):
    import folder_paths
    from safetensors.torch import save_file
    path = tmp_path / "not_h3.safetensors"
    if malformed:
        path.write_bytes(b"not a safetensors file")
    else:
        save_file({"wrong_model.weight": torch.ones(1)}, str(path))
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda *args: str(path))

    def unexpected(*args):
        raise AssertionError("wrong architecture must not reach sampling or weight hashing")

    monkeypatch.setattr(runtime, "_sha256_file", unexpected)
    monkeypatch.setattr(runtime, "_native_stage", unexpected)
    with pytest.raises(ValueError, match="Not a compatible H3 learned upscaler"):
        runtime.sample_progressive_h3(tiny_model(), conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler("euler"), native_flow_sigmas(2, 12.), upscaler_model=path.name, seed=1,
            low_evaluations=1)
