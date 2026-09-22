import json
import uuid

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from h3_audio_t8_pkg import long_video_dual_model_stages as stages
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.h3_memory_advanced import (
    configure_chunk_feed_forward,
    configure_low_vram_attention,
)
from test_prompt_relay_core_compat import model_fixture, bound_layout
from test_h3_memory_advanced import _small_model


def latent(value):
    return {"samples": NestedTensor([torch.full((1, 24, 3, 2, 2), value),
                                      torch.full((1, 32, 2, 11), value)])}


def test_stage_binding_keeps_independent_lora_stack_and_base_untouched():
    first = model_fixture()
    second = first.clone()
    first.patches = {"x": [(1., (torch.ones(2),), 1., None, None)]}
    second.patches = {"x": [(.4, (torch.zeros(2),), 1., None, None)]}
    second.patches_uuid = uuid.uuid4()
    binding, _ = bound_layout("joint_av_exp")

    def builder(model):
        assert not model.patches
        return relay.patch_prompt_relay_model(model, binding, 32)

    a = stages.bind_stage_conditioning(first, builder)[0]
    b = stages.bind_stage_conditioning(second, builder)[0]
    assert a.model is b.model is first.model
    assert a.patches_uuid == first.patches_uuid != b.patches_uuid == second.patches_uuid
    assert a.patches == first.patches and b.patches == second.patches
    assert a.patches is not first.patches and a.patches["x"] is not first.patches["x"]
    a.patches["x"].append((.1, (), 1., None, None))
    assert len(first.patches["x"]) == len(b.patches["x"]) == 1
    assert not first.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)


def test_builder_cannot_swap_stage_base():
    with pytest.raises(RuntimeError, match="changed the base"):
        stages.bind_stage_conditioning(model_fixture(), lambda **kwargs: (model_fixture(),))


def test_each_model_supplies_its_own_clock(monkeypatch):
    first, second = object(), object()
    seen = []
    def plan(model, base, coarse, refine):
        seen.append(model)
        tag = 1. if model is first else 2.
        return torch.tensor([tag, .5]), torch.tensor([tag, 0.]), json.dumps({"tag": tag})
    monkeypatch.setattr(stages, "build_learned_two_pass_parity_plan", plan)
    coarse, refine, report = stages.dual_model_schedules(first, second)
    assert seen == [first, second]
    assert float(coarse[0]) == 1. and float(refine[0]) == 2.
    assert report["total_nfe"] == 8


@pytest.mark.parametrize("output_kind", ["denoised_x0", "zero_sigma_output"])
def test_sample_exports_correct_value_not_noisy_tail(monkeypatch, output_kind):
    model = model_fixture()
    calls = []
    model.model.process_latent_out = lambda value: calls.append("normalize") or value * 3.
    source = latent(0.)
    def sample(*args, **kwargs):
        wrapper = args[0].get_wrappers("apply_model", "t8_dual_stage_network_observer")[0]
        for _ in range(len(kwargs['sigmas']) - 1):
            wrapper(lambda: None)
        kwargs["preview_state"]["x0"] = latent(2.)["samples"]
        return latent(9.)
    monkeypatch.setattr(stages, "_sample_prepared_segment", sample)
    output, report = stages.sample_model_stage(model, [], source, sampler=None,
        sigmas=torch.tensor([1., .5, 0.]), seed=37, output_kind=output_kind)
    expected = 6. if output_kind == "denoised_x0" else 9.
    assert all(torch.all(value == expected) for value in output["samples"].unbind())
    assert all(torch.all(value == 0.) for value in source["samples"].unbind())
    assert calls == (["normalize"] if output_kind == "denoised_x0" else [])
    assert report["nfe"] == 2
    assert report["completed_network_forwards"] == 2
    assert not model.get_wrappers("apply_model", "t8_dual_stage_network_observer")


@pytest.mark.parametrize("foreign_wrapper", [False, True])
def test_stage_observer_preserves_real_relay_owner_and_warns_for_foreign(monkeypatch, caplog, foreign_wrapper):
    from comfy.patcher_extension import WrapperExecutor
    binding, layout = bound_layout("joint_av_exp")
    model, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    seen = []

    def sample(observed, *args, **kwargs):
        seen.append(observed)
        if foreign_wrapper:
            observed.add_wrapper_with_key("diffusion_model", "foreign", lambda executor, *a, **k: executor(*a, **k))
        observed.prepare_state(torch.tensor(.5), observed.model_options)
        options = observed.model_options['transformer_options']
        def diffusion(x, timestep, context, transformer_options, **kw):
            assert transformer_options[relay.PROMPT_RELAY_RUNTIME_KEY]['binding_hash'] == binding['binding_hash']
            return latent(9.)
        def apply():
            inner = WrapperExecutor.new_executor(diffusion, [fn for group in observed.wrappers['diffusion_model'].values() for fn in group])
            return inner.execute([torch.zeros(1)], None, None, options,
                minimax_payload={'layout': layout}, **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding['binding_hash']})
        outer = WrapperExecutor.new_executor(apply, observed.get_wrappers('apply_model', 't8_dual_stage_network_observer'))
        result = outer.execute()
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
        return result

    monkeypatch.setattr(stages, '_sample_prepared_segment', sample)
    with caplog.at_level('WARNING'):
        _, report = stages.sample_model_stage(model, [], latent(0.), sampler=None, sigmas=torch.tensor([1., 0.]),
                                             seed=0, output_kind='zero_sigma_output')
    assert report['completed_network_forwards'] == 1
    if foreign_wrapper:
        assert 'another diffusion-model wrapper' in caplog.text
    assert not seen[0].get_wrappers('apply_model', 't8_dual_stage_network_observer')
    assert len(model.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)) == 1


def test_stage_reports_authenticated_t8_memory_and_executes_full_relay_chain(monkeypatch):
    from comfy.patcher_extension import WrapperExecutor

    binding, layout = bound_layout("joint_av_exp")
    model, _ = configure_low_vram_attention(_small_model(1), 4)
    model, _ = configure_chunk_feed_forward(model, 2, 4096)
    model, _ = relay.patch_prompt_relay_model(model, binding, 32)

    def sample(observed, *args, **kwargs):
        observed.patch_model(load_weights=False)
        try:
            options = observed.model_options["transformer_options"]

            def diffusion(*_args, **_kwargs):
                return latent(9.0)

            def apply():
                inner = WrapperExecutor.new_executor(
                    diffusion,
                    observed.get_all_wrappers("diffusion_model"),
                )
                return inner.execute(
                    [torch.zeros(1)],
                    None,
                    None,
                    options,
                    minimax_payload={"layout": layout},
                    **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding["binding_hash"]},
                )

            outer = WrapperExecutor.new_executor(
                apply,
                observed.get_wrappers(
                    "apply_model", "t8_dual_stage_network_observer"
                ),
            )
            return outer.execute()
        finally:
            observed.unpatch_model(unpatch_weights=False)

    monkeypatch.setattr(stages, "_sample_prepared_segment", sample)
    _, report = stages.sample_model_stage(
        model,
        [],
        latent(0.0),
        sampler=None,
        sigmas=torch.tensor([1.0, 0.0]),
        seed=0,
        output_kind="zero_sigma_output",
    )
    assert report["completed_network_forwards"] == 1
    assert report["prompt_relay_execution"] == {
        "completed_forwards": 1,
        "routed_attention_calls": 0,
    }
    assert report["memory_composition"]["head_chunks"] == 4
    assert report["memory_composition"]["ffn_settings"] == [2, 4096]
    assert report["backend"]["memory_composition"] == report["memory_composition"]


def test_missing_x0_refuses_upscale_instead_of_using_noisy_output(monkeypatch):
    monkeypatch.setattr(stages, "_sample_prepared_segment", lambda *a, **kw: latent(9.))
    with pytest.raises(RuntimeError, match="did not return denoised x0"):
        stages.sample_model_stage(model_fixture(), [], latent(0.), sampler=None,
                                 sigmas=torch.tensor([1., .5]), seed=0)


@pytest.mark.parametrize("sigmas", [[1., .5], [1., float("nan"), 0.], [1., 1., 0.], [1., 0., -.1]])
def test_high_stage_requires_finite_descent_to_zero(sigmas):
    with pytest.raises(ValueError):
        stages.sample_model_stage(model_fixture(), [], latent(0.), sampler=None,
            sigmas=torch.tensor(sigmas), seed=0, output_kind="zero_sigma_output")
def test_native_pytorch_counter_preserves_math_and_never_counts_failed_calls():
    from h3_audio_t8_pkg.long_video_dual_model_stages import _NativePytorchObserver
    from comfy.ldm.modules import attention
    observer = _NativePytorchObserver()
    q, k, v = [torch.randn(1, 2, 5, 8) for _ in range(3)]
    kwargs = dict(skip_reshape=True, skip_output_reshape=True)
    result = observer.attention(q, k, v, 2, **kwargs)
    expected = attention.attention_pytorch(q, k, v, 2, **kwargs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert observer.report()['completed_calls'] == {'pytorch:completed': 1}
    def broken(*args, **kwargs):
        raise RuntimeError('actual kernel failure')
    observer.function = broken
    with pytest.raises(RuntimeError, match='actual kernel failure'):
        observer.attention(q, k, v, 2, **kwargs)
    assert observer.report()['completed_calls'] == {'pytorch:completed': 1}


@pytest.mark.parametrize('fail', [False, True])
def test_stage_times_native_prepare_and_forward_without_leaking_wrappers(monkeypatch, fail):
    from comfy.patcher_extension import WrappersMP
    seen = []
    def sample(observed, *args, **kwargs):
        seen.append(observed)
        prepare = observed.get_wrappers(WrappersMP.PREPARE_SAMPLING, 't8_dual_stage_prepare_timer')
        assert len(prepare) == 1
        assert prepare[0](lambda value: value, 'sentinel') == 'sentinel'
        forward = observed.get_wrappers(WrappersMP.APPLY_MODEL, 't8_dual_stage_network_observer')[0]
        if fail:
            forward(lambda: (_ for _ in ()).throw(RuntimeError('test forward failure')))
        forward(lambda: None)
        return latent(9.)
    monkeypatch.setattr(stages, '_sample_prepared_segment', sample)
    model = model_fixture()
    if fail:
        with pytest.raises(RuntimeError, match='test forward failure'):
            stages.sample_model_stage(model, [], latent(0.), sampler=None,
                sigmas=torch.tensor([1., 0.]), seed=0, output_kind='zero_sigma_output')
    else:
        _, report = stages.sample_model_stage(model, [], latent(0.), sampler=None,
            sigmas=torch.tensor([1., 0.]), seed=0, output_kind='zero_sigma_output')
        events = report['execution_timings']['events']
        assert [e['phase'] for e in events] == ['prepare_sampling_including_model_load', 'forward_including_dynamic_transfers']
        assert all(e['status'] == 'completed' and e['seconds'] >= 0 for e in events)
    for patcher in [model, *seen]:
        assert not patcher.get_wrappers(WrappersMP.PREPARE_SAMPLING, 't8_dual_stage_prepare_timer')
        assert not patcher.get_wrappers(WrappersMP.APPLY_MODEL, 't8_dual_stage_network_observer')
