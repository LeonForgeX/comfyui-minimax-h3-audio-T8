import copy

import pytest
import torch
from comfy.ldm.modules import attention
from comfy.model_management import InterruptProcessingException

from h3_audio_t8_pkg import h3_attention_ownership as ownership
from h3_audio_t8_pkg import fast_h3_vsa_advanced as fast
from h3_audio_t8_pkg import sla_precision_v2_advanced as sla
from h3_audio_t8_pkg.sampling import native_flow_sigmas
from test_vdn_attention_compat import model_fixture, sparse


def _guard(model, owner):
    return model.wrappers["diffusion_model"][f"t8_attention_owner_{owner}"][0]


@pytest.mark.parametrize("backend", ["attention_pytorch", "attention_sage"])
def test_plain_backend_remains_a_valid_input(backend):
    model = model_fixture()
    if not hasattr(model, "set_model_optimized_attention"):
        pytest.skip("Core has no per-model backend selector")
    model.set_model_optimized_attention(getattr(attention, backend))
    prepared, removed = ownership.prepare_attention_owner(model, "test")
    assert prepared is model and removed == 0


@pytest.mark.parametrize("owner", ["FastH3 VSA", "SLA Precision V2"])
@pytest.mark.parametrize("before", [False, True])
def test_real_official_sparse_before_after_and_repeated_callback(owner, before, monkeypatch):
    pytest.importorskip("comfy_extras.nodes_sparse_attention")
    model = model_fixture()
    incoming = sparse(model) if before else model
    original_options = copy.deepcopy(incoming.model_options)
    if owner == "FastH3 VSA":
        monkeypatch.setattr(fast, "probe_comfy_kitchen_vsa", lambda: {"external_vsa_executor_available": True})
        monkeypatch.setattr(fast, "_gate_modules", lambda _: ([object(), object()], None))
        patched, receipt, error = fast.apply_fast_h3_vsa(incoming)
        assert error is None
        assert receipt["runtime_attention_ownership_checked"]
    else:
        incoming.model_options["transformer_options"].update(
            minimax_h3_sigma_shift_video=12., minimax_h3_sigma_shift_audio=3.)
        original_options = copy.deepcopy(incoming.model_options)
        patched, runtime, _ = sla.patch_sla_precision_v2(incoming, native_flow_sigmas(8, 12.), dense_backend="pytorch")
        assert runtime.config["runtime_attention_ownership_checked"]
    assert incoming.model_options == original_options
    expected = patched.model_options["transformer_options"].copy()
    downstream = sparse(patched)
    guard = _guard(downstream, owner)
    for _ in range(2):
        downstream.prepare_state(torch.tensor(0.5), downstream.model_options)
        options = downstream.model_options["transformer_options"]
        assert guard(lambda *a, **kw: "executed", None, None, None, options) == "executed"
        assert options.get("optimized_attention_override") is expected.get("optimized_attention_override")
        assert options.get("patches_replace", {}).get("dit", {}) == expected.get("patches_replace", {}).get("dit", {})


def test_unknown_changes_are_rejected_without_modifying_live_options():
    def expected():
        pass
    def unknown():
        pass
    options = {"optimized_attention_override": unknown}
    before = options.copy()
    with pytest.raises(RuntimeError, match="override was replaced"):
        ownership.validate_attention_owner(options, owner="SLA", expected_override=expected,
                                           expected_dit={}, owns_override=True)
    assert options == before
    options = {"patches_replace": {"dit": {("double_block", 0): unknown}}}
    before = copy.deepcopy(options)
    with pytest.raises(RuntimeError, match="blocks were replaced"):
        ownership.validate_attention_owner(options, owner="FastH3", expected_override=None,
                                           expected_dit={("double_block", 0): expected}, owns_override=False)
    assert options == before


def test_missing_options_never_runs_model_and_unknown_binding_keeps_contract(monkeypatch):
    model = model_fixture()
    ownership.bind_attention_owner_guard(model, "test", owns_override=False)
    with pytest.raises(RuntimeError, match="options are missing"):
        _guard(model, "test")(lambda: pytest.fail("model must not run"))
    model.model_options["transformer_options"]["optimized_attention_override"] = lambda: None
    monkeypatch.setattr(fast, "probe_comfy_kitchen_vsa", lambda: {"external_vsa_executor_available": True})
    monkeypatch.setattr(fast, "_gate_modules", lambda _: ([object()], None))
    returned, receipt, error = fast.apply_fast_h3_vsa(model)
    assert returned is model and receipt is None and "another algorithm" in error


@pytest.mark.parametrize("before", [False, True])
def test_eav_real_wrapper_normalizes_official_sparse_before_payload_check(before):
    from types import SimpleNamespace
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    pytest.importorskip("comfy_extras.nodes_sparse_attention")
    model = model_fixture()
    incoming = sparse(model) if before else model
    sigmas = torch.cat((torch.linspace(1., 0.05, 20), torch.zeros(1)))
    patched, _, _ = eav.build_eav_model(incoming, sigmas, mode="report_only", tau=4.,
                                       start_video_progress=0., end_video_progress=1.,
                                       max_workspace_mib=32, g_hard_limit=1.5)
    expected = patched.model_options["transformer_options"]["optimized_attention_override"]
    downstream = sparse(patched)
    wrapper = downstream.get_wrappers("diffusion_model", eav.EAV_WRAPPER_KEY)[0]
    for _ in range(2):
        downstream.prepare_state(torch.tensor(0.5), downstream.model_options)
        options = downstream.model_options["transformer_options"]
        with pytest.raises(RuntimeError, match="could not find.*minimax_payload"):
            wrapper(SimpleNamespace(wrappers=[wrapper]), None, None, None, options)
        assert options["optimized_attention_override"] is expected
        assert not options.get("patches_replace", {}).get("dit")
    options["optimized_attention_override"] = lambda: None
    with pytest.raises(RuntimeError, match="override was replaced"):
        wrapper(SimpleNamespace(wrappers=[wrapper]), None, None, None, options)


@pytest.mark.parametrize("before", [False, True])
def test_world_real_forward_checks_sparse_ownership_before_action_payload(before):
    from h3_audio_t8_pkg import h3_world_advanced as world
    pytest.importorskip("comfy_extras.nodes_sparse_attention")
    model = model_fixture()
    diffusion = model.model.diffusion_model
    # The model factory is real Core; tiny placeholder blocks suffice for
    # binding and pre-forward ownership checks, not a generation claim.
    diffusion.blocks = torch.nn.ModuleList([copy.deepcopy(diffusion.blocks[0]) for _ in range(50)])
    model.model.extra_conds = lambda **kw: {}
    incoming = sparse(model) if before else model
    patched, report = world.patch_h3_world_model(incoming, compile_flex_attention=False)
    assert report["runtime_attention_ownership_checked"]
    downstream = sparse(patched)
    forward = downstream.get_model_object("diffusion_model._forward")
    payload = {world.PAYLOAD_FLAG: world.PATCH_VERSION}
    for _ in range(2):
        downstream.prepare_state(torch.tensor(0.5), downstream.model_options)
        options = downstream.model_options["transformer_options"]
        with pytest.raises(RuntimeError, match="packed layout is missing"):
            forward(None, None, None, options, minimax_payload=payload)
    options["patches_replace"]["dit"][("double_block", 0)] = lambda *a: None
    with pytest.raises(RuntimeError, match="blocks were replaced"):
        forward(None, None, None, options, minimax_payload=payload)


def test_speed_preserves_actual_native_sparse_and_callbacks_during_inspection():
    from h3_audio_t8_pkg.speed_advanced import _ensure_native_h3_model
    pytest.importorskip("comfy_extras.nodes_sparse_attention")
    model = sparse(model_fixture())
    options = copy.deepcopy(model.model_options)
    callbacks = copy.deepcopy(model.callbacks)
    _ensure_native_h3_model(model)
    assert model.model_options == options and model.callbacks == callbacks
    for _ in range(2):
        model.prepare_state(torch.tensor(0.5), model.model_options)
        _ensure_native_h3_model(model)
        assert model.model_options["transformer_options"].get("patches_replace", {}).get("dit")


def test_speed_allows_plain_backend_but_not_unknown_override():
    from h3_audio_t8_pkg.speed_advanced import _ensure_native_h3_model
    model = model_fixture()
    if not hasattr(model, "set_model_optimized_attention"):
        pytest.skip("Core has no backend selector")
    model.set_model_optimized_attention(attention.attention_pytorch)
    _ensure_native_h3_model(model)
    model.model_options["transformer_options"]["optimized_attention_override"] = lambda *a: None
    with pytest.raises(ValueError, match="unknown attention"):
        _ensure_native_h3_model(model)


@pytest.mark.parametrize("before", [True, False])
@pytest.mark.parametrize("mode", ["disabled", "report_only", "apply_exp"])
def test_eav_stg_sparse_main_post_cfg_and_weak_preserve_skip_identity(before, mode, monkeypatch):
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    from h3_audio_t8_pkg import detail_sampling_advanced as detail
    source = model_fixture()
    incoming = sparse(source) if before else source
    original = copy.deepcopy(incoming.model_options)
    patched, runtime, _ = eav.build_eav_stg_model(
        incoming, torch.cat((torch.linspace(1., .05, 20), torch.zeros(1))),
        mode=mode, tau=4., start_video_progress=0., end_video_progress=1.,
        max_workspace_mib=32, g_hard_limit=1.5, stg_scale=.35,
        stg_double_blocks="1", stg_start_progress=.25, stg_end_progress=.85,
        shift_video=12., rescale=0.)
    assert incoming.model_options == original
    assert runtime.config["native_sparse_stg_main_and_weak_checked"]
    downstream = sparse(patched)
    wrapper = downstream.get_wrappers("diffusion_model", eav.EAV_STG_WRAPPER_KEY)[0]
    def executor(x, *_args, **_kwargs):
        return x
    executor.wrappers = [wrapper]
    def check_dispatch(options):
        if mode == "disabled":
            token = object()
            assert wrapper(executor, token, None, None, options) is token
            assert eav.EAV_RUNTIME_KEY not in options
        else:
            with pytest.raises(RuntimeError, match="could not find.*minimax_payload"):
                wrapper(executor, None, None, None, options)
    for _ in range(2):
        downstream.prepare_state(torch.tensor(.9), downstream.model_options)
        check_dispatch(downstream.model_options["transformer_options"])

    original_options = copy.deepcopy(downstream.model_options)
    def weak_call(model, cond, x, sigma, options):
        skip = options["transformer_options"]["patches_replace"]["dit"][("double_block", 1)]
        assert skip({"value": 17}, {}) == {"value": 17}
        downstream.prepare_state(sigma, options)
        check_dispatch(options["transformer_options"])
        assert options["transformer_options"]["patches_replace"]["dit"][("double_block", 1)] is skip
        return [torch.tensor(1.)]
    monkeypatch.setattr(detail.comfy.samplers, "calc_cond_batch", weak_call)
    callback = downstream.model_options["sampler_post_cfg_function"][0]
    args = dict(model=source.model, cond=[{}], input=torch.tensor(0.), sigma=torch.tensor(.9),
                denoised=torch.tensor(2.), cond_denoised=torch.tensor(3.), model_options=downstream.model_options)
    assert callback(args).item() == pytest.approx(2.7)
    assert downstream.model_options == original_options
    downstream.model_options["transformer_options"]["optimized_attention_override"] = lambda *a: None
    with pytest.raises(RuntimeError, match="override"):
        callback(args)


def test_eav_and_stg_both_off_preserve_exact_sparse_model_and_callbacks():
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    source = sparse(model_fixture())
    options, callbacks = copy.deepcopy(source.model_options), copy.deepcopy(source.callbacks)
    returned, runtime, _ = eav.build_eav_stg_model(
        source, torch.cat((torch.linspace(1., .05, 20), torch.zeros(1))),
        mode="disabled", tau=4., start_video_progress=0., end_video_progress=1.,
        max_workspace_mib=32, g_hard_limit=1.5, stg_scale=0.,
        stg_double_blocks="1", stg_start_progress=.25, stg_end_progress=.85,
        shift_video=12., rescale=0.)
    assert returned is source
    assert returned.model_options == options and returned.callbacks == callbacks
    assert not returned.wrappers
    assert not runtime.config.get("eav_disabled_guard_only_no_feta")


def _disabled_stg_model():
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    model, runtime, _ = eav.build_eav_stg_model(
        sparse(model_fixture()), torch.cat((torch.linspace(1., .05, 20), torch.zeros(1))),
        mode="disabled", tau=4., start_video_progress=0., end_video_progress=1.,
        max_workspace_mib=32, g_hard_limit=1.5, stg_scale=.35,
        stg_double_blocks="1", stg_start_progress=.25, stg_end_progress=.85,
        shift_video=12., rescale=0.)
    return sparse(model), runtime


@pytest.mark.parametrize("corruption", ["marker", "missing_marker", "skip_identity", "extra_block"])
def test_disabled_eav_stg_rejects_corrupted_weak_branch_without_leaking(corruption, monkeypatch):
    from comfy.patcher_extension import WrapperExecutor
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    from h3_audio_t8_pkg import detail_sampling_advanced as detail
    model, _ = _disabled_stg_model()
    original = copy.deepcopy(model.model_options)
    wrapper = model.get_wrappers("diffusion_model", eav.EAV_STG_WRAPPER_KEY)[0]
    def weak_call(_model, _cond, x, sigma, options):
        model.prepare_state(sigma, options)
        options = options["transformer_options"]
        if corruption == "marker":
            options[eav.EAV_STG_BRANCH_KEY]["binding_hash"] = "forged"
        elif corruption == "missing_marker":
            options.pop(eav.EAV_STG_BRANCH_KEY)
        else:
            hooks = options["patches_replace"]["dit"]
            real = hooks[("double_block", 1)]
            def forged(args, _extra):
                return args
            # Matching the public attributes must not authenticate a new hook.
            forged.__dict__.update(real.__dict__)
            hooks[("double_block", 1 if corruption == "skip_identity" else 7)] = forged
        def forbidden(*_args, **_kwargs):
            pytest.fail("corrupted weak branch reached diffusion")
        executor = WrapperExecutor.new_executor(forbidden, [wrapper])
        return [executor.execute(x, sigma, None, options)]
    monkeypatch.setattr(detail.comfy.samplers, "calc_cond_batch", weak_call)
    args = dict(model=model.model, cond=[{}], input=torch.tensor(0.), sigma=torch.tensor(.9),
                denoised=torch.tensor(2.), cond_denoised=torch.tensor(3.), model_options=model.model_options)
    with pytest.raises(RuntimeError, match="marker was invalid|blocks were replaced"):
        model.model_options["sampler_post_cfg_function"][0](args)
    assert model.model_options == original


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, InterruptProcessingException])
def test_disabled_eav_stg_weak_error_or_cancel_then_retry_has_no_stale_state(failure, monkeypatch):
    from comfy.patcher_extension import WrapperExecutor
    from h3_audio_t8_pkg import enhance_a_video_advanced as eav
    from h3_audio_t8_pkg import detail_sampling_advanced as detail
    model, runtime = _disabled_stg_model()
    original = copy.deepcopy(model.model_options)
    before = runtime.snapshot(consume=False)
    wrapper = model.get_wrappers("diffusion_model", eav.EAV_STG_WRAPPER_KEY)[0]
    calls = []
    def weak_call(_model, _cond, x, sigma, options):
        model.prepare_state(sigma, options)
        def diffusion(value, timestep, context, transformer_options, **kwargs):
            assert value is x and timestep is sigma and context is None
            assert eav.EAV_RUNTIME_KEY not in transformer_options
            assert set(transformer_options["patches_replace"]["dit"]) == {("double_block", 1)}
            assert kwargs["sentinel"] is x
            calls.append(True)
            if len(calls) == 1:
                raise failure("controlled weak forward interruption")
            return torch.tensor(1.)
        executor = WrapperExecutor.new_executor(diffusion, [wrapper])
        return [executor.execute(x, sigma, None, options["transformer_options"], sentinel=x)]
    monkeypatch.setattr(detail.comfy.samplers, "calc_cond_batch", weak_call)
    args = dict(model=model.model, cond=[{}], input=torch.tensor(0.), sigma=torch.tensor(.9),
                denoised=torch.tensor(2.), cond_denoised=torch.tensor(3.), model_options=model.model_options)
    callback = model.model_options["sampler_post_cfg_function"][0]
    with pytest.raises(failure, match="controlled weak"):
        callback(args)
    assert model.model_options == original
    assert runtime.snapshot(consume=False) == before
    assert callback(args).item() == pytest.approx(2.7)
    assert calls == [True, True] and model.model_options == original
    assert runtime.snapshot(consume=False) == before
