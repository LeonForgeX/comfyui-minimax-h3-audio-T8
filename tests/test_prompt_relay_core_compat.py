"""Real Core Relay ownership, composition, routed math and cancellation regressions."""
from types import SimpleNamespace

import pytest
import torch

from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.ldm.modules import attention
from comfy.model_base import MiniMaxH3
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrapperExecutor
from h3_audio_t8_pkg import prompt_relay_advanced as relay


def model_fixture():
    base = MiniMaxH3.__new__(MiniMaxH3)
    torch.nn.Module.__init__(base)
    diffusion = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(diffusion)
    diffusion.blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for block in diffusion.blocks:
        block.attn = torch.nn.Identity()
    base.diffusion_model = diffusion
    base.model_sampling = SimpleNamespace(percent_to_sigma=lambda p: 1 - p)
    return ModelPatcher(base, torch.device('cpu'), torch.device('cpu'))


def sparse(model):
    apply_block_sparse_attention = pytest.importorskip("comfy_extras.nodes_sparse_attention").apply_block_sparse_attention
    return apply_block_sparse_attention(model, tau=1.3, topk_ratio=0, vsa=False,
                                        start_percent=0, end_percent=1, min_tokens=4096,
                                        dense_blocks=set(), sink_conditioning='off',
                                        extra_tokens=0, verbose=False)


def test_native_control_reaches_core_contract_without_hash_monkeypatch():
    assert relay._assert_core_contract(model_fixture())


def test_real_core_unpatch_restores_native_method_and_can_rebind_relay():
    from h3_audio_t8_pkg.long_video import patch_long_video_model
    source = model_fixture()
    first = patch_long_video_model(source)
    first.patch_model(load_weights=False)
    assert source.model.extra_conds.__func__ is not type(source.model).extra_conds
    first.unpatch_model(unpatch_weights=False)
    assert source.model.__dict__['extra_conds'].__func__ is type(source.model).extra_conds
    assert not first.object_patches_backup
    assert relay._assert_core_contract(source)
    binding, _ = bound_layout('joint_av_exp')
    rebound, _ = relay.patch_prompt_relay_model(source, binding, 32)
    assert relay.prompt_relay_model_contract(rebound)['binding_hash'] == binding['binding_hash']


def test_native_function_bound_to_a_different_instance_is_not_native_restore():
    source, other = model_fixture(), model_fixture()
    source.model.extra_conds = other.model.extra_conds
    with pytest.raises(RuntimeError, match='instance-level extra_conds'):
        relay._assert_core_contract(source)


@pytest.mark.parametrize('backend', ['pytorch', 'sage'])
def test_plain_core_backend_before_relay_should_be_accepted(backend):
    model = model_fixture()
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    set_h3_attention_backend(model, getattr(attention, 'attention_' + backend))
    assert relay._assert_core_contract(model)


def test_official_sparse_before_relay_requires_scoped_owner_adaptation():
    model = sparse(model_fixture())
    assert relay._assert_core_contract(model)


@pytest.mark.parametrize('downstream_sparse', [False, True])
def test_runtime_reaches_pairing_guard_after_known_attention_normalization(downstream_sparse):
    model = model_fixture()
    contract = relay._assert_core_contract(model)
    patched, _ = relay._install_prompt_relay_model(model, {'binding_hash': 'cpu_probe'}, 64, contract)
    if downstream_sparse:
        patched = sparse(patched)
    patched.prepare_state(torch.tensor(.5), patched.model_options)
    wrappers = patched.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
    assert len(wrappers) == 1
    def forbidden(*args, **kwargs):
        pytest.fail('probe must not execute diffusion without paired conditioning')
    executor = WrapperExecutor.new_executor(forbidden, wrappers)
    with pytest.raises(RuntimeError, match='not the paired outputs'):
        executor.execute(None, None, None, patched.model_options['transformer_options'])


def test_unknown_override_is_still_rejected():
    model = model_fixture()
    model.model_options['transformer_options']['optimized_attention_override'] = lambda *a, **k: None
    with pytest.raises(RuntimeError, match='optimized_attention_override'):
        relay._assert_core_contract(model)


def test_copied_binding_marker_must_not_authenticate_foreign_runtime_override():
    model = model_fixture()
    contract = relay._assert_core_contract(model)
    patched, _ = relay._install_prompt_relay_model(model, {'binding_hash': 'cpu_probe'}, 64, contract)
    options = patched.model_options['transformer_options']

    def foreign(*args, **kwargs):
        pytest.fail('foreign attention must never run')

    # Public descriptive tags are not proof that this is our installed closure.
    foreign._t8_prompt_relay_binding_hash = 'cpu_probe'
    options['optimized_attention_override'] = foreign
    wrappers = patched.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
    executor = WrapperExecutor.new_executor(lambda *a, **k: pytest.fail('must reject before diffusion'), wrappers)
    with pytest.raises(RuntimeError, match='override was replaced'):
        executor.execute(None, None, None, options)


@pytest.mark.parametrize('phase', ['before_bind', 'runtime'])
def test_unknown_attention_hook_is_rejected_before_pairing_or_diffusion(phase):
    model = model_fixture()

    def foreign(*args, **kwargs):
        pytest.fail('unrecognized hook must never run')

    if phase == 'before_bind':
        model.model_options['transformer_options']['patches'] = {'attn1_patch': [foreign]}
        with pytest.raises(RuntimeError, match='attention hook'):
            relay._assert_core_contract(model)
        return
    contract = relay._assert_core_contract(model)
    patched, _ = relay._install_prompt_relay_model(model, {'binding_hash': 'cpu_probe'}, 64, contract)
    options = patched.model_options['transformer_options']
    options['patches'] = {'attn1_patch': [foreign]}
    wrappers = patched.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
    executor = WrapperExecutor.new_executor(lambda *a, **k: pytest.fail('must reject before diffusion'), wrappers)
    with pytest.raises(RuntimeError, match='attention hook'):
        executor.execute(None, None, None, options)


def bound_layout(query_route):
    layout = relay.build_packed_layout(4, 3, 2, 2, 2, frame_count=15)
    binding = {
        'schema': relay.PROMPT_RELAY_PATCH_VERSION,
        'plan_hash': 'small_cpu_routing_probe', 'text_len': 4,
        'query_route': query_route,
        'events': [
            {'text_key_start': 0, 'text_key_end': 2, 'midpoint': 1., 'window': .5, 'sigma': 1.},
            {'text_key_start': 2, 'text_key_end': 4, 'midpoint': 4., 'window': .5, 'sigma': 1.},
        ],
    }
    return relay._bind_layout_contract(binding, layout, resolved_task='t2va', keyframes=None, refs=None), layout


@pytest.mark.parametrize('order', ['native_control', 'sparse_before', 'sparse_after'])
@pytest.mark.parametrize('query_route', ['video_only_paper', 'joint_av_exp'])
def test_real_routed_attention_matches_dense_math_across_repeated_prepare(monkeypatch, order, query_route):
    # Only the numerical delegate is pinned to PyTorch for this CPU test.
    # Core selectors, owner checks, actual PackedLayout and callbacks stay real.
    monkeypatch.setattr(attention, 'optimized_attention', attention.attention_pytorch)
    binding, layout = bound_layout(query_route)
    source = model_fixture()
    model = sparse(source) if order == 'sparse_before' else source
    patched, _ = relay.patch_prompt_relay_model(model, binding, 32)
    if order == 'sparse_after':
        patched = sparse(patched)
    count = int(layout.seq_len)
    generator = torch.Generator().manual_seed(710)
    q, k, v = [torch.randn(1, 2, count, 4, generator=generator) for _ in range(3)]
    route = relay._runtime_route(layout, binding, torch.device('cpu'))
    bias = torch.zeros(count, count)
    for segment in route['query_segments']:
        times = segment['query_times']
        # Independent equation, not make_prompt_relay_bias under test.
        for event in binding['events']:
            outside = ((times - event['midpoint']).abs() - event['window']).clamp_min(0)
            penalty = -.5 * (outside / event['sigma']).square()
            bias[segment['start']:segment['end'], event['text_key_start']:event['text_key_end']] += penalty[:, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = expected.transpose(1, 2).reshape(1, count, 8)
    observed_routes = []

    def execute_attention(x, timestep, context, options, **kwargs):
        observed_routes.append(options[relay.PROMPT_RELAY_RUNTIME_KEY]['query_route'])
        return attention.attention_pytorch(q, k, v, 2, skip_reshape=True, transformer_options=options)

    for _ in range(2):
        patched.prepare_state(torch.tensor(.5), patched.model_options)
        options = patched.model_options['transformer_options']
        wrappers = patched.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
        executor = WrapperExecutor.new_executor(execute_attention, wrappers)
        actual = executor.execute([torch.zeros(1)], None, None, options,
                                  minimax_payload={'layout': layout},
                                  **{relay.PROMPT_RELAY_PAYLOAD_KEY: binding['binding_hash']})
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options
    assert observed_routes == [query_route, query_route]
    assert source.model_options['transformer_options'] == {}


@pytest.mark.parametrize('downstream_sparse', [False, True])
def test_real_comfy_cancel_cleans_runtime_and_allows_retry(downstream_sparse):
    from comfy.model_management import InterruptProcessingException

    binding, layout = bound_layout('joint_av_exp')
    patched, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    if downstream_sparse:
        patched = sparse(patched)
    wrappers = patched.get_wrappers('diffusion_model', relay.PROMPT_RELAY_WRAPPER_KEY)
    for cancel in [True, False]:
        patched.prepare_state(torch.tensor(.5), patched.model_options)
        options = patched.model_options['transformer_options']

        def execute(x, timestep, context, runtime_options, **kwargs):
            assert runtime_options[relay.PROMPT_RELAY_RUNTIME_KEY]['binding_hash'] == binding['binding_hash']
            if cancel:
                raise InterruptProcessingException()
            return 'clean_retry'

        executor = WrapperExecutor.new_executor(execute, wrappers)
        args = ([torch.zeros(1)], None, None, options)
        kwargs = {'minimax_payload': {'layout': layout}, relay.PROMPT_RELAY_PAYLOAD_KEY: binding['binding_hash']}
        if cancel:
            with pytest.raises(InterruptProcessingException):
                executor.execute(*args, **kwargs)
        else:
            assert executor.execute(*args, **kwargs) == 'clean_retry'
        assert relay.PROMPT_RELAY_RUNTIME_KEY not in options


def compose_eav(model, mode):
    from h3_audio_t8_pkg.enhance_a_video_advanced import build_eav_prompt_relay_model

    sigmas = torch.cat([torch.linspace(1., .05, 20), torch.zeros(1)])
    return build_eav_prompt_relay_model(model, sigmas, mode=mode, tau=4.,
                                      start_video_progress=0., end_video_progress=1.,
                                      max_workspace_mib=32, g_hard_limit=1.5)


@pytest.mark.parametrize('mode', ['disabled', 'report_only', 'apply_exp'])
@pytest.mark.parametrize('intermediate_sparse', [False, True])
def test_eav_relay_composer_entry_keeps_known_sparse_from_becoming_a_false_conflict(mode, intermediate_sparse):
    binding, _ = bound_layout('video_only_paper')
    source, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    incoming = sparse(source) if intermediate_sparse else source
    patched, _, _ = compose_eav(incoming, mode)
    if mode == 'disabled':
        assert patched is incoming
    else:
        assert patched is not incoming
    assert source.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY)['binding_hash'] == binding['binding_hash']


def test_eav_relay_composer_must_not_accept_a_forged_relay_override():
    binding, _ = bound_layout('video_only_paper')
    source, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)

    def foreign(*args, **kwargs):
        pytest.fail('forged attention must not run')

    foreign._t8_prompt_relay_binding_hash = binding['binding_hash']
    source.model_options['transformer_options']['optimized_attention_override'] = foreign
    with pytest.raises(RuntimeError, match='override|owner|binding'):
        compose_eav(source, 'apply_exp')


@pytest.mark.parametrize('downstream_sparse', [False, True])
def test_eav_relay_composer_runtime_reaches_pairing_guard(downstream_sparse):
    from h3_audio_t8_pkg.enhance_a_video_advanced import EAV_PROMPT_RELAY_WRAPPER_KEY

    binding, _ = bound_layout('video_only_paper')
    source, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    patched, _, _ = compose_eav(source, 'apply_exp')
    if downstream_sparse:
        patched = sparse(patched)
    patched.prepare_state(torch.tensor(.5), patched.model_options)
    wrappers = patched.get_wrappers('diffusion_model', EAV_PROMPT_RELAY_WRAPPER_KEY)
    executor = WrapperExecutor.new_executor(lambda *a, **k: pytest.fail('unpaired diffusion must not run'), wrappers)
    with pytest.raises(RuntimeError, match='binding hashes differ'):
        executor.execute(None, None, None, patched.model_options['transformer_options'])


def projected_relay_model():
    from h3_audio_t8_pkg.long_video import patch_long_video_model
    from h3_audio_t8_pkg.prompt_relay_long_video_advanced import (
        PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
        PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
    )

    binding, _ = bound_layout('video_only_paper')
    patched, _ = relay.patch_prompt_relay_model(model_fixture(), binding, 32)
    patched = patch_long_video_model(patched)
    patched.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, {
        'schema': PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
        'global_plan_hash': 'small_cpu_global_plan', 'projected_plan_hash': 'small_cpu_projected_plan',
        'binding_hash': binding['binding_hash'], 'segment_index': 1,
    })
    return patched


@pytest.mark.parametrize('intermediate_sparse', [False, True])
@pytest.mark.parametrize('mode', ['disabled', 'report_only', 'apply_exp'])
def test_eav_relay_long_video_entry_preserves_projected_owner(mode, intermediate_sparse):
    from h3_audio_t8_pkg.enhance_a_video_advanced import build_eav_prompt_relay_long_video_model

    source = projected_relay_model()
    incoming = sparse(source) if intermediate_sparse else source
    sigmas = torch.cat([torch.linspace(1., .05, 20), torch.zeros(1)])
    patched, runtime, _ = build_eav_prompt_relay_long_video_model(
        incoming, sigmas, segment_index=1, context_frames=22, mode=mode,
        tau=4., start_video_progress=0., end_video_progress=1., max_workspace_mib=32, g_hard_limit=1.5,
    )
    assert runtime.config['prompt_relay_contract']['global_plan_hash'] == 'small_cpu_global_plan'
    if mode == 'disabled':
        assert patched is incoming
    else:
        assert patched is not incoming
