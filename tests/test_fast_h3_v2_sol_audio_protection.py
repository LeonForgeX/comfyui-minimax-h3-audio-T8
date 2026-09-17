"""Actual installed selector ownership and CPU routing; not full-model quality."""
from itertools import permutations

from comfy.patcher_extension import CallbacksMP
import pytest
import torch

from h3_audio_t8_pkg import fast_h3_v2_advanced as v2
from h3_audio_t8_pkg import h3_memory_advanced as memory
from h3_audio_t8_pkg import long_video_dual_identity as identity
from test_fast_h3_v2_memory_bridge import _gated_model
from test_fast_h3_v2_stage_identity import _setup
from test_relay_sol_backend import installed_sol as installed_sol_fixture
from test_relay_kj_backend import kj as kj_fixture

installed_sol = installed_sol_fixture
kj = kj_fixture


def _sol_source(installed_sol):
    return installed_sol.SolAttentionPatch().patch(_gated_model(), True, .5, min_tokens=256,
        strict=True, thresh_type='diag', int8_qk=False, int8_pv=False)[0]


def test_only_authenticated_sol_gets_prefix_adapter_and_source_is_unchanged(installed_sol):
    source = _sol_source(installed_sol)
    original = source.model_options['transformer_options']['optimized_attention_override']
    model = _setup(source, 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    assert runtime.previous_override is original
    assert runtime.override is not original
    assert runtime.dense_sol_backend.kernel is installed_sol.sol_attn
    assert runtime.dense_sol_backend.config['tau'] == .5
    assert runtime.dense_sol_contract()['h3_exact_prefix']['policy'] == 'packed_nonvideo_q_and_kv_exact_v1'
    assert runtime.snapshot()['dense_sol_backend']['completed_calls'] == {}
    assert runtime.snapshot()['attention_dispatch_observed'] is True
    assert source.model_options['transformer_options']['optimized_attention_override'] is original
    assert source.get_attachment(v2.KEY) is None


def test_real_cpu_fallback_is_counted_and_keeps_inputs_and_named_mask(installed_sol):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    q, k, v = [torch.randn(1, 3, 9, 128, dtype=torch.bfloat16) for _ in range(3)]
    saved = [value.clone() for value in (q, k, v)]
    mask = torch.zeros(9, 9, dtype=q.dtype)
    mask[:, :3] = -4
    actual = runtime.override(None, q, k, v, 3, mask=mask, skip_reshape=True,
        skip_output_reshape=True, transformer_options=model.model_options['transformer_options'])
    torch.testing.assert_close(actual, torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        rtol=.02, atol=.001)
    assert runtime.snapshot()['dense_sol_backend']['completed_calls'] == {'fallback:relay_or_other_mask': 1}
    assert runtime.snapshot()['actual_vsa_dispatched'] is False
    for before, after in zip(saved, (q, k, v), strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize('order', list(permutations(('v2', 'attention', 'ffn'))))
def test_sol_and_t8_memory_both_orders_rebind_fresh_protected_runtime(installed_sol, order):
    source = _sol_source(installed_sol)
    original = source.model_options['transformer_options']['optimized_attention_override']
    model = source
    for step in order:
        if step == 'v2':
            model = _setup(model, 'dense_compat_exp')
        elif step == 'attention':
            model, _ = memory.configure_low_vram_attention(model, 2)
        else:
            model, _ = memory.configure_chunk_feed_forward(model, 2, 256)
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    assert runtime.previous_override is original
    assert runtime.head_chunks == 2
    assert model.get_wrappers('diffusion_model', v2.KEY) == [runtime.guard]
    assert model.callbacks[CallbacksMP.ON_PREPARE_STATE][v2.KEY] == [runtime.prepare]
    assert identity.stage_model_identity(model)['fast_h3_v2']['protected_sol']['configuration']['tau'] == .5


def test_protected_sol_cache_identity_excludes_counts_but_binds_tau_and_source(installed_sol):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    first = identity.stage_model_identity(model)['sha256']
    runtime.dense_sol_backend.counters['sol:completed'] = 42
    runtime.dense_sol_backend.last_exact_prefix = (0, 7)
    assert identity.stage_model_identity(model.clone())['sha256'] == first
    runtime.dense_sol_backend.config['tau'] = .7
    with pytest.raises(RuntimeError, match='protected Sol backend was replaced'):
        identity.stage_model_identity(model)


@pytest.mark.parametrize('mutation', ['kernel', 'override', 'prepare', 'cleanup', 'wrapper', 'method'])
def test_protected_sol_allows_selected_override_but_rejects_private_backend_damage(installed_sol, mutation, caplog):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    def foreign(*args, **kwargs):
        pytest.fail('foreign implementation ran')
    if mutation == 'kernel':
        runtime.dense_sol_backend.kernel = foreign
    elif mutation == 'override':
        model.model_options['transformer_options']['optimized_attention_override'] = foreign
    elif mutation == 'method':
        runtime.dense_sol_backend.attention = foreign
    elif mutation == 'wrapper':
        model.wrappers['diffusion_model'][v2.KEY] = [foreign]
    else:
        role = CallbacksMP.ON_PREPARE_STATE if mutation == 'prepare' else CallbacksMP.ON_CLEANUP
        model.callbacks[role][v2.KEY] = [foreign]
    if mutation == 'override':
        assert v2.capture_fast_h3_v2_owner(model).runtime is runtime
        assert model.model_options['transformer_options']['optimized_attention_override'] is foreign
        assert 'advisory' in caplog.text
    else:
        with pytest.raises(RuntimeError, match='was replaced'):
            v2.capture_fast_h3_v2_owner(model)


def test_prepare_resets_own_sol_counts_and_cleanup_retains_actual_calls(installed_sol):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    runtime.dense_sol_backend.counters['sol:completed'] = 5
    runtime.dense_sol_backend.last_exact_prefix = (0, 9)
    runtime.prepare(model, None, model.model_options)
    assert not runtime.dense_sol_backend.counters
    assert runtime.dense_sol_backend.last_exact_prefix is None
    runtime.dense_sol_backend.counters['sol:completed'] = 8
    runtime.cleanup(model)
    assert runtime.dense_sol_backend.counters['sol:completed'] == 8
    runtime.prepare(model, None, model.model_options)
    assert not runtime.dense_sol_backend.counters


def test_returned_contract_cannot_mutate_active_sol_configuration(installed_sol):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    contract = runtime.dense_sol_contract()
    contract['configuration']['tau'] = 3.
    assert runtime.dense_sol_backend.config['tau'] == .5
    v2.capture_fast_h3_v2_owner(model)


def test_plain_dense_profile_does_not_install_sol_or_new_wrappers():
    model = _setup(profile='dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    assert runtime.dense_sol_backend is None
    assert not model.get_wrappers('diffusion_model', v2.KEY)


def _sol_with_kj(installed_sol, kj):
    source, = kj[0].PathchSageAttentionKJ().patch(_gated_model(), 'auto', False)
    return installed_sol.SolAttentionPatch().patch(source, True, .5, min_tokens=256,
        strict=True, thresh_type='diag', int8_qk=False, int8_pv=False)[0]


@pytest.mark.parametrize('mutation', ['kernel', 'masked_kernel', 'attention', '_selected_attention', 'report'])
def test_nested_kj_fallback_replacements_rejected_before_execution(installed_sol, kj, mutation):
    model = _setup(_sol_with_kj(installed_sol, kj), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    fallback = runtime.dense_sol_backend.fallback
    assert fallback is not None
    first = identity.stage_model_identity(model)['sha256']
    fallback.counters['pytorch:non_cuda'] = 7
    assert identity.stage_model_identity(model)['sha256'] == first
    def foreign(*args, **kwargs):
        pytest.fail('foreign nested fallback executed')
    setattr(fallback, mutation, foreign)
    with pytest.raises(RuntimeError, match='protected Sol backend was replaced'):
        v2.capture_fast_h3_v2_owner(model)
    with pytest.raises(RuntimeError, match='protected Sol backend was replaced'):
        identity.stage_model_identity(model)
    q = torch.zeros(1, 3, 9, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='protected Sol backend was replaced'):
        runtime.override(None, q, q, q, 3, mask=torch.zeros(9, 9), skip_reshape=True,
            transformer_options=model.model_options['transformer_options'])


def test_nested_fallback_counts_are_per_run_and_original_source_is_unchanged(installed_sol, kj):
    source = _sol_with_kj(installed_sol, kj)
    original = source.model_options['transformer_options']['optimized_attention_override']
    model = _setup(source, 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    q, k, v = [torch.randn(1, 3, 9, 128, dtype=torch.bfloat16) for _ in range(3)]
    mask = torch.zeros(9, 9, dtype=q.dtype)
    for _ in range(2):
        runtime.prepare(model, None, model.model_options)
        assert not runtime.dense_sol_backend.fallback.counters
        actual = runtime.override(None, q, k, v, 3, mask=mask, skip_reshape=True,
            skip_output_reshape=True, transformer_options=model.model_options['transformer_options'])
        torch.testing.assert_close(actual, torch.nn.functional.scaled_dot_product_attention(q, k, v,
            attn_mask=mask), rtol=.02, atol=.001)
        runtime.cleanup(model)
        assert runtime.snapshot()['dense_sol_backend']['fallback']['completed_calls'] == {'pytorch:non_cuda': 1}
    assert source.model_options['transformer_options']['optimized_attention_override'] is original


@pytest.mark.parametrize('method', ['attention', 'report'])
def test_sol_class_execution_replacements_rejected_before_report(installed_sol, monkeypatch, method):
    model = _setup(_sol_source(installed_sol), 'dense_compat_exp')
    runtime = v2.capture_fast_h3_v2_owner(model).runtime
    def foreign(*args, **kwargs):
        pytest.fail('foreign Sol method executed')
    monkeypatch.setattr(type(runtime.dense_sol_backend), method, foreign)
    with pytest.raises(RuntimeError, match='protected Sol backend was replaced'):
        v2.capture_fast_h3_v2_owner(model)
