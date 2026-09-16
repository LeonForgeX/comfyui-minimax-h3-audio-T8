from types import SimpleNamespace
import sys

import pytest

from tools.fast_h3_v2_backend_audit import observe_dense_backend, validate_completed_backend
from tools.run_fast_h3_v2_probe import build_graph


def test_backend_observer_imports_inside_isolated_custom_node_namespace(monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import ModuleType
    name = 'fasth3_isolated_probe_tools'
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'tools')]
    monkeypatch.setitem(sys.modules, name, package)
    spec = importlib.util.spec_from_file_location(name + '.fast_h3_v2_backend_audit',
        Path(package.__path__[0]) / 'fast_h3_v2_backend_audit.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
        assert module.CallAudit.__module__ == name + '.progressive_qualification'
    finally:
        sys.modules.pop(name + '.progressive_qualification', None)


def test_selector_observer_does_not_replace_model_or_functions_and_restores_profiler():
    def kernel():
        return object()
    def selector():
        return kernel()
    contract = {'kind': 'test_authenticated_selector', 'completed_calls': {}}
    model = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': selector}})
    backend = SimpleNamespace(kernel=kernel, report=lambda: contract.copy())
    captured = []
    def capture(value):
        captured.append(value)
        return backend
    observer = observe_dense_backend(model, capture)
    previous = sys.getprofile()
    with observer:
        selector()
    report = validate_completed_backend(observer)
    assert captured == [selector]
    assert model.model_options['transformer_options']['optimized_attention_override'] is selector
    assert sys.getprofile() is previous
    assert report['counts']['selector']['successful_returns'] == 1
    assert report['counts']['selected_kernel']['successful_returns'] == 1
    assert report['pytorch_fallback_observed'] is False
    assert 'completed_calls' not in report['backend_contract']
    assert 'completed_calls' in contract


def test_unknown_dense_override_is_not_renamed_or_silently_unobserved():
    model = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': lambda: None}})
    with pytest.raises(ValueError, match='authenticated'):
        observe_dense_backend(model, lambda _: None)
    assert observe_dense_backend(SimpleNamespace(model_options={}), lambda _: None) is None


@pytest.mark.parametrize('failure', ['no_calls', 'exception'])
def test_missing_or_failed_kernel_calls_cannot_claim_compatibility(failure):
    def kernel():
        raise RuntimeError('kernel error')
    def selector():
        return kernel()
    model = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': selector}})
    backend = SimpleNamespace(kernel=kernel, report=lambda: {'kind': 'test'})
    observer = observe_dense_backend(model, lambda _: backend)
    with observer:
        if failure == 'exception':
            with pytest.raises(RuntimeError, match='kernel error'):
                selector()
    assert sys.getprofile() is None
    with pytest.raises(RuntimeError, match='not observed'):
        validate_completed_backend(observer)


def test_sol_probe_uses_explicit_dense_strict_scope_without_altering_trained_profile():
    graph, reports = build_graph(profile='dense_compat_exp', backend='sol')
    assert graph['3']['class_type'] == 'SolAttentionPatch'
    assert graph['3']['inputs']['strict'] is True
    assert graph['3']['inputs']['tau'] == .5
    assert graph['10']['inputs']['model'] == ['3', 0]
    assert graph['10']['inputs']['profile'] == 'dense_compat_exp'
    assert reports['network'] == '22'


@pytest.mark.parametrize('prefix', [None, (0, 0), (0, 3)])
def test_protected_sol_probe_requires_actual_prefix_not_a_static_policy(prefix):
    def kernel():
        return object()
    def selector():
        return kernel()
    backend = SimpleNamespace(kernel=kernel, report=lambda: {
        'kind': 'test_protected_backend', 'h3_exact_prefix': {'last_block_range': prefix}})
    receipt = SimpleNamespace(runtime=SimpleNamespace(dense_sol_backend=backend))
    model = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': selector}})
    def reject_direct(_):
        pytest.fail('must observe the authenticated runtime adapter, not original selector')
    observer = observe_dense_backend(model, reject_direct, capture_v2_owner=lambda _: receipt)
    with observer:
        selector()
    if prefix == (0, 3):
        report = validate_completed_backend(observer)
        assert report['backend_contract']['observed_selector_route'] == 'v2_h3_exact_prefix_adapter'
    else:
        with pytest.raises(RuntimeError, match='not actually observed'):
            validate_completed_backend(observer)
