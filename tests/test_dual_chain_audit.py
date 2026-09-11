"""Ensure long-pilot evidence cannot quietly qualify another attention route."""
import importlib.util
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('dual_chain_audit_tested', TOOLS / 'audit_dual_model_chain.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(params=[
    ('kj', False, 'audited_kj_selector', {'sage:unbiased': 200}),
    ('kj', True, 'audited_kj_selector', {'sage:unbiased': 200, 'sage:biased': 3800}),
    ('sol', False, 'audited_sol_attn_selector', {'sol:completed': 200}),
    ('pytorch', False, 'native_core_pytorch_selector', {'pytorch:completed': 200}),
])
def case(request):
    backend, relay, kind, counts = request.param
    return backend, relay, {'completed_network_forwards': 4,
        'backend': {'kind': kind, 'completed_calls': dict(counts)}}


def test_exact_route_accepts(case):
    backend, relay, report = case
    module.validate_backend(report, backend, relay)


@pytest.mark.parametrize('fault', ['missing_forward', 'extra_forward', 'owner', 'fallback', 'missing_call'])
def test_route_fault_rejects(case, fault):
    backend, relay, report = case
    if fault == 'missing_forward':
        report['completed_network_forwards'] = 3
    elif fault == 'extra_forward':
        report['completed_network_forwards'] = 5
    elif fault == 'owner':
        report['backend']['kind'] = 'unknown_owner'
    elif fault == 'fallback':
        report['backend']['completed_calls']['sdpa:fallback'] = 1
    else:
        key = next(iter(report['backend']['completed_calls']))
        report['backend']['completed_calls'][key] -= 1
    with pytest.raises(ValueError):
        module.validate_backend(report, backend, relay)


def test_relay_sol_not_qualified_by_plain_counter():
    with pytest.raises(ValueError, match='Relay Sol'):
        module.validate_backend({'completed_network_forwards': 4, 'backend': {
            'kind': 'audited_sol_attn_selector', 'completed_calls': {'sol:completed': 200}}}, 'sol', True)
