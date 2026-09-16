"""Test-only call observation of authenticated, unchanged Dense selectors."""
from copy import deepcopy

from .progressive_qualification import CallAudit


def observe_dense_backend(model, capture_backend, *, capture_v2_owner=None):
    from comfy.ldm.modules import attention
    override = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
    if override is None:
        return None
    receipt = capture_v2_owner(model) if capture_v2_owner is not None else None
    protected = receipt.runtime.dense_sol_backend if receipt is not None else None
    backend = protected if protected is not None else capture_backend(override)
    if backend is None:
        raise ValueError('Dense selector call probe requires an authenticated KJ/Sol owner')
    contract = backend.report()
    contract.pop('completed_calls', None)
    contract['observed_selector_route'] = 'v2_h3_exact_prefix_adapter' if protected is not None else 'external_direct_override'
    observer = CallAudit({'selector': override, 'selected_kernel': backend.kernel,
                          'pytorch': attention.attention_pytorch})
    observer.backend_contract = contract
    observer.protected_backend = protected
    return observer


def validate_completed_backend(observer):
    counts = deepcopy(observer.counts)
    for key in ('selector', 'selected_kernel'):
        row = counts[key]
        if row['calls'] <= 0 or row['calls'] != row['successful_returns'] or row['empty_returns']:
            raise RuntimeError('Dense selector/kernel was not observed completing: ' + key)
    contract = deepcopy(observer.backend_contract)
    if observer.protected_backend is not None:
        contract.update(observer.protected_backend.report())
        prefix = contract['h3_exact_prefix']['last_block_range']
        if not prefix or prefix[0] != 0 or prefix[1] <= 0:
            raise RuntimeError('Protected Sol exact prefix was not actually observed')
    return {'status': 'actual_selected_backend_calls_completed', 'counts': counts,
            'backend_contract': contract,
            'pytorch_fallback_observed': counts['pytorch']['calls'] > 0,
            'scope': 'read-only thread-local Python calls; not CUDA timing or quality equivalence'}
