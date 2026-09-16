"""Read-only, identical profiling policy for the two fixed thermal recipes."""
from copy import deepcopy

from .progressive_qualification import CallAudit


def observe_thermal_backend(model, profile, capture_v2, capture_dense, attention):
    receipt = capture_v2(model)
    if profile == 'trained_v2_dmd8':
        if receipt is None or receipt.profile != 'trained_vsa_exp' or receipt.runtime.head_chunks != 1:
            raise ValueError('Thermal V2 requires its authenticated trained h1 recipe')
        sparse = receipt.runtime.sparse
        functions = dict(selector=sparse.h3_sparse_attention, selected_kernel=sparse.ck.sol_attn_chunked,
                         pytorch=attention.attention_pytorch)
        contract = dict(kind='native_h3_learned_vsa_chunked_producer', profile=receipt.profile)
    elif profile == 'production_ema_b_native8':
        if receipt is not None:
            raise ValueError('Production baseline must not carry a V2 owner')
        override = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
        backend = capture_dense(override)
        if backend is None or backend.report().get('kind') != 'audited_kj_selector':
            raise ValueError('Thermal baseline requires an authenticated KJ selector')
        functions = dict(selector=override, selected_kernel=backend.kernel,
                         pytorch=attention.attention_pytorch)
        contract = backend.report()
        contract.pop('completed_calls', None)
    else:
        raise ValueError('Unknown fixed thermal recipe')
    observer = CallAudit(functions)
    observer.backend_contract = contract
    observer.profile = profile
    return observer


def completed_thermal_backend(observer):
    counts = deepcopy(observer.counts)
    for name in ('selector', 'selected_kernel', 'pytorch'):
        row = counts[name]
        if row['calls'] != row['successful_returns'] or row['empty_returns']:
            raise RuntimeError('Thermal backend call did not complete: ' + name)
    selector, kernel = counts['selector']['calls'], counts['selected_kernel']['calls']
    if selector <= 0 or selector != kernel:
        raise RuntimeError('Thermal selector and selected kernel must both really execute')
    if observer.profile == 'trained_v2_dmd8' and selector != 400:
        raise RuntimeError('Fixed trained h1 thermal recipe must execute400 VSA producers')
    return dict(status='actual_thermal_backend_calls_completed', observed=True,
                profile=observer.profile, counts=counts, backend_contract=deepcopy(observer.backend_contract),
                policy='same thread-local CPython CallAudit across full sampler region on both recipes',
                scope='successful Python selector/kernel-wrapper calls; wall times include observation overhead, not bare CUDA-kernel timing')
