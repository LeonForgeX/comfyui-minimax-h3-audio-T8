"""Fresh bounded parallel hashing of inert, already-CPU native weight tensors.

CUDA copies stay on the caller's stream/thread. No tensor hashes or storage
versions survive this call. Unknown tensor/container readers remain sequential.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch


def weight_state_identity(state):
    from .long_video_dual_identity import content_identity
    if type(state) not in (dict, OrderedDict) or any(type(key) is not str for key in state):
        return content_identity(state)
    # An effectful reader could mutate a different tensor while a worker is
    # reading it. In that case the complete capture stays on the caller thread.
    if any(type(value) not in (torch.Tensor, torch.nn.Parameter) or vars(value) for value in state.values()):
        return content_identity(state)
    eligible = {}
    for key, value in state.items():
        if (type(value) in (torch.Tensor, torch.nn.Parameter) and not vars(value)
                and value.device.type == 'cpu' and value.layout == torch.strided
                and not value.is_quantized and value.is_contiguous()
                and value.numel() * value.element_size() >= 1024 * 1024):
            eligible[key] = value
    if len(eligible) < 2 or sum(v.numel() * v.element_size() for v in eligible.values()) < 32 * 1024**2:
        return content_identity(state)
    # Four independent tensors at most. Existing per-tensor finite checks and
    # 1/4MiB scratch/transfer limits are unchanged; total scratch stays bounded.
    # Check cancellation inside each original tensor reader. The context joins
    # every worker even when an integrity error propagates to the caller.
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='h3-weight-digest') as pool:
        futures = {key: pool.submit(content_identity, value) for key, value in eligible.items()}
        return {key: futures[key].result() if key in futures else content_identity(state[key])
                for key in sorted(state)}
