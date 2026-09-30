"""Public finished-stage release through Core; unrelated residency is preserved."""
from __future__ import annotations

import torch

from .long_video_dual_residency import release_stage_residency

CAPABILITY = 't8.stage_residency_release.v1'


def _snapshot(components):
    devices = {str(getattr(getattr(value, 'patcher', value), 'load_device', 'cpu'))
               for value in components}
    result = []
    for name in sorted(devices):
        device = torch.device(name)
        row = dict(device=name, kind='boundary_snapshot_not_peak')
        if device.type == 'cuda' and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device)
            row.update(free_bytes=int(free), total_bytes=int(total),
                torch_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                torch_reserved_bytes=int(torch.cuda.memory_reserved(device)))
        result.append(row)
    return result


def release_finished_stage(*components):
    components = tuple(value for value in components if value is not None)
    if not components:
        raise ValueError('Select at least one finished-stage MODEL/CLIP/VAE')
    before = _snapshot(components)
    report = release_stage_residency(*components)
    return dict(schema=1, capability=CAPABILITY, before=before, after=_snapshot(components),
                release=report, numerical_sampling_changed=False,
                peak_claim=False, guarantee_against_oom=False)
