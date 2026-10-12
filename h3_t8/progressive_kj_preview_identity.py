"""Authenticate KJ's H3 preview observer without removing it from execution."""
from __future__ import annotations

import torch

from .progressive_source_audit import SourceAudit, module_for, require

_PREVIEW = 'd07c8e5ff7d7abd3e31acefe2cf29f618b5da5341b3b434e72ec2ac0953230b0'
_TINY = '61bf4daa634ef6bb901ff8630071caed12e733d17e7c4ba50082dc0aa5506591'


def preview_keys(model):
    """Only the actual source-bound read-only H3 observer may be identity-neutral.

VAE inputs and other model families need their own audit. Display settings,
node ids and the generated PIL frame list cannot change H3 denoising.
"""
    from comfy.patcher_extension import WrappersMP
    wrappers = model.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).get('kj_preview_override', [])
    if not wrappers:
        return set()
    from comfy.latent_formats import MiniMaxH3AV, MiniMaxH3Video
    require(type(model.model.latent_format) in (MiniMaxH3AV, MiniMaxH3Video), 'KJ preview draft adapter requires native H3 latents')
    require(len(wrappers) == 1, 'KJ preview draft adapter requires one observer')
    wrapper = wrappers[0]
    cls = type(wrapper)
    module = module_for(cls)
    audit = SourceAudit(module, _PREVIEW)
    audit.plain_class(cls, '_PreviewOverrideWrapper')
    audit.plain_class(module._AsyncPreviewEncoder, '_AsyncPreviewEncoder', constants=('_STOP',))
    require(type(module._AsyncPreviewEncoder._STOP) is object, 'KJ preview encoder sentinel changed')
    audit.top_functions()
    require(module.torch is torch, 'KJ preview Torch owner was changed')
    require(set(vars(wrapper)) == {'max_resolution', 'node_id', 'jpeg_quality', 'suppress_default',
            'preview_frames', 'preview_fps', 'vae', 'tiny_vae', 'audio_vae', 'frames'},
            'KJ preview observer contains unknown execution state')
    require(wrapper.vae is None and wrapper.audio_vae is None,
            'KJ preview custom VAE inputs need a separate read-only audit')
    for value, lo, hi in ((wrapper.max_resolution, 0, 8192), (wrapper.jpeg_quality, 30, 100),
                          (wrapper.preview_frames, 1, 1024), (wrapper.preview_fps, 1, 60)):
        require(type(value) is int and lo <= value <= hi, 'KJ preview settings differ from the audited schema')
    require(type(wrapper.suppress_default) is bool and (wrapper.node_id is None or type(wrapper.node_id) is str)
            and type(wrapper.tiny_vae) is str and type(wrapper.frames) is list,
            'KJ preview settings differ from the audited schema')
    tiny = module_for(module.load_tiny_vae_decoder)
    tiny_audit = SourceAudit(tiny, _TINY)
    tiny_audit.top_functions()
    # The decoder is constructed afresh by the audited loader, not an arbitrary
    # user object supplied through the VAE socket. Its first layer allocates a
    # new tensor; in-place output clamps never write to the sampler's x0.
    for name in ('TinyVAEDecoder', 'TAEHVDecoder'):
        decoder = getattr(tiny, name)
        for method, function in vars(decoder).items():
            if callable(function):
                tiny_audit.function(function, name + '.' + method)
    require(module.load_tiny_vae_decoder is tiny.load_tiny_vae_decoder and tiny.torch is torch,
            'KJ preview tiny decoder owner was changed')
    return {'kj_preview_override'}
