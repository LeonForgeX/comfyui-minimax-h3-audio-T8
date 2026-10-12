"""Draft-only identity for Kijai SolAttnPatch + native H3 + KJ Sage.

Inspect the real factories, selected delegates and installed hooks. Never swap
forwards, delete hooks, change Sol gates or substitute attention during sampling.
Morton reordering/probing/custom delegates remain outside this scoped adapter.
"""
from __future__ import annotations

import math
import ast
from pathlib import Path
import sys
from types import FunctionType, MethodType

import torch

from .progressive_source_audit import SourceAudit, module_for, require, source_digest

_SOURCE = '76dcbf6f5c58c1f655ee674aecddb165c6069408bc6058e3a5b2383d7198509f'
_HOOKS = '6c6468f3469bf3b66b68a2515632511ab56d0ebe31ce19d8603a6d3f0991a2e5'
_KERNELS = {
    '_tri_fwd.py': '4d2a8a1ae1524e5a1e2d52a034bfdfd42ba6c1cf3e1f03d29b877f2f57974b24',
    '_int8_fwd.py': '17b42a3481e65539493bf0abc0a29f081ca53c02d0f72d4ce32a2a637f378547',
    '_preprocess.py': 'efe30d6aa72debd04c5f78860cdfb01e98cfa8503e4603d8392aca42af1dd8ee',
    '_fused_prep.py': '970d9756033b5c4a992b48bb656e5a365a21380b16d5af4ed7a51b71a4cbd9ca',
    '_fused_quant.py': '24b36b7d8fe31bd3c7eb2259dc8eb7f63ceed2ee0aa340abc2bdedac78152502',
    '_autotune_log.py': 'c9156e74f4cb8ed9e2be16c08885415c96daa093153fedcf0461e6868e9fa3bb',
}


def _kernel_contract(module):
    """Bind installed kernel source and actual entry points, not node labels."""
    import importlib.metadata
    directory = Path(module.__file__).parent
    for name, digest in _KERNELS.items():
        require(source_digest(directory / name) == digest, 'Sol draft kernel source revision is not audited')
        loaded = sys.modules.get(module.__name__ + '.' + name[:-3])
        require(loaded is not None, 'Sol draft kernel module is not loaded')
        audit = SourceAudit(loaded, digest)
        audit.top_functions()
        # Triton wraps decorated Python kernels in JITFunction/Autotuner. Check
        # the underlying live code as well as the on-disk implementation.
        for node in audit.tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.decorator_list:
                continue
            function = getattr(loaded, node.name, None)
            for _ in range(4):
                if type(function) is FunctionType:
                    break
                require(type(function).__module__.startswith('triton.'), 'Sol draft JIT owner was replaced')
                function = getattr(function, 'fn', None)
            audit.function(function, node.name)
    for attribute, filename, symbol in (
            ('_sol_attn_kernel', '_tri_fwd.py', 'sol_attn'),
            ('_sol_attn_int8_kernel', '_int8_fwd.py', 'sol_attn_int8')):
        function = getattr(module, attribute, None)
        owner = module_for(function)
        require(owner is not None and Path(owner.__file__).resolve() == (directory / filename).resolve()
                and getattr(owner, symbol, None) is function and owner.torch is torch,
                'Sol draft kernel entry point was replaced')
        SourceAudit(owner, _KERNELS[filename]).function(function, symbol)
    versions = {}
    for name in ('triton', 'triton-windows'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    require(versions, 'Sol draft requires the installed Triton runtime')
    return {'sources': dict(_KERNELS), 'runtime': versions}


def _bound(method, module, name):
    return isinstance(method, MethodType) and method.__self__ is module and method.__func__ is getattr(type(module), name)


class SolDraftStack:
    def __init__(self, model, audit, override):
        self.model, self.audit = model, audit
        self.network = model.model.diffusion_model
        self.options = model.model_options.get('transformer_options', {})
        module = audit.module
        audit.top_functions()
        require(module.torch is torch and module.HEAD_DIM == 128 and module.BLOCK_SIZE == 64,
                'Sol draft globals differ from the audited source')
        cells = audit.function(override, 'make_override.<locals>.override')
        fields = {'tau', 'min_tokens', 'sigma_start', 'sigma_end', 'verbose', 'int8_qk',
                  'sink_conditioning', 'use_tma', 'dense_blocks', 'tau_profile', 'int8_pv', 'previous'}
        require(set(cells) == fields, 'Sol draft selector closure changed')
        require(cells['previous'] is None and cells['dense_blocks'] == frozenset()
                and type(cells['dense_blocks']) is frozenset and type(cells['tau_profile']) is dict
                and not cells['tau_profile'] and not self.options.get('sol_morton')
                and cells['sink_conditioning'] in ('exact_kv', 'exact_kv_and_rows'),
                'Sol draft selector uses an unaudited delegate, depth gate or Morton mode')
        for name in ('tau', 'sigma_start', 'sigma_end'):
            require(type(cells[name]) in (int, float) and math.isfinite(cells[name]), 'Sol draft has invalid gates')
        require(type(cells['min_tokens']) is int and cells['min_tokens'] > 0,
                'Sol draft has invalid token gate')
        for name in ('verbose', 'int8_qk', 'use_tma', 'int8_pv'):
            require(type(cells[name]) is bool, 'Sol draft settings have an unknown schema')
        gate = {key: cells[key] for key in ('sigma_start', 'sigma_end', 'min_tokens')}
        require(self.options.get('sol_compose') == gate, 'Sol draft compose and selector gates differ')
        self.contract = {'kind': 'kj_sol_attn_triton_h3_draft_v1', 'source': audit.digest,
                         'settings': {key: cells[key] for key in fields - {'previous', 'dense_blocks', 'tau_profile'}},
                         'kernels': _kernel_contract(module)}
        self._hooks()

    def unwrap(self, function, owner):
        cells = self.audit.function(function, '_compose_module_patch.<locals>.forward', attributes=('_sol_composed',))
        require(function._sol_composed is True and set(cells) == {'module', 'patched_forward', 'stock'}
                and cells['module'] is owner and cells['stock'] is type(owner).forward,
                'Sol draft composed forward owner changed')
        return cells['patched_forward']

    def memory(self):
        from .relay_kj_memory import inspect_memory_composition
        clone = self.model.clone()
        clone.is_injected = False
        self.methods = {}
        for index, block in enumerate(self.network.blocks):
            key = f'diffusion_model.blocks.{index}.attn.forward'
            function = clone.object_patches.get(key)
            require(function is not None, 'Sol draft adapter requires the complete KJ Sage forward set')
            clone.object_patches[key] = self.unwrap(function, block.attn)
            self.methods[key] = function
        memory = inspect_memory_composition(clone)
        require(memory is not None and memory['kind'] == 'kj_memory_sage' and memory['backend'] is None,
                'Sol draft composed KJ memory owner is not audited')
        return memory

    def _hooks(self):
        model = self.network
        from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
        require(type(model) is MiniMaxH3Model and len(model.blocks) > 0,
                'Sol draft hooks require the native H3 network')
        hooks = module_for(model._forward)
        audit = SourceAudit(hooks, _HOOKS)
        audit.top_functions()
        require(hooks.torch is torch and hooks.BLOCK_SIZE == 64, 'Sol draft layout globals changed')
        forward = audit.function(model._forward, 'install_h3_morton.<locals>._forward')
        rope = audit.function(model.rope_freqs, 'install_h3_morton.<locals>.rope_freqs')
        require(set(forward) == {'model', 'original_forward'} and forward['model'] is model
                and _bound(forward['original_forward'], model, '_forward')
                and set(rope) == {'model', 'original_rope_freqs'} and rope['model'] is model
                and _bound(rope['original_rope_freqs'], model, 'rope_freqs'),
                'Sol draft model delegates differ from native H3')
        layout = audit.function(PackedLayout.__init__, '_patch_packed_layout.<locals>.__init__')
        original = layout.get('original_init')
        core = sys.modules[MiniMaxH3Model.__module__]
        from .progressive_native_bypass_identity import _source
        require(set(layout) == {'original_init'}, 'Sol draft layout delegate changed')
        _source(core, [('PackedLayout.__init__', original)])
        require(not getattr(model, '_sol_morton_active', False)
                and all(getattr(model, key, None) is None for key in
                        ('_sol_morton_span', '_sol_morton_state', '_sol_transformer_options')),
                'Sol draft capture encountered a running or stale hook state')
        # Layout lookup entries are derived runtime data. Validate their exact
        # native owner/bounds rather than binding addresses to persistent drafts.
        for identity, entry in hooks._SPANS.items():
            require(type(entry) is tuple and len(entry) == 3, 'Sol draft layout cache is malformed')
            item, bounds, _ = entry
            require(type(item) is PackedLayout and id(item.position_ids) == identity
                    and bounds == next(((a, b) for a, b, kind in item.segments if kind == 'video'), None),
                    'Sol draft layout cache does not match native video bounds')
        self.hook_sets = {}
        for index, block in enumerate(model.blocks):
            pre = list(block._forward_pre_hooks.values())
            post = list(block._forward_hooks.values())
            require(len(pre) == 2 and len(post) == (1 if index == len(model.blocks) - 1 else 0)
                    and not block._forward_pre_hooks_with_kwargs and not block._forward_hooks_with_kwargs
                    and not block._forward_hooks_always_called, 'Sol draft has additional or changed block hooks')
            first = audit.function(pre[0], 'install_h3_morton.<locals>.pre_hook')
            second = self.audit.function(pre[1], '_install_compose_hooks.<locals>.pre_hook')
            require(set(first) == {'first', 'model'} and first['first'] is model.blocks[0] and first['model'] is model
                    and second == {'attn_attr': 'attn'}, 'Sol draft block hook closure changed')
            if post:
                tail = audit.function(post[0], 'install_h3_morton.<locals>.post_hook')
                require(set(tail) == {'model'} and tail['model'] is model, 'Sol draft output hook closure changed')
            self.hook_sets[id(block)] = (tuple(pre), tuple(post))
        self.contract['layout_source'] = audit.digest

    def hooks_match(self, module):
        expected = self.hook_sets.get(id(module), ((), ()))
        return (tuple(module._forward_pre_hooks.values()), tuple(module._forward_hooks.values())) == expected

    def forward_matches(self, name, module, current, expected):
        if current is None or _bound(current, module, 'forward'):
            return True
        if name + '.forward' not in self.methods or expected is None:
            return False
        if type(current) is FunctionType:
            current = self.unwrap(current, module)
        return (isinstance(current, MethodType) and isinstance(expected, MethodType)
                and current.__self__ is module and current.__func__ is expected.__func__)


def inspect_sol(model):
    function = model.model_options.get('transformer_options', {}).get('optimized_attention_override')
    if not (type(function) is FunctionType and function.__qualname__ == 'make_override.<locals>.override'):
        return None
    audit = SourceAudit(module_for(function), _SOURCE)
    return SolDraftStack(model, audit, function)
