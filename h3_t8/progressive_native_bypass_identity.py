"""First-pass-only identity for the exact Core inference LoRA bypass lifecycle.

This does not certify arbitrary injections, training adapters or callback stacks.
The selected patcher is never changed, and the public stage identity keeps its
original nonportable policy. Every boundary binds fresh adapter compute bytes.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import types

import torch

from .long_video_dual_identity import content_identity
from .patch_stack_policy import UnverifiedModelStack, _bypass_weight_identity, _configured_force_forward


def _codes(code):
    yield code
    for child in code.co_consts:
        if isinstance(child, types.CodeType):
            yield from _codes(child)


def _source(module, functions):
    """Compare live code to its loaded source, without executing that source."""
    path = Path(module.__file__)
    payload = path.read_bytes()
    compiled = tuple(_codes(compile(payload, str(path), 'exec', dont_inherit=True)))
    definitions = [node for node in ast.walk(ast.parse(payload)) if isinstance(node, ast.FunctionDef)]
    for function in functions:
        if (type(function) is not types.FunctionType or function.__globals__ is not vars(module)
                or function.__code__ not in compiled):
            raise UnverifiedModelStack('Native draft bypass live implementation differs from Core source')
        declarations = [node for node in definitions if node.name == function.__name__
                        and node.lineno == function.__code__.co_firstlineno]
        if len(declarations) != 1:
            raise UnverifiedModelStack('Native draft bypass source declaration is ambiguous')
        declaration = declarations[0]
        try:
            positional = tuple(ast.literal_eval(node) for node in declaration.args.defaults)
            keyword = {arg.arg: ast.literal_eval(node) for arg, node in
                       zip(declaration.args.kwonlyargs, declaration.args.kw_defaults) if node is not None}
        except (ValueError, TypeError):
            raise UnverifiedModelStack('Native draft bypass source has an unaudited default') from None
        try:
            defaults_match = (not (not _plain(function.__defaults__) or not _plain(function.__kwdefaults__))
                and content_identity(function.__defaults__) == content_identity(positional or None)
                and content_identity(function.__kwdefaults__) == content_identity(keyword or None))
        except ValueError:
            defaults_match = False
        if not defaults_match:
            raise UnverifiedModelStack('Native draft bypass live defaults differ from Core source')
    return {'sha256': hashlib.sha256(payload).hexdigest(),
            'defaults': [content_identity((fn.__defaults__, fn.__kwdefaults__)) for fn in functions]}


def _sources():
    from comfy import model_management as mm, model_patcher, patcher_extension
    from comfy.weight_adapter import bypass, base, lora
    if (bypass.torch is not torch or bypass.nn is not torch.nn
            or bypass.WeightAdapterBase is not base.WeightAdapterBase
            or bypass.WeightAdapterTrainBase is not base.WeightAdapterTrainBase
            or bypass.PatcherInjection is not patcher_extension.PatcherInjection
            or bypass.comfy.model_management is not mm
            or lora.torch is not torch or lora.F is not torch.nn.functional
            or lora.WeightAdapterBase is not base.WeightAdapterBase
            or lora.F.linear is not torch._C._nn.linear
            or any(getattr(lora.F, name) is not getattr(torch, name) for name in ('conv1d', 'conv2d', 'conv3d'))):
        raise UnverifiedModelStack('Native draft bypass has replaced Core execution globals')
    hook, manager = bypass.BypassForwardHook, bypass.BypassInjectionManager
    if (lora.LoRAAdapter.__bases__ != (base.WeightAdapterBase,)
            or any(cls.__bases__ != (object,) for cls in
                   (hook, manager, base.WeightAdapterBase, patcher_extension.PatcherInjection))):
        raise UnverifiedModelStack('Native draft bypass class has a replaced inheritance owner')
    for module, classes in ((bypass, (hook, manager)), (lora, (lora.LoRAAdapter,)),
                            (base, (base.WeightAdapterBase,)),
                            (patcher_extension, (patcher_extension.PatcherInjection,))):
        tree = ast.parse(Path(module.__file__).read_bytes())
        for cls in classes:
            declarations = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__]
            if len(declarations) != 1 or cls.__module__ != module.__name__:
                raise UnverifiedModelStack('Native draft bypass class differs from its Core owner')
            expected = {'__module__', '__doc__'}
            if cls.__bases__ == (object,):
                expected.update(('__dict__', '__weakref__'))
            for node in declarations[0].body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    expected.add(node.name)
                elif isinstance(node, ast.AnnAssign):
                    expected.add('__annotations__')
                    if node.value is not None:
                        expected.add(node.target.id)
                elif isinstance(node, ast.Assign):
                    expected.update(target.id for target in node.targets if isinstance(target, ast.Name))
            if set(vars(cls)) != expected:
                raise UnverifiedModelStack('Native draft bypass class contains unbound execution members')
    return {module.__name__: _source(module, functions) for module, functions in (
        (bypass, (bypass.get_module_type_info, hook.__init__, hook._bypass_forward,
                  hook.inject, hook.eject, hook._move_adapter_weights_to_device,
                  manager.__init__, manager.add_adapter, manager._get_module_by_key, manager.create_injections)),
        (lora, (lora.LoRAAdapter.__init__, lora.LoRAAdapter.h)),
        (base, (base.WeightAdapterBase.g, base.WeightAdapterBase.bypass_forward)),
        (patcher_extension, (patcher_extension.PatcherInjection.__init__,)),
        (model_patcher, (model_patcher.ModelPatcher.clone, model_patcher.ModelPatcher.get_clone_model_override,
                         model_patcher.ModelPatcher.model_state_dict, model_patcher.ModelPatcher.use_ejected,
                         model_patcher.ModelPatcher.inject_model, model_patcher.ModelPatcher.eject_model)),
        (mm, (mm.get_torch_device,)),
    )}


def _plain(value):
    if type(value) in (str, int, float, bool, type(None), torch.dtype, torch.device):
        return True
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        return not vars(value)
    if type(value) in (list, tuple):
        return all(_plain(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain(item) for key, item in value.items())
    return False


def _manager(injection):
    from comfy.weight_adapter.bypass import BypassInjectionManager
    from comfy.patcher_extension import PatcherInjection
    if type(injection) is not PatcherInjection or set(vars(injection)) != {'inject', 'eject'}:
        raise UnverifiedModelStack('First-pass bypass injection is not the exact Core factory')
    codes = {code.co_name: code for code in BypassInjectionManager.create_injections.__code__.co_consts
             if isinstance(code, types.CodeType) and code.co_name in ('inject_all', 'eject_all')}
    owners = []
    for role, name in (('inject', 'inject_all'), ('eject', 'eject_all')):
        function = getattr(injection, role)
        closure = getattr(function, '__closure__', None)
        if (type(function) is not types.FunctionType or function.__code__ is not codes.get(name)
                or function.__globals__ is not BypassInjectionManager.create_injections.__globals__
                or function.__defaults__ is not None or function.__kwdefaults__ is not None
                or function.__code__.co_freevars != ('self',) or not closure or len(closure) != 1):
            raise UnverifiedModelStack('First-pass bypass injection has a foreign execution owner')
        owners.append(closure[0].cell_contents)
    if (owners[0] is not owners[1] or type(owners[0]) is not BypassInjectionManager
            or set(vars(owners[0])) != {'adapters', 'hooks'}
            or type(owners[0].adapters) is not dict or type(owners[0].hooks) is not list
            or len(owners[0].adapters) != len(owners[0].hooks)):
        raise UnverifiedModelStack('First-pass bypass manager has unbound execution state')
    return owners[0]


def _selected(model):
    """Return ordered native selections and authenticated lifecycle owners."""
    import comfy.weight_adapter.bypass as bypass
    from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
    from comfy.weight_adapter.lora import LoRAAdapter
    from comfy.weight_adapter.base import WeightAdapterBase
    from .patch_stack_policy import _native_bypass_force_state

    if type(model) not in (ModelPatcher, ModelPatcherDynamic) or type(model.injections) is not dict:
        raise UnverifiedModelStack('First-pass bypass needs an exact native Core patcher')
    if type(model.skip_injection) is not bool or model.skip_injection is True:
        raise UnverifiedModelStack('First-pass bypass injection suppression is not a portable native execution')
    if type(model.is_injected) is not bool:
        raise UnverifiedModelStack('First-pass bypass loaded state is not native')
    for name in ('clone', 'model_state_dict', 'get_clone_model_override', 'use_ejected',
                 'inject_model', 'eject_model'):
        if getattr(getattr(model, name), '__func__', None) is not getattr(type(model), name):
            raise UnverifiedModelStack('First-pass bypass patcher method was replaced')
    modules = dict(model.model.named_modules())
    records, hooks = [], {}
    fields = {'loaded_keys', 'weights', 'multiplier', 'is_conv', 'conv_dim', 'kernel_size',
              'in_channels', 'out_channels', 'kw_dict'}
    for group, injections in model.injections.items():
        if type(group) is not str or type(injections) is not list:
            raise UnverifiedModelStack('First-pass bypass selection needs plain ordered Core groups')
        for injection in injections:
            manager = _manager(injection)
            targets = []
            for (key, selection), hook in zip(manager.adapters.items(), manager.hooks):
                if (type(key) is not str or type(selection) is not tuple or len(selection) != 2
                        or type(hook) is not bypass.BypassForwardHook
                        or _native_bypass_force_state(hook) is None
                        or modules.get(key) is not hook.module or hook.adapter is not selection[0]
                        or type(selection[1]) not in (float, int)
                        or type(hook.multiplier) not in (float, int) or hook.multiplier != selection[1]):
                    raise UnverifiedModelStack('First-pass bypass target/strength differs from its Core factory')
                adapter = hook.adapter
                if model.is_injected:
                    current, seen, installed = vars(hook.module).get('forward'), set(), False
                    while current is not None:
                        owner = getattr(current, '__self__', None)
                        if (type(owner) is not bypass.BypassForwardHook
                                or getattr(current, '__func__', None) is not bypass.BypassForwardHook._bypass_forward
                                or owner.module is not hook.module or id(owner) in seen):
                            break
                        seen.add(id(owner))
                        if owner is hook:
                            installed = True
                            break
                        current = owner.original_forward
                    if not installed or hook.original_forward is None:
                        raise UnverifiedModelStack('First-pass bypass loaded flag has no installed native owner')
                if (type(adapter) is not LoRAAdapter or set(vars(adapter)) != fields
                        or type(adapter.loaded_keys) is not set
                        or any(type(key) is not str for key in adapter.loaded_keys)
                        or type(adapter.weights) not in (list, tuple) or len(adapter.weights) != 6
                        or not _plain(adapter.weights)
                        or type(adapter.multiplier) not in (float, int) or adapter.multiplier != hook.multiplier
                        or getattr(adapter.h, '__func__', None) is not LoRAAdapter.h
                        or getattr(adapter.g, '__func__', None) is not WeightAdapterBase.g
                        or getattr(adapter.bypass_forward, '__func__', None) is not WeightAdapterBase.bypass_forward):
                    raise UnverifiedModelStack('First-pass bypass adapter is not the exact inference LoRA schema')
                info = bypass.get_module_type_info(hook.module)
                expected = {name: info[name] for name in ('is_conv', 'conv_dim', 'kernel_size',
                                                         'in_channels', 'out_channels')}
                expected['kw_dict'] = ({name: info[name] for name in ('stride', 'padding', 'dilation', 'groups')}
                                       if info['is_conv'] else {})
                if any(not _plain(getattr(adapter, name)) or type(getattr(adapter, name)) is not type(value)
                       or getattr(adapter, name) != value
                       for name, value in expected.items()):
                    raise UnverifiedModelStack('First-pass bypass adapter configuration differs from its native target')
                weight = getattr(hook.module, 'weight', None)
                dtype = getattr(weight, 'dtype', None)
                if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    dtype = None
                targets.append({'target': key, 'multiplier': content_identity(hook.multiplier),
                    'configuration': content_identity(expected), 'loaded_keys': sorted(adapter.loaded_keys),
                    'weights': _bypass_weight_identity(adapter.weights, dtype)})
                hooks[id(hook)] = hook
            records.append({'group': group, 'targets': targets})
    return records, hooks, modules


class _Projection:
    """Private generated proof; validation recaptures the selected actual state."""
    def __init__(self, model, source, contract, hooks):
        self.model, self.source, self.contract, self.hooks = model, source, contract, hooks


def _capture(model, source=None):
    from comfy import model_management as mm
    source = model if source is None else source
    if (source.model is not model.model or type(source.injections) is not dict
            or type(model.injections) is not dict or source.injections.keys() != model.injections.keys()
            or any(type(model.injections[group]) is not list or type(source.injections[group]) is not list
                   or len(source.injections[group]) != len(model.injections[group])
                   or any(left is not right for left, right in zip(source.injections[group], model.injections[group]))
                   for group in source.injections)):
        raise UnverifiedModelStack('First-pass bypass inspected clone differs from its selected source')
    implementations = _sources()
    records, hooks, modules = _selected(source)
    # A selected LOW clone can share an already loaded HIGH network. Project
    # that temporary lifecycle only after the same exact native authentication.
    for loaded in mm.current_loaded_models:
        if type(loaded) is not mm.LoadedModel:
            continue
        owner = loaded.model
        if owner is None or owner.model is not model.model or not owner.is_injected:
            continue
        try:
            _, active, _ = _selected(owner)
        except UnverifiedModelStack:
            continue
        hooks.update(active)
    for record in records:
        for target in record['targets']:
            module = modules[target['target']]
            target['force_forward'] = content_identity(_configured_force_forward(module, hooks))
    contract = {'schema': 't8.first_pass.native_core_bypass_lora.v1',
                'selections': records, 'implementations': implementations}
    return _Projection(model, source, contract, hooks)


def validate_projection(model, proof):
    """Called by the private audited stage path; no caller boolean skips proof."""
    if type(proof) is not _Projection or proof.model is not model:
        raise UnverifiedModelStack('First-pass bypass projection belongs to another patcher')
    current = _capture(model, proof.source)
    if current.contract != proof.contract:
        raise ValueError('First-pass native bypass execution inputs changed')
    return current.contract, current.hooks


def draft_stage_model_identity(model, *, source=None):
    """Keep global identity policies unchanged; only drafts use this adapter."""
    from .long_video_dual_identity import _audited_stage_model_identity, stage_model_identity
    if not model.injections:
        return stage_model_identity(model)
    from .taeh3_sampling_preview import cache_projection
    from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
    source = model if source is None else source
    projected = cache_projection(model)
    if projected is not model and type(projected) in (ModelPatcher, ModelPatcherDynamic):
        projected.is_injected = False
    try:
        proof = _capture(projected, source)
        return _audited_stage_model_identity(projected, _draft_bypass=proof)
    except UnverifiedModelStack:
        # Arbitrary user stacks continue to sample with the existing explicit
        # nonportable status. Real tensor corruption and sampler errors escape.
        return stage_model_identity(model)
