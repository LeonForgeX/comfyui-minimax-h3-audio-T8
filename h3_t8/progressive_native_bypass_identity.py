"""First-pass-only identity for the exact Core inference LoRA bypass lifecycle.

This does not certify arbitrary injections, training adapters or callback stacks.
The selected patcher is never changed, and the public stage identity keeps its
original nonportable policy. Every boundary binds fresh adapter compute bytes.
"""
from __future__ import annotations

import ast
import dis
import hashlib
import inspect
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


def _symbol_codes(code, prefix=''):
    for child in code.co_consts:
        if isinstance(child, types.CodeType):
            symbol = getattr(child, 'co_qualname', prefix + child.co_name)
            yield symbol, child
            suffix = '.<locals>.' if child.co_flags & inspect.CO_NEWLOCALS else '.'
            yield from _symbol_codes(child, symbol + suffix)


def _class_metadata(cls, declaration, compiled):
    """Authenticate CPython 3.13's inert metadata, never arbitrary extra state.

    Match the current interpreter's class-body constants when available. The
    source-only fallback lets older interpreters inspect the same source-bound
    metadata without executing source or reconstructing a class.
    """
    generated = {}
    for code in _codes(compiled):
        if code.co_name != declaration.name or code.co_firstlineno != declaration.lineno:
            continue
        instructions = tuple(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if (instruction.opname == 'STORE_NAME'
                    and instruction.argval in ('__firstlineno__', '__static_attributes__') and index):
                previous = instructions[index - 1]
                if previous.opname == 'LOAD_CONST':
                    generated[instruction.argval] = previous.argval
    source_first = min([declaration.lineno] + [node.lineno for node in declaration.decorator_list])
    source_static = {node.attr for function in declaration.body
                     if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
                     for node in ast.walk(function)
                     if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                     and isinstance(node.value, ast.Name) and node.value.id == 'self'}
    members = vars(cls)
    names = {'__firstlineno__', '__static_attributes__'} & members.keys()
    if not names:
        if generated:
            raise UnverifiedModelStack('Native draft bypass compiler class metadata differs from Core source')
        return set()
    first, attributes = members.get('__firstlineno__'), members.get('__static_attributes__')
    valid = (names == {'__firstlineno__', '__static_attributes__'} and type(first) is int
             and first == generated.get('__firstlineno__', source_first)
             and type(attributes) is tuple and all(type(name) is str for name in attributes)
             and len(attributes) == len(set(attributes))
             and (attributes == generated['__static_attributes__'] if '__static_attributes__' in generated
                  else set(attributes) == source_static))
    if not valid:
        raise UnverifiedModelStack('Native draft bypass compiler class metadata differs from Core source')
    return names


def _source(module, functions):
    """Compare live code to its loaded source, without executing that source."""
    path = Path(module.__file__)
    payload = path.read_bytes()
    compiled = dict(_symbol_codes(compile(payload, str(path), 'exec', dont_inherit=True)))
    definitions = [node for node in ast.walk(ast.parse(payload)) if isinstance(node, ast.FunctionDef)]
    for symbol, function in functions:
        if (type(function) is not types.FunctionType or function.__globals__ is not vars(module)
                or function.__code__ != compiled.get(symbol)):
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
            'defaults': [content_identity((fn.__defaults__, fn.__kwdefaults__)) for _, fn in functions]}


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
        payload = Path(module.__file__).read_bytes()
        tree = ast.parse(payload)
        compiled = compile(payload, str(Path(module.__file__)), 'exec', dont_inherit=True)
        for cls in classes:
            declarations = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls.__name__]
            if len(declarations) != 1 or cls.__module__ != module.__name__:
                raise UnverifiedModelStack('Native draft bypass class differs from its Core owner')
            expected = {'__module__', '__doc__'}
            expected.update(_class_metadata(cls, declarations[0], compiled))
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
        (bypass, (('get_module_type_info', bypass.get_module_type_info),
                  ('BypassForwardHook.__init__', hook.__init__),
                  ('BypassForwardHook._bypass_forward', hook._bypass_forward),
                  ('BypassForwardHook.inject', hook.inject), ('BypassForwardHook.eject', hook.eject),
                  ('BypassForwardHook._move_adapter_weights_to_device', hook._move_adapter_weights_to_device),
                  ('BypassInjectionManager.__init__', manager.__init__),
                  ('BypassInjectionManager.add_adapter', manager.add_adapter),
                  ('BypassInjectionManager._get_module_by_key', manager._get_module_by_key),
                  ('BypassInjectionManager.create_injections', manager.create_injections))),
        (lora, (('LoRAAdapter.__init__', lora.LoRAAdapter.__init__), ('LoRAAdapter.h', lora.LoRAAdapter.h))),
        (base, (('WeightAdapterBase.g', base.WeightAdapterBase.g),
                ('WeightAdapterBase.bypass_forward', base.WeightAdapterBase.bypass_forward))),
        (patcher_extension, (('PatcherInjection.__init__', patcher_extension.PatcherInjection.__init__),)),
        (model_patcher, (('ModelPatcher.clone', model_patcher.ModelPatcher.clone),
                         ('ModelPatcher.get_clone_model_override', model_patcher.ModelPatcher.get_clone_model_override),
                         ('ModelPatcher.model_state_dict', model_patcher.ModelPatcher.model_state_dict),
                         ('ModelPatcher.use_ejected', model_patcher.ModelPatcher.use_ejected),
                         ('ModelPatcher.inject_model', model_patcher.ModelPatcher.inject_model),
                         ('ModelPatcher.eject_model', model_patcher.ModelPatcher.eject_model))),
        (mm, (('get_torch_device', mm.get_torch_device),)),
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


def _decline_code(error):
    """Fixed privacy-safe explanation codes; never export arbitrary error text."""
    text = str(error)
    if text == 'Native draft bypass compiler class metadata differs from Core source':
        return 'native_class_metadata_unverified'
    if text.startswith('Native draft bypass') and 'class contains unbound execution members' not in text:
        return 'native_core_source_unverified'
    codes = {
        'Native draft bypass class contains unbound execution members': 'native_core_class_state',
        'First-pass bypass needs an exact native Core patcher': 'native_core_patcher_owner',
        'First-pass bypass injection suppression is not a portable native execution': 'native_core_injection_suppressed',
        'First-pass bypass loaded state is not native': 'native_core_loaded_state',
        'First-pass bypass patcher method was replaced': 'native_core_patcher_methods',
        'First-pass bypass selection needs plain ordered Core groups': 'native_core_injection_groups',
        'First-pass bypass injection is not the exact Core factory': 'native_core_injection_factory',
        'First-pass bypass injection has a foreign execution owner': 'native_core_injection_owner',
        'First-pass bypass manager has unbound execution state': 'native_core_manager_state',
        'First-pass bypass target/strength differs from its Core factory': 'native_core_target_mapping',
        'First-pass bypass loaded flag has no installed native owner': 'native_core_hook_lifecycle',
        'First-pass bypass adapter is not the exact inference LoRA schema': 'native_core_lora_adapter_state',
        'First-pass bypass adapter configuration differs from its native target': 'native_core_target_configuration',
        'First-pass bypass inspected clone differs from its selected source': 'native_core_inspection_selection',
    }
    from .long_video_dual_identity import model_stack_reason
    return codes.get(text, model_stack_reason(error))


def draft_stage_model_identity(model, *, source=None):
    """Keep global identity policies unchanged; only drafts use this adapter."""
    from .long_video_dual_identity import _audited_stage_model_identity
    from .taeh3_sampling_preview import cache_projection
    from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
    source = model if source is None else source
    projected = cache_projection(model)
    if projected is not model and type(projected) in (ModelPatcher, ModelPatcherDynamic):
        projected.is_injected = False
    try:
        proof = _capture(projected, source) if projected.injections else None
        return _audited_stage_model_identity(projected, _draft_bypass=proof, _draft_user_stack=True)
    except UnverifiedModelStack as error:
        # Arbitrary user stacks continue to sample with the existing explicit
        # nonportable status. Real tensor corruption and sampler errors escape.
        from .patch_stack_policy import nonportable_model_identity
        # Preserve the actual refusal after native injection authentication.
        # Restarting the generic classifier would stop at injections and hide
        # the attachment/attention/wrapper that actually prevented saving LOW.
        return {**nonportable_model_identity(projected, str(error),
                    schema='t8.h3.dual_stage_model/user_stack_v1'),
                'invalid_reason_code': _decline_code(error)}
