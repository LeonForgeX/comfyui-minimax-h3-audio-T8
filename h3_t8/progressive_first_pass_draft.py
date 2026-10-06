"""Public immutable native LOW drafts; HIGH configuration is verified separately.

These are clean-video/noisy-audio sampler boundaries, never accepted AV segments.
Only an authenticated LOW execution can publish the atomic completion receipt.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
import time
import types
import uuid

from safetensors.torch import save_file

from .long_video_delivery import _atomic_write_json, _sha256_file
from .long_video_dual_identity import content_identity, stage_model_identity
from .patch_stack_policy import UnverifiedModelStack, model_identity_matches, warn_patch_stack
from .progressive_checkpoint import (
    ProgressiveCheckpointSession, canonical, digest, implementation_identity, native_model_identity,
)
from .progressive_continuation_runtime import _input_identity

CAPABILITY = 't8.progressive.first_pass_draft.v1'
FINGERPRINT = 't8.first_pass_fingerprint.v1'
MODES = ('disabled', 'save_or_reuse', 'reuse_only', 'save_new')


def scope_from_json(value):
    if not isinstance(value, str) or len(value.encode('utf-8')) > 262144:
        raise ValueError('First-pass draft scope must be bounded JSON text')
    def invalid(_):
        raise ValueError('Nonfinite first-pass draft scope')
    scope = json.loads(value, parse_constant=invalid)
    if type(scope) is not dict or not scope:
        raise ValueError('First-pass draft needs a nonempty caller-owned project/pack/parent scope')
    return json.loads(canonical(scope))


def _portable(value):
    if isinstance(value, dict):
        return value.get('portable_cache_reuse') is not False and all(_portable(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(_portable(x) for x in value)
    return True


def _inspection_model(model):
    """Authenticate only T8's exact native long-video payload owner on a clone."""
    from .long_video import patch_long_video_model
    from comfy.model_base import MiniMaxH3
    inspection = model.clone()
    payload = inspection.object_patches.get('extra_conds')
    if payload is not None:
        function = getattr(payload, '__func__', None)
        codes = [x for x in patch_long_video_model.__code__.co_consts
                 if isinstance(x, types.CodeType) and x.co_name == '_patched_extra_conds']
        closure = getattr(function, '__closure__', None)
        original = closure[0].cell_contents if closure and len(closure) == 1 else None
        if (len(codes) == 1 and getattr(function, '__code__', None) is codes[0]
                and getattr(payload, '__self__', None) is model.model
                and getattr(original, '__self__', None) is model.model
                and getattr(original, '__func__', None) is MiniMaxH3.extra_conds):
            inspection.object_patches.pop('extra_conds')
        # Unknown owners stay present so existing identity policy disables
        # portable reuse while retaining the user's selected sampling stack.
    return inspection


def low_model_identity(model, sampler=None):
    inspection = _inspection_model(model)
    if sampler is not None:
        return native_model_identity(inspection, sampler)
    from .sampling import MiniMaxH3FlowSampling
    from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow
    sampling = model.get_model_object('model_sampling')
    def unauthenticated(reason):
        return {'weights': stage_model_identity(inspection), 'portable_cache_reuse': False,
                'invalid_reason': reason}
    if type(sampling) is not MiniMaxH3FlowSampling:
        return unauthenticated('Unauthenticated legacy sampling coordinates')
    extra = set(vars(sampling)) - set(vars(MiniMaxH3FlowSampling(model.model.model_config)))
    if extra:
        return unauthenticated('Legacy sampling contains unbound execution state')
    for name, owner in (('calculate_input', CONST), ('calculate_denoised', CONST),
                        ('noise_scaling', CONST), ('inverse_noise_scaling', CONST),
                        ('timestep', ModelSamplingDiscreteFlow), ('sigma', ModelSamplingDiscreteFlow),
                        ('percent_to_sigma', ModelSamplingDiscreteFlow)):
        if getattr(getattr(sampling, name), '__func__', None) is not getattr(owner, name):
            return unauthenticated('Legacy sampling coordinate method was replaced: ' + name)
    inspection.object_patches.pop('model_sampling', None)
    return {'weights': stage_model_identity(inspection),
            'coordinate_schema': 't8.dual_clock_euler.native_flow.v1',
            'coordinates': content_identity({name: getattr(sampling, name, None) for name in
                ('shift', 'audio_scale', 'multiplier', 'noise_scale', 'sigmas')})}


def fingerprint_first_pass(model, positive, av_latent, sigmas, seed, scope_json):
    """No sampling or weight mutation; actual tensors and LOW weights are hashed."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('First-pass seed must be unsigned 64-bit')
    scope = scope_from_json(scope_json)
    identity = low_model_identity(model)
    portable = _portable(identity)
    reason = identity.get('invalid_reason')
    try:
        inputs = _input_identity({'positive': positive, 'av_latent': av_latent, 'sigmas': sigmas})
    except UnverifiedModelStack as error:
        portable, reason = False, str(error)
        inputs = {'portable_cache_reuse': False, 'execution_nonce': uuid.uuid4().hex}
        warn_patch_stack('First-pass conditioning has an opaque owner; portable draft reuse disabled')
    contract = dict(schema=FINGERPRINT, model=identity, inputs=inputs, seed=seed,
                    scope=scope, implementation=implementation_identity())
    return dict(schema=FINGERPRINT, fingerprint=digest(contract), contract=contract,
                portable_reuse=portable, invalid_reason=reason or
                (None if portable else 'LOW model or conditioning has unauthenticated execution owners'))


def _read_only_identity_model(model):
    """Only inert native identity getters can share a synchronous snapshot.

    Unknown getters, clone callbacks and state-dict hooks keep the original
    independent captures. No tensor hash, version or storage metadata is cached.
    """
    import inspect
    import torch
    from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic
    from comfy.ldm.modules.attention import ComfyAttention
    from comfy.weight_adapter.lora import LoRAAdapter
    from .patch_stack_policy import _native_bypass_projection

    if type(model) not in (ModelPatcher, ModelPatcherDynamic):
        return False
    for name in ('clone', 'get_model_object', 'get_attachment', 'model_state_dict', 'get_clone_model_override',
                 'model_size', 'is_dynamic', 'use_ejected', 'inject_model', 'eject_model', 'get_all_callbacks'):
        if getattr(getattr(model, name, None), '__func__', None) is not getattr(type(model), name):
            return False
    callbacks = model.callbacks
    if (type(callbacks) is not dict
            or any(type(group) is not dict or len(group) for group in callbacks.values())):
        return False
    if type(model.additional_models) is not dict or len(model.additional_models):
        return False
    if model.current_hooks is not None or model.forced_hooks is not None:
        return False
    if type(model.attachments) is not dict:
        return False
    if any(inspect.getattr_static(value, 'on_model_patcher_clone', None) is not None
           for value in getattr(model, 'attachments', {}).values()):
        return False
    pending, seen = [model.model], set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        if type(module).__getattribute__ is not object.__getattribute__:
            return False
        # Never run an unknown modules()/named_modules()/state_dict getter to
        # decide whether those getters have no execution side effects.
        if any(inspect.getattr_static(module, name, None) is not getattr(torch.nn.Module, name)
               for name in ('modules', 'named_modules', 'state_dict')):
            return False
        members = vars(module)
        save = inspect.getattr_static(module, '_save_to_state_dict', None)
        if save is not torch.nn.Module._save_to_state_dict:
            # Core's attention owner adds only inert JSON metadata to its state.
            if (type(module) is not ComfyAttention or save is not ComfyAttention._save_to_state_dict
                    or not _plain_json_value(members.get('config'))):
                return False
        if members.get('_state_dict_hooks') or members.get('_state_dict_pre_hooks'):
            return False
        for name in ('_parameters', '_buffers'):
            storage = members.get(name)
            if type(storage) is not dict or any(not _read_only_input_value(value) for value in storage.values()):
                return False
        children = members.get('_modules')
        if type(children) is not dict:
            return False
        pending.extend(child for child in children.values() if child is not None)
    injections = getattr(model, 'injections', {})
    if type(injections) is not dict or any(type(group) is not list for group in injections.values()):
        return False
    authenticated = set()
    hooks, _ = _native_bypass_projection(model, record_selection=False, authenticated_injections=authenticated)
    if any(id(injection) not in authenticated for group in injections.values() for injection in group):
        return False
    fields = {'loaded_keys', 'weights', 'multiplier', 'is_conv', 'conv_dim', 'kernel_size',
              'in_channels', 'out_channels', 'kw_dict'}
    for hook in hooks.values():
        adapter = hook.adapter
        if type(adapter) is not LoRAAdapter or set(vars(adapter)) != fields:
            return False
        if type(adapter.weights) not in (list, tuple):
            return False
        if any(not _read_only_input_value(value) for value in adapter.weights):
            return False
    return True


def _plain_json_value(value):
    if type(value) in (str, int, float, bool, type(None)):
        return True
    if type(value) in (list, tuple):
        return all(_plain_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json_value(item) for key, item in value.items())
    return False


def _read_only_input_value(value, seen=None):
    """Inspect plain input containers without invoking tensor/owner methods."""
    import inspect
    import torch
    from comfy.nested_tensor import NestedTensor
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        # Exact Tensor classes can still carry instance detach/to overrides.
        return not vars(value)
    if type(value) in (str, int, float, bool, type(None), torch.dtype, torch.device):
        return True
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen = seen | {id(value)}
    if type(value) is NestedTensor:
        members = vars(value)
        return (set(members) == {'tensors', 'is_nested'} and type(members['tensors']) is list
                and inspect.getattr_static(value, 'unbind', None) is NestedTensor.unbind
                and all(_read_only_input_value(part, seen) for part in members['tensors']))
    if type(value) in (list, tuple):
        return all(_read_only_input_value(part, seen) for part in value)
    if type(value) is dict:
        return all(type(key) is str and _read_only_input_value(part, seen) for key, part in value.items())
    return False


class ProgressiveFirstPassDraftSession(ProgressiveCheckpointSession):
    def __init__(self, root, scope_json, mode, draft_id=""):
        if mode not in MODES[1:]:
            raise ValueError('Unsupported first-pass draft mode')
        path = Path(root)
        if not path.is_absolute() or path.resolve() != path.absolute():
            raise ValueError('Draft directory must be an absolute caller-owned path without symlinks')
        super().__init__(path)
        if not isinstance(draft_id, str) or (draft_id and re.fullmatch(r'[0-9a-f]{64}', draft_id) is None):
            raise ValueError('Selected draft_id must be a LOW cache SHA256')
        self.scope, self.mode, self.selected_id = scope_from_json(scope_json), mode, draft_id
        self.cacheable = True

    def bind_low(self, low, high, sampler, plan, *, inputs, settings, continuation=None,
                 producers=None, relay=False):
        if (not self.active or self.contract is not None
                or self.owner != (os.getpid(), threading.get_ident())):
            raise RuntimeError('Bind one locked first-pass draft exactly once')
        self.models, self.sampler, self.inputs = (low,), sampler, inputs
        self.continuation, self.producers, self.relay_binding = continuation, producers, None
        self.plan, self.high = plan, high
        self.low_identity = low_model_identity(low, sampler)
        # One selected patcher has one complete snapshot at this boundary.
        # Different patchers keep independent scans even when Core shares their
        # network: their backups, LoRA and selected execution owners may differ.
        self.high_identity = (json.loads(canonical(self.low_identity)) if high is low and _read_only_identity_model(low)
                              else low_model_identity(high, sampler))
        self.cacheable = _portable(self.low_identity) and not relay
        try:
            inputs_identity = _input_identity(inputs)
        except UnverifiedModelStack:
            self.cacheable = False
            inputs_identity = {'portable_cache_reuse': False, 'execution_nonce': uuid.uuid4().hex}
        self.contract = json.loads(canonical(dict(schema=CAPABILITY, scope=self.scope,
            plan=plan.report(), model=self.low_identity, inputs=inputs_identity,
            settings=settings, implementation=implementation_identity())))
        self.identity = digest(self.contract)
        self.filename = 'low-boundary-' + self.identity
        if self.selected_id and self.selected_id != self.identity:
            self._reject_incompatible_selection()
        if not self.cacheable:
            warn_patch_stack('LOW draft owner is not portable; this run can sample but cannot reuse a saved draft')
            if self.mode == 'reuse_only':
                raise ValueError('Selected LOW stack has no portable first-pass draft identity')
        return self.identity

    def _reject_incompatible_selection(self):
        path = self.root / ('low-boundary-' + self.selected_id + '.json')
        if not path.exists():
            raise ValueError('Selected first-pass draft_id has no completed receipt in this draft directory')
        if path.is_symlink() or path.resolve().parent != self.root or path.stat().st_size > 16 * 1024**2:
            raise ValueError('Selected first-pass draft receipt path is invalid')
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            contract = record['contract']
            valid = (record.get('schema') == 1 and record.get('capability') == CAPABILITY
                     and record.get('cache_id') == self.selected_id and type(contract) is dict
                     and contract.get('schema') == CAPABILITY and digest(contract) == self.selected_id)
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            raise ValueError('Selected first-pass draft receipt is corrupt') from error
        if not valid:
            raise ValueError('Selected first-pass draft receipt is corrupt')
        changed = [name for name in ('model', 'inputs', 'scope', 'plan', 'settings', 'implementation')
                   if contract.get(name) != self.contract.get(name)]
        raise ValueError('Selected first-pass draft_id is incompatible; changed LOW contract fields: '
                         + ', '.join(changed or ['schema']))

    def verify(self):
        if not self.active or self.contract is None or self.owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('First-pass draft is not bound and exclusively locked')
        if digest(self.contract) != self.identity:
            raise ValueError('First-pass draft execution inputs changed')
        current_low = low_model_identity(self.models[0], self.sampler)
        if not model_identity_matches(self.low_identity, current_low):
            raise ValueError('First-pass draft execution inputs changed')
        current_high = (current_low if self.high is self.models[0] and _read_only_identity_model(self.high)
                        else low_model_identity(self.high, self.sampler))
        if (not model_identity_matches(self.high_identity, current_high)
                or implementation_identity() != self.contract['implementation']):
            raise ValueError('First-pass draft execution inputs changed')
        if self.cacheable and _input_identity(self.inputs) != self.contract['inputs']:
            raise ValueError('First-pass draft LOW conditioning or source changed')
        if self.continuation is not None:
            self.continuation.verify()

    def load_low(self):
        self.verify()
        return self._load_low_verified()

    def _bind_and_load_low(self, low, high, sampler, plan, **kwargs):
        """One synchronous initial boundary, with no caller mutation window.

        No tensor identity survives this operation for later validation. Public
        bind_low/load_low remain separate and load_low always verifies again.
        """
        self.bind_low(low, high, sampler, plan, **kwargs)
        if (self.continuation is None and _read_only_input_value(self.inputs)
                and _read_only_input_value(kwargs['settings'])
                and _read_only_identity_model(low) and _read_only_identity_model(high)):
            return self._load_low_verified()
        return self.load_low()

    def _load_low_verified(self):
        if not self.cacheable:
            return None
        path = self.root / (self.filename + '.json')
        if path.is_symlink() or (path.exists() and path.stat().st_size > 16 * 1024**2):
            raise ValueError('First-pass draft completion receipt path is invalid')
        if self.mode == 'save_new':
            if path.exists():
                raise FileExistsError('LOW draft is immutable; choose a new generation take or LOW seed, '
                                      'use a new owned directory, or select save_or_reuse')
            return None
        if (_read_only_input_value(self.inputs) and _read_only_input_value(self.contract['settings'])
                and _read_only_identity_model(self.models[0]) and _read_only_identity_model(self.high)):
            loaded = super()._load_low_verified_boundary()
        else:
            loaded = super().load_low()
        if loaded is None and self.mode == 'reuse_only':
            raise ValueError('No compatible saved first-pass draft; LOW will not be resampled')
        if loaded is not None:
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('capability') != CAPABILITY or record.get('cache_id') != self.identity
                    or (self.root / record['tensor_file']).is_symlink()
                    or (self.root / record['tensor_file']).stat().st_size != record.get('tensor_size_bytes')):
                raise ValueError('First-pass draft public receipt is corrupt')
        return loaded

    def save_low(self, tensors, report):
        self.verify()
        if not self.cacheable:
            return {'status': 'nonportable_not_saved'}
        self._validate_boundary(tensors)
        if report['callbacks'] != self.plan.low_evaluations or report['actual_forwards'] != self.plan.low_evaluations:
            raise ValueError('Cannot save an unfinished first-pass draft')
        path = self.root / (self.filename + '.json')
        if path.exists():
            raise FileExistsError('First-pass draft completion receipt is immutable')
        tensor_path = self.root / (self.filename + '-' + uuid.uuid4().hex + '.safetensors')
        save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}, str(tensor_path))
        with tensor_path.open('r+b') as handle:
            handle.flush()
            os.fsync(handle.fileno())
        self.verify()
        record = dict(schema=1, capability=CAPABILITY, cache_id=self.identity,
            contract=self.contract, tensor_file=tensor_path.name,
            tensor_sha256=_sha256_file(tensor_path), tensor_size_bytes=tensor_path.stat().st_size,
            low_report=report, report_sha256=digest(report), created_at=time.time())
        _atomic_write_json(path, record)
        return self.public_receipt()

    def public_receipt(self):
        path = self.root / (self.filename + '.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        return dict(schema=1, capability=CAPABILITY, cache_id=self.identity,
            receipt_path=str(path), tensor_sha256=record['tensor_sha256'],
            size_bytes=record['tensor_size_bytes'] + path.stat().st_size,
            compression='none', portable_reuse=True)
