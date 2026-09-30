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
        if not self.active or self.contract is not None:
            raise RuntimeError('Bind one locked first-pass draft exactly once')
        self.models, self.sampler, self.inputs = (low,), sampler, inputs
        self.continuation, self.producers, self.relay_binding = continuation, producers, None
        self.plan, self.high = plan, high
        self.low_identity = low_model_identity(low, sampler)
        self.high_identity = low_model_identity(high, sampler)
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
        if (digest(self.contract) != self.identity
                or not model_identity_matches(self.low_identity, low_model_identity(self.models[0], self.sampler))
                or not model_identity_matches(self.high_identity, low_model_identity(self.high, self.sampler))
                or implementation_identity() != self.contract['implementation']):
            raise ValueError('First-pass draft execution inputs changed')
        if self.cacheable and _input_identity(self.inputs) != self.contract['inputs']:
            raise ValueError('First-pass draft LOW conditioning or source changed')
        if self.continuation is not None:
            self.continuation.verify()

    def load_low(self):
        self.verify()
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
