"""User-selected MODEL stacks are advisory, not an admission allowlist.

Do not catch sampler/kernel errors here. Unknown executable state cannot claim
portable cache identity: use a fresh execution nonce rather than reject its use.
"""
from __future__ import annotations

from functools import wraps
from contextvars import ContextVar
import hashlib
import json
import logging
import uuid

_audit_advisories = ContextVar("t8_patch_stack_audit_advisories", default=None)


class UnverifiedModelStack(ValueError):
    """Classifier cannot give a portable identity; not a sampling prohibition."""


def warn_patch_stack(message):
    advisories = _audit_advisories.get()
    if advisories is not None and message not in advisories:
        advisories.append(message)
    logging.warning(
        "[MiniMax H3 compatibility advisory / 组合风险自负] %s; continuing. "
        "This is not a compatibility or quality guarantee. Existing patches may "
        "bypass this node; real execution errors still propagate.", message,
    )


def advisory_audit(function):
    """Keep an incomplete composition audit from falsely claiming verification.

    No exception is caught: kernel failures, invalid clocks/inputs and damaged
    receipts still propagate normally.
    """
    @wraps(function)
    def audit(*args, **kwargs):
        advisories = []
        token = _audit_advisories.set(advisories)
        try:
            latent, serialized = function(*args, **kwargs)
            if advisories:
                report = json.loads(serialized)
                report.update(status="executed_user_stack_unverified",
                              compatibility_advisories=advisories,
                              composition_verified=False)
                serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
            return latent, serialized
        finally:
            _audit_advisories.reset(token)
    return audit


def advisory_inspection(function):
    """Only for optional composition classifiers, never receipt/hash validators."""
    @wraps(function)
    def inspect(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, RuntimeError) as error:
            warn_patch_stack(f"{function.__name__}: unverified composition: {error}")
            return None
    return inspect


def nonportable_model_identity(model, reason, *, schema):
    """Allow sampling without unsafe cross-run reuse of unaudited live state."""
    warn_patch_stack(f"{reason}; portable cache reuse disabled for this execution")
    selected = {name: _execution_selection(getattr(model, name, None)) for name in (
        "patches", "object_patches", "wrappers", "callbacks", "injections",
        "weight_wrapper_patches", "hook_patches", "forced_hooks", "current_hooks",
        "model_options", "additional_models",
    )}
    bypass, bypass_selection = _native_bypass_projection(model)
    bypass.update(_loaded_bypass_hooks(model))
    if bypass_selection:
        selected["native_bypass_selection"] = bypass_selection
    network = getattr(model, "model", None)
    if callable(getattr(network, "named_modules", None)):
        selected["network_execution"] = {
            name: _execution_selection({
                "forward": _configured_forward(module, bypass),
                "pre_hooks": getattr(module, "_forward_pre_hooks", {}),
                "hooks": getattr(module, "_forward_hooks", {}),
            }) for name, module in network.named_modules()
            if _configured_forward(module, bypass) is not None or getattr(module, "_forward_pre_hooks", {})
            or getattr(module, "_forward_hooks", {})
        }
    # Capture selected/live owners first: Core's state getter can eject/reinject
    # native hooks, which must not hide an unexpected forward replacement.
    state = model.model_state_dict() if hasattr(model, "model_state_dict") else model.model.state_dict()
    if not state:
        raise ValueError("MODEL has no loaded tensor state")
    # Validate actual tensor bytes too; opaque hooks are not a way around NaN,
    # missing tensors or malformed weight storage checks.
    from .long_video_dual_identity import content_identity, _original_state
    state_digest = hashlib.sha256(
        repr(content_identity(_original_state(model, state))).encode("utf-8")
    ).hexdigest()
    return {
        "schema": schema, "sha256": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
        "backend": {"kind": "user_selected_unverified"},
        "memory": None, "composition": {"kind": "user_selected_unverified"},
        "portable_cache_reuse": False, "model_filename_trusted": False,
        "execution_weight_sha256": state_digest,
        "execution_selection": selected,
        "opaque_internal_state_verified": False,
        "tensor_count": len(state), "lora_target_count": len(getattr(model, "patches", {})),
    }


def _configured_forward(module, bypass):
    """Project only authenticated Core bypass installation to its prior owner.

    Core clones share the network. Loading a sampled clone installs its bypass
    forwards there, and ejecting restores an explicit native bound method. Those
    lifecycle transitions do not replace the selected injection or its weights.
    """
    current = vars(module).get("forward")
    seen = set()
    while current is not None:
        owner = getattr(current, "__self__", None)
        hook = bypass.get(id(owner))
        if (hook is None or hook.module is not module or hook.original_forward is None
                or getattr(current, "__func__", None) is not type(hook)._bypass_forward):
            break
        if id(hook) in seen:
            raise ValueError("Native bypass forward ownership cycle")
        seen.add(id(hook))
        current = hook.original_forward
    if (getattr(current, "__self__", None) is module
            and getattr(current, "__func__", None) is type(module).forward):
        return None
    return current


def _bypass_weight_identity(value, dtype):
    """Hash native compute bytes after Core's declared dtype cast, in <=4MiB.

    Core moves/casts these adapter tensors when injecting. Validate source bytes
    too, but bind the bytes the same native injection will actually compute with.
    Never invoke an adapter, hook, arbitrary repr or quantized materialization.
    """
    import torch
    from .long_video_dual_identity import content_identity
    if isinstance(value, torch.Tensor) and dtype is not None and value.dtype != dtype:
        content_identity(value)
        flat = value.detach().reshape(-1)
        sha = hashlib.sha256()
        stride = max(1, 4 * 1024**2 // max(value.element_size(), torch.empty((), dtype=dtype).element_size()))
        for start in range(0, flat.numel(), stride):
            chunk = flat[start:start + stride].to(device="cpu", dtype=dtype)
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError("Bypass adapter compute tensor has nonfinite values")
            sha.update(chunk.view(torch.uint8).numpy().tobytes())
        return {"tensor_sha256": sha.hexdigest(), "dtype": str(dtype), "shape": list(value.shape)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [_bypass_weight_identity(item, dtype) for item in value]}
    return _execution_selection(value)


def _loaded_bypass_hooks(model):
    """Core's active sampled clone can own forwards on this shared network.

    Only exact native loaded entries and authenticated injection factories grant
    lifecycle projection. Their selection records never become this MODEL's
    contract: it still binds its own selected injections and adapter weights.
    """
    from comfy import model_management as mm
    hooks = {}
    for loaded in mm.current_loaded_models:
        if type(loaded) is not mm.LoadedModel:
            continue
        owner = loaded.model
        if owner is not None and owner.model is model.model and owner.is_injected:
            active, _ = _native_bypass_projection(owner, record_selection=False)
            hooks.update(active)
    return hooks


def native_bypass_hooks(model):
    """Read-only projection owners for selected and active shared Core clones."""
    hooks, _ = _native_bypass_projection(model, record_selection=False)
    hooks.update(_loaded_bypass_hooks(model))
    return hooks


def capture_native_bypass_forwards(model):
    """Bind installed native target forwards before the first real stage call."""
    hooks, _ = _native_bypass_projection(model, record_selection=False)
    modules = {id(hook.module): hook.module for hook in hooks.values()}
    return tuple((module, _execution_selection(vars(module).get("forward")))
                 for module in modules.values())


def verify_native_bypass_forwards(snapshot):
    """Check before Core eject can overwrite an unexpected new live owner."""
    if any(_execution_selection(vars(module).get("forward")) != expected
           for module, expected in snapshot):
        raise ValueError("First-pass draft execution inputs changed: native bypass forward owner")


def _native_bypass_projection(model, *, record_selection=True):
    """Recognize exact Core factories only; all stacks remain nonportable."""
    import types
    import torch
    try:
        from comfy.weight_adapter.bypass import BypassInjectionManager, BypassForwardHook
        from comfy.patcher_extension import PatcherInjection
    except ImportError:
        return {}, []
    codes = {code.co_name: code for code in BypassInjectionManager.create_injections.__code__.co_consts
             if isinstance(code, types.CodeType) and code.co_name in ("inject_all", "eject_all")}
    if set(codes) != {"inject_all", "eject_all"}:
        return {}, []
    modules = {id(module): name for name, module in model.model.named_modules()}
    hooks, records = {}, []
    for group, injections in getattr(model, "injections", {}).items():
        for injection in injections:
            if type(injection) is not PatcherInjection or set(vars(injection)) != {"inject", "eject"}:
                continue
            owners = []
            for role, name in (("inject", "inject_all"), ("eject", "eject_all")):
                function = getattr(injection, role)
                closure = getattr(function, "__closure__", None)
                if (getattr(function, "__code__", None) is not codes[name]
                        or function.__code__.co_freevars != ("self",) or not closure or len(closure) != 1):
                    break
                owners.append(closure[0].cell_contents)
            if (len(owners) != 2 or owners[0] is not owners[1]
                    or type(owners[0]) is not BypassInjectionManager):
                continue
            manager = owners[0]
            if set(vars(manager)) != {"adapters", "hooks"}:
                continue
            for hook in manager.hooks:
                if (type(hook) is not BypassForwardHook
                        or set(vars(hook)) != {"module", "adapter", "multiplier", "original_forward"}
                        or id(hook.module) not in modules):
                    continue
                hooks[id(hook)] = hook
                if not record_selection:
                    continue
                dtype = getattr(getattr(hook.module, "weight", None), "dtype", None)
                if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    dtype = None
                adapter = hook.adapter
                records.append(dict(group=_execution_selection(group), manager=id(manager), hook=id(hook),
                    target=modules[id(hook.module)], adapter=id(adapter), multiplier=_execution_selection(hook.multiplier),
                    configuration=_execution_selection({key: value for key, value in vars(adapter).items()
                                                       if key != "weights"}),
                    weights=_bypass_weight_identity(getattr(adapter, "weights", None), dtype)))
    return hooks, records


def nonportable_component_identity(component, reason, *, schema):
    result = nonportable_model_identity(component.patcher, reason, schema=schema)
    result["execution_selection"]["component"] = _execution_selection({
        "tokenizer": getattr(component, "tokenizer", None),
        "tokenizer_options": getattr(component, "tokenizer_options", None),
        "use_clip_schedule": getattr(component, "use_clip_schedule", None),
        "apply_hooks_to_conds": getattr(component, "apply_hooks_to_conds", None),
        "methods": {name: getattr(component, name, None) for name in (
            "tokenize", "encode_from_tokens", "encode_from_tokens_scheduled",
            "encode", "decode", "encode_tiled", "decode_tiled")},
    })
    return result


def _execution_selection(value):
    """Process-local selection snapshot, never a portable callable identity.

    Opaque callable internals are deliberately not certified. Detect replacement
    and weight/LoRA mutation without running hooks, repr or dequantization.
    """
    from .long_video_dual_identity import content_identity
    if isinstance(value, dict):
        return [( _execution_selection(key), _execution_selection(item))
                for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [_execution_selection(item) for item in value]
    try:
        return content_identity(value)
    except UnverifiedModelStack:
        function = getattr(value, "__func__", value)
        owner = getattr(value, "__self__", None)
        result = {"process_object": id(function), "bound_owner": id(owner)}
        try:
            from comfy.weight_adapter.lora import LoRAAdapter
        except ImportError:
            LoRAAdapter = None
        if LoRAAdapter is not None and isinstance(value, LoRAAdapter):
            result["adapter_weights"] = _execution_selection(value.weights)
        return result


def model_identity_matches(expected, current):
    """Within-run check only: nonce remains in persisted cache keys."""
    if isinstance(expected, dict) and isinstance(current, dict):
        if expected.keys() != current.keys():
            return False
        ignored = {"sha256"} if (expected.get("portable_cache_reuse") is False
                                 and current.get("portable_cache_reuse") is False) else set()
        return all(model_identity_matches(expected[key], current[key])
                   for key in expected.keys() - ignored)
    if isinstance(expected, (list, tuple)) and isinstance(current, (list, tuple)):
        return len(expected) == len(current) and all(
            model_identity_matches(a, b) for a, b in zip(expected, current))
    return expected == current


def compose_dit_hook(previous, current, owner):
    """Preserve the foreign owner, inserting our route only when it delegates."""
    if previous is None:
        return current
    if not callable(previous):
        raise TypeError(f"{owner}: existing DiT hook must be callable")
    warn_patch_stack(f"{owner}: retaining foreign DiT owner; a non-delegating hook may bypass this node")

    def composed(args, extra):
        return previous(args, {**extra, "original_block": lambda local: current(local, extra)})
    return composed


def slice_attention_mask(mask, start, end):
    if mask is None or mask.ndim < 2 or mask.shape[-2] == 1:
        return mask
    return mask[..., start:end, :]


def merge_attention_bias(bias, mask):
    if mask is None:
        return bias
    import torch
    return bias.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else bias + mask
