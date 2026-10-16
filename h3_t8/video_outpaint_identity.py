"""Bounded content identities for native stock H3 execution inputs."""
from __future__ import annotations

from .patch_stack_policy import UnverifiedModelStack

from enum import Enum
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

from .runtime_precision_identity import matmul_precision_identity
from .video_outpaint_plan import canonical
from .video_outpaint_model_patches import inspect_outpaint_model_patches_advisory, verify_instance_forward


_FLOAT8_NONFINITE_BITS = {
    # These four storage formats are exhaustively compared with Torch's
    # widened finite classification. FN has no infinity; FNUZ reserves 0x80.
    'torch.float8_e4m3fn': (0x7f, 0x7f),
    'torch.float8_e4m3fnuz': (0xff, 0x80),
    'torch.float8_e5m2': (0x7c, 0x7c),
    'torch.float8_e5m2fnuz': (0xff, 0x80),
}


def _update_tensor_digest(digest, chunk):
    """Check one fresh CPU chunk and hash its original bytes without a copy.

    NumPy avoids repeated Torch CPU thread-pool dispatch for these read-only
    checks. Half/bfloat finite classification uses their exact exponent bits,
    so it needs neither widening nor a change to the byte identity. The view
    and all scratch arrays stay bounded by the caller's <=4MiB chunk.
    """
    raw = chunk.view(torch.uint8).numpy()
    if chunk.dtype in (torch.float16, torch.bfloat16):
        exponent = 0x7c00 if chunk.dtype == torch.float16 else 0x7f80
        finite = not np.any((raw.view(np.uint16) & exponent) == exponent)
    elif chunk.dtype in (torch.float32, torch.float64):
        storage = np.float32 if chunk.dtype == torch.float32 else np.float64
        finite = bool(np.isfinite(raw.view(storage)).all())
    elif str(chunk.dtype) in _FLOAT8_NONFINITE_BITS:
        mask, nonfinite = _FLOAT8_NONFINITE_BITS[str(chunk.dtype)]
        finite = not np.any((raw & mask) == nonfinite)
    elif str(chunk.dtype).startswith('torch.float8_'):
        # A newly introduced format keeps the previous bounded widened check
        # until its storage encoding has its own exhaustive audited adapter.
        finite = bool(torch.isfinite(chunk.float()).all())
    else:
        # Preserve the original contract for other dtypes, including formats
        # which need a separate audited identity adapter.
        finite = not chunk.is_floating_point() or bool(torch.isfinite(chunk).all())
    if not finite:
        raise ValueError("execution tensor has nonfinite values")
    digest.update(memoryview(raw))


def value_identity(value, *, interrupt_check=None):
    """Hash tensor bytes in <=4MiB copies without repr/address-based identities."""
    if isinstance(value, torch.Tensor):
        if (value.device.type == "meta" or value.layout != torch.strided or not value.is_contiguous()
                or value.is_quantized):
            raise ValueError("execution tensor requires an unsupported materializing conversion")
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        stride = max(1, 4*1024*1024//value.element_size())
        for start in range(0, flat.numel(), stride):
            if interrupt_check:
                interrupt_check()
            chunk = flat[start:start+stride].cpu()
            _update_tensor_digest(digest, chunk)
        return {"tensor_sha256": digest.hexdigest(), "dtype": str(value.dtype), "shape": list(value.shape)}
    if value is None or isinstance(value, (str, bool, int, float)):
        canonical(value)
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": type(value).__name__, "value": str(value)}
    if isinstance(value, Enum):
        return {"enum": type(value).__module__+"."+type(value).__qualname__, "name": value.name}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [value_identity(v, interrupt_check=interrupt_check) for v in value]}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: value_identity(v, interrupt_check=interrupt_check) for k, v in sorted(value.items())}
    raise UnverifiedModelStack(f"execution value {type(value).__name__} has no portable identity adapter")


def native_stock_model_identity(model, *, interrupt_check=None):
    try:
        return _audited_native_stock_model_identity(model, interrupt_check=interrupt_check)
    except UnverifiedModelStack as error:
        from .patch_stack_policy import nonportable_model_identity
        return nonportable_model_identity(model, str(error), schema="t8.h3.outpaint/user_stack_v1")


def _audited_native_stock_model_identity(model, *, interrupt_check=None):
    """Actual MODEL contents, not its filename or a caller-supplied SHA label.

    Bare native MODEL and the pinned reference KJ memory pair are covered. LoRA,
    other attention wrappers and accelerated compositions need verified adapters;
    their implementation/acceptance remains in the complete integration plan.
    """
    from comfy.model_base import MiniMaxH3

    base = getattr(model, "model", None)
    if not isinstance(base, MiniMaxH3):
        raise ValueError("outpaint stock execution requires a native MiniMax H3 MODEL")
    from .video_outpaint_regional import REGIONAL_WRAPPER_KEY, regional_model_contract
    regional = regional_model_contract(model)
    if regional is not None and regional.get("portable_cache_reuse") is False:
        raise UnverifiedModelStack("Regional outpaint retains an unaudited attention delegate")
    for name in ("patches", "weight_wrapper_patches", "additional_models", "wrappers",
                 "callbacks", "injections", "hook_patches", "forced_hooks", "current_hooks"):
        if name == "wrappers" and regional is not None:
            continue
        if getattr(model, name, None):
            raise UnverifiedModelStack(f'stock outpaint identity does not yet cover {name}; use the pending composition adapter')
    attachments = {key: value for key, value in getattr(model, "attachments", {}).items() if value is not None}
    if regional is None:
        if attachments:
            raise UnverifiedModelStack('stock outpaint identity does not cover MODEL attachments')
    elif set(attachments) != {REGIONAL_WRAPPER_KEY}:
        raise UnverifiedModelStack('regional outpaint identity found unknown MODEL attachments')
    memory, allowed_forwards = inspect_outpaint_model_patches_advisory(model, regional_contract=regional)
    if memory.get("portable_cache_reuse") is False:
        raise UnverifiedModelStack("Unverified outpaint patch stack cannot reuse portable cache")
    composition = memory if regional is None else {
        "kind": "regional_outpaint_composition", "regional": regional, "memory": memory,
    }
    implementations = {}
    classes = []
    for name, module in base.named_modules():
        cls = type(module)
        source = inspect.getsourcefile(cls)
        if source is None:
            raise ValueError("native model implementation source is unavailable")
        if source not in implementations:
            implementations[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        classes.append((name, cls.__module__+"."+cls.__qualname__))
        try:
            verify_instance_forward(module, name, allowed_forwards)
        except ValueError as error:
            if "runtime forward replacement/hooks" not in str(error):
                raise
            raise UnverifiedModelStack(str(error)) from error
    state = base.state_dict()
    if not state:
        raise ValueError("native model has no loaded tensor state")
    config = {"unet_config": base.model_config.unet_config, "model_type": base.model_type,
              "manual_cast_dtype": base.manual_cast_dtype,
              "load_device": str(model.load_device), "force_cast_weights": model.force_cast_weights}
    data = {"schema": "t8.h3.outpaint.native_stock_model/v2", "classes": classes,
            "composition": composition,
            "runtime": {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "matmul_precision": matmul_precision_identity(),
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
                "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
                "sdpa_memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()},
            "implementations": sorted(implementations.values()),
            "config": value_identity(config, interrupt_check=interrupt_check),
            "state": value_identity(state, interrupt_check=interrupt_check),
            "provider_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {"sha256": hashlib.sha256(canonical(data).encode()).hexdigest(), "schema": data["schema"],
            "composition": composition,
            "model_filename_trusted": False, "max_copy_bytes": 4*1024*1024,
            "tensor_count": sum(isinstance(t, torch.Tensor) for t in state.values())}
