"""Strict loader for the original MiniMax-H3 HyperFlow two-time adapter.

The format and AV grid are specified by Video-Rebirth/hyperflow (Apache-2.0),
revision 1dd2f342aba5ab51da02b62885939655e8e268da. This implementation
accepts the *original* diffusers-layout artifact, without a pruned fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE_REVISION = "7b61c8edb895aaf25b248f064e9726ffdcc7ec46"
ORIGINAL_FILENAME = "minimax_h3_hyperflow_8step_v1.0.safetensors"
DEFAULT_RAW_GRID = (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0)
_KEY = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<matrix>[AB])\.weight$")
_BLOCK = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class HyperFlowMetadata:
    version: str
    gate: float
    rank: int
    alpha: float
    raw_sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    base_model: str
    license: str
    raw: dict[str, str]


@dataclass
class HyperFlowWeights:
    path: Path
    source_sha256: str
    metadata: HyperFlowMetadata
    # Comfy target path -> A, B, effective alpha. QKV rank is 3x source rank.
    patches: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    endpoint: dict[str, tuple[torch.Tensor, torch.Tensor, float]]
    source_tensor_count: int
    fused_qkv_groups: int


def _positive_finite(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"HyperFlow {label} must be finite and positive")
    return number


def _grid(values: object) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != 9:
        raise ValueError("HyperFlow v1 requires exactly nine raw sigma points")
    grid = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in grid):
        raise ValueError("HyperFlow sigma grid contains a non-finite value")
    if grid[0] != 1.0 or grid[-1] != 0.0 or any(a <= b for a, b in zip(grid, grid[1:])):
        raise ValueError("HyperFlow sigma grid must decrease strictly from 1 to 0")
    if any(abs(observed - trained) > 1e-6 for observed, trained in zip(grid, DEFAULT_RAW_GRID)):
        raise ValueError("HyperFlow v1 adapter grid differs from the trained original 9-point schedule")
    return grid


def parse_metadata(raw: dict[str, str] | None) -> HyperFlowMetadata:
    data = dict(raw or {})
    required = (
        "hyperflow", "hyperflow_version", "hyperflow_gate", "lora_rank",
        "lora_alpha", "base_model", "hyperflow_sigmas",
        "hyperflow_video_shift", "hyperflow_audio_shift",
    )
    missing = [key for key in required if key not in data]
    if missing or data.get("hyperflow", "").lower() != "true":
        raise ValueError(f"Not a complete HyperFlow adapter: missing {missing or ['hyperflow=true']}")
    if data["hyperflow_version"] != "1.0":
        raise ValueError(f"Unsupported HyperFlow version: {data['hyperflow_version']}")
    rank = int(data["lora_rank"])
    if rank <= 0:
        raise ValueError("HyperFlow rank must be positive")
    gate = float(data["hyperflow_gate"])
    if not math.isfinite(gate) or not 0.0 <= gate <= 1.0:
        raise ValueError("HyperFlow gate must be finite in [0, 1]")
    video_shift = _positive_finite(data["hyperflow_video_shift"], "video shift")
    audio_shift = _positive_finite(data["hyperflow_audio_shift"], "audio shift")
    if video_shift != 12.0 or audio_shift != 3.0:
        raise ValueError("HyperFlow v1 was trained with video/audio shifts 12/3")
    return HyperFlowMetadata(
        version=data["hyperflow_version"], gate=gate, rank=rank,
        alpha=_positive_finite(data["lora_alpha"], "LoRA alpha"),
        raw_sigmas=_grid(json.loads(data["hyperflow_sigmas"])),
        video_shift=video_shift, audio_shift=audio_shift,
        base_model=data["base_model"], license=data.get("license", "unspecified"), raw=data,
    )


def _target(module: str) -> tuple[str, str | None]:
    if module.startswith("time_embedder.linear_"):
        n = module.removeprefix("time_embedder.linear_")
        if n in {"1", "2"}:
            return "time_embedder.proj_" + ("in" if n == "1" else "out"), None
    if module.startswith("endpoint_time_embedder.linear_"):
        n = module.removeprefix("endpoint_time_embedder.linear_")
        if n in {"1", "2"}:
            return "endpoint_time_embedder.proj_" + ("in" if n == "1" else "out"), None
    match = _BLOCK.fullmatch(module)
    if match is None:
        raise ValueError(f"Unknown HyperFlow module: {module}")
    family, index, tail = match.groups()
    if family == "transformer_blocks":
        if not 0 <= int(index) < 50:
            raise ValueError(f"Unexpected DiT block {index}")
        prefix = f"blocks.{index}"
    else:
        if not 0 <= int(index) < 2:
            raise ValueError(f"Unexpected token-refiner block {index}")
        prefix = f"token_refiner.blocks.{index}"
    if tail in {"attn.to_q", "attn.to_k", "attn.to_v"}:
        return prefix + ".attn.qkv_proj", tail[-1]
    direct = {
        "attn.to_out.0": ".attn.out_proj",
        "ff.net.0.proj": ".mlp.fc1",
        "ff.net.2": ".mlp.fc2",
    }
    if tail not in direct:
        raise ValueError(f"Unknown HyperFlow module: {module}")
    return prefix + direct[tail], None


def plan_source_keys(keys: list[str]) -> dict[str, dict[str, object]]:
    """All 632 source tensors must map once; no unknown/pruned key is accepted."""
    plan: dict[str, dict[str, object]] = {}
    for key in keys:
        match = _KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Non-HyperFlow source tensor: {key}")
        target, part = _target(match["module"])
        entry = plan.setdefault(target, {"parts": {}})
        if part is None:
            slot = entry
        else:
            slot = entry["parts"].setdefault(part, {})
        if match["matrix"] in slot:
            raise ValueError(f"Duplicate HyperFlow matrix: {key}")
        slot[match["matrix"]] = key
    if len(keys) != 632 or len(plan) != 212:
        raise ValueError(f"Incomplete HyperFlow v1 artifact: {len(keys)} tensors / {len(plan)} targets")
    for target, entry in plan.items():
        if target.endswith(".attn.qkv_proj"):
            parts = entry["parts"]
            if set(parts) != {"q", "k", "v"} or any(set(slot) != {"A", "B"} for slot in parts.values()):
                raise ValueError(f"Incomplete QKV group: {target}")
        elif {key for key in entry if key != "parts"} != {"A", "B"}:
            raise ValueError(f"Incomplete HyperFlow pair: {target}")
    required_time = {
        "time_embedder.proj_in", "time_embedder.proj_out",
        "endpoint_time_embedder.proj_in", "endpoint_time_embedder.proj_out",
    }
    if not required_time.issubset(plan):
        raise ValueError("HyperFlow endpoint or base time adapter is missing; pruned build is not supported")
    return plan


def _check_pair(a: torch.Tensor, b: torch.Tensor, rank: int, target: str) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
        raise ValueError(f"Invalid HyperFlow A/B shape for {target}: {tuple(a.shape)}, {tuple(b.shape)}")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError(f"Non-finite HyperFlow tensor at {target}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hyperflow_original(path: str | Path) -> HyperFlowWeights:
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise FileNotFoundError(f"HyperFlow safetensors file not found: {source}")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        meta = parse_metadata(handle.metadata())
        keys = list(handle.keys())
        source_plan = plan_source_keys(keys)
        patches: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        endpoint: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
        qkv_count = 0
        for target, entry in sorted(source_plan.items()):
            if target.endswith(".attn.qkv_proj"):
                parts = entry["parts"]
                factors = []
                for part in ("q", "k", "v"):
                    a = handle.get_tensor(parts[part]["A"])
                    b = handle.get_tensor(parts[part]["B"])
                    _check_pair(a, b, meta.rank, target + "." + part)
                    factors.append((a, b))
                widths_in = {a.shape[1] for a, _ in factors}
                widths_out = {b.shape[0] for _, b in factors}
                if len(widths_in) != 1 or len(widths_out) != 1:
                    raise ValueError(f"QKV dimensions disagree at {target}")
                a_fused = torch.cat([a for a, _ in factors], dim=0).contiguous()
                inner = factors[0][1].shape[0]
                b_fused = torch.zeros((3 * inner, 3 * meta.rank), dtype=factors[0][1].dtype)
                for i, (_, b) in enumerate(factors):
                    b_fused[i * inner:(i + 1) * inner, i * meta.rank:(i + 1) * meta.rank] = b
                patches[target] = (a_fused, b_fused.contiguous(), 3 * meta.alpha)
                qkv_count += 1
                continue
            a = handle.get_tensor(entry["A"])
            b = handle.get_tensor(entry["B"])
            _check_pair(a, b, meta.rank, target)
            if target.endswith(".mlp.fc1"):
                if b.shape[0] % 2:
                    raise ValueError(f"SwiGLU B rows must be even at {target}")
                value, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, value), dim=0).contiguous()
            if target.startswith("endpoint_time_embedder."):
                endpoint[target.removeprefix("endpoint_time_embedder.")] = (a, b, meta.alpha)
            else:
                patches[target] = (a, b, meta.alpha)
    if len(patches) != 210 or len(endpoint) != 2 or qkv_count != 52:
        raise ValueError("HyperFlow v1 conversion did not consume every required module")
    return HyperFlowWeights(
        path=source, source_sha256=_sha256(source), metadata=meta,
        patches=patches, endpoint=endpoint, source_tensor_count=len(keys),
        fused_qkv_groups=qkv_count,
    )
