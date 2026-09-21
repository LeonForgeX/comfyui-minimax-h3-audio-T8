"""Versioned Director sampling settings and the exact two-pass canvas plan."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


DEFAULT_TWO_PASS = {
    "mode": "two_pass",
    "preset": "standard_4plus4_v1",
    "output_mp": "auto",
    "upscaler": "auto",
    "low_loras": [],
    "high_loras": [],
}


def _lora_rows(value: Any, stage: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{stage} LoRA 必须是列表")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{stage} 第 {index + 1} 条 LoRA 无效")
        name = row.get("name")
        enabled = row.get("enabled", True)
        strength = row.get("strength", 1.0)
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise ValueError(f"{stage} LoRA 需要文件名与启用状态")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError(f"{stage} LoRA 强度必须是 -2–2 的有限数字")
        if enabled and not name:
            raise ValueError(f"{stage} 已启用 LoRA 必须选择文件")
        result.append({
            "id": str(row.get("id") or f"{stage}-{index}"),
            "name": name,
            "strength": float(strength),
            "enabled": enabled,
            "source": str(row.get("source") or "user"),
        })
    if len({row["id"] for row in result}) != len(result):
        raise ValueError(f"{stage} LoRA 行 ID 重复")
    return result


def normalize_sampling(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"mode": "single"}
    if not isinstance(raw, Mapping):
        raise ValueError("采样设置必须是对象")
    mode = raw.get("mode", "single")
    if mode == "single":
        result = {"mode": "single"}
        if "loras" in raw:
            result["loras"] = _lora_rows(raw["loras"], "单采")
            result["lora_mode"] = str(raw.get("lora_mode", "manual"))
            if result["lora_mode"] not in {"auto", "none", "manual"}:
                raise ValueError("单采 LoRA 模式无效")
        if "resolution_mp" in raw:
            resolution = raw["resolution_mp"]
            if resolution != "auto":
                try:
                    resolution = float(resolution)
                except (TypeError, ValueError) as error:
                    raise ValueError("单采最终 MP 无效") from error
                if not math.isfinite(resolution) or not 0.2 <= resolution <= 2.1:
                    raise ValueError("单采最终 MP 必须在 0.2–2.1 范围内")
            result["resolution_mp"] = resolution
        return result
    if mode != "two_pass":
        raise ValueError("采样方式必须是单采或双采")
    preset = raw.get("preset", DEFAULT_TWO_PASS["preset"])
    if preset != "standard_4plus4_v1":
        raise ValueError(f"不支持的双采预设：{preset}")
    target = raw.get("output_mp", "auto")
    if target != "auto":
        if isinstance(target, bool) or not isinstance(target, (int, float, str)):
            raise ValueError("双采最终 MP 无效")
        try:
            target = float(target)
        except ValueError as error:
            raise ValueError("双采最终 MP 无效") from error
        if not math.isfinite(target) or not 0.2 <= target <= 2.1:
            raise ValueError("双采最终 MP 必须在 0.2–2.1 范围内")
    upscaler = raw.get("upscaler", "auto")
    if not isinstance(upscaler, str):
        raise ValueError("3D 放大模型必须是文件名")
    return {
        "mode": "two_pass",
        "preset": preset,
        "output_mp": target,
        "upscaler": upscaler,
        "low_loras": _lora_rows(raw.get("low_loras", []), "一采"),
        "high_loras": _lora_rows(raw.get("high_loras", []), "二采"),
    }


def effective_sampling(doc: Mapping[str, Any], shot: Mapping[str, Any]) -> dict[str, Any]:
    inherited = shot.get("samplingInherit", True)
    if not isinstance(inherited, bool):
        raise ValueError("镜头采样继承开关无效")
    source = doc.get("sampling") if inherited else shot.get("sampling")
    if source is None and not inherited:
        raise ValueError("本镜独立采样设置缺失")
    return normalize_sampling(source)


def two_pass_canvas(fraction: float, requested_mp: float | str) -> dict[str, Any]:
    from .director_project import director_canvas
    from .learned_latent_upscale_advanced import learned_upscale_geometry

    wanted_width, wanted_height = director_canvas(fraction, requested_mp)
    choices = []
    for dx in (-32, 0, 32):
        for dy in (-32, 0, 32):
            low_width = max(32, round((wanted_width / 2 + dx) / 32) * 32)
            low_height = max(32, round((wanted_height / 2 + dy) / 32) * 32)
            try:
                actual = learned_upscale_geometry(
                    source_latent_width=low_width // 16,
                    source_latent_height=low_height // 16,
                    size_mode="target_megapixels",
                    scale_by=2.0,
                    target_megapixels=wanted_width * wanted_height / 1_000_000,
                    target_width=wanted_width,
                    target_height=wanted_height,
                    aspect_policy="preserve_source",
                    max_anisotropy=1.05,
                )
            except ValueError:
                continue
            high_width, high_height = actual["output_width"], actual["output_height"]
            score = (
                abs(math.log((high_width / high_height) / fraction)),
                abs(high_width * high_height - wanted_width * wanted_height),
                abs(low_width * low_height - wanted_width * wanted_height / 4),
            )
            choices.append((score, low_width, low_height, actual))
    if not choices:
        raise ValueError("所选画幅／MP 无法得到合法双采尺寸，请调整最终总像素")
    _, low_width, low_height, actual = min(choices, key=lambda item: item[0])
    return {
        "low_width": low_width,
        "low_height": low_height,
        "width": actual["output_width"],
        "height": actual["output_height"],
        "actual_megapixels": actual["output_pixels"] / 1_000_000,
        "requested_megapixels": requested_mp,
        "scale_x": actual["scale_x"],
        "scale_y": actual["scale_y"],
        "memory_warning": actual["memory_warning"],
        "geometry_version": 1,
    }


def sampling_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(normalize_sampling(value))
