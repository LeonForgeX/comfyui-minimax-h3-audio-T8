"""Experimental MiniMax H3 five-view contact-sheet conditioning and decoding.

The five latent positions are independent image slots, not video frames.  This
protocol follows the author's H3 Contact Sheet graph; it deliberately stays
separate from the regular Still and video generation paths.
"""

from __future__ import annotations

import json
import math

import torch

import comfy.model_management
import comfy.nested_tensor
import node_helpers

from .core import nested_av_parts, resize_image
from .prompt_tags import prepare_prompt
from .still_image import assert_still_layout_contract


FIVE_VIEW_PROTOCOL = "t8_h3_five_view_v1"
FIVE_VIEW_SLOTS = 5
FIVE_VIEW_AUDIO_T = 28
MIN_SIZE = 512
MAX_SIZE = 2048
SIZE_MULTIPLE = 32


def _validate_size(size: int) -> int:
    size = int(size)
    if size < MIN_SIZE or size > MAX_SIZE or size % SIZE_MULTIPLE:
        raise ValueError("five-view size must be a multiple of 32 from 512 to 2048")
    return size


def _validate_reference(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("five-view reference must be an IMAGE tensor [1,H,W,C]")
    if image.shape[0] != 1 or image.shape[1] < 1 or image.shape[2] < 1 or image.shape[-1] < 3:
        raise ValueError("five-view reference must contain exactly one RGB image")


def _resize_reference(image: torch.Tensor, size: int) -> tuple[torch.Tensor, int, int]:
    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, math.sqrt((size * size) / (height * width)))
    ref_width = max(SIZE_MULTIPLE, round(width * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    ref_height = max(SIZE_MULTIPLE, round(height * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return resize_image(image, ref_width, ref_height), ref_width, ref_height


def build_five_view_conditioning(clip, video_vae, prompt: str, ref_image, size: int):
    """Build the author's five-slot AV target with one MiniMax image reference."""
    size = _validate_size(size)
    _validate_reference(ref_image)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("five-view prompt must describe the desired subject and views")

    conditioned_prompt, prompt_warnings = prepare_prompt(
        prompt,
        {"pictures": 1, "videos": 0, "audios": 0},
        source_audio_ordinal=0,
        prompt_primary_audio_ordinal=0,
        strict=True,
    )
    if "<Picture 1>" not in conditioned_prompt:
        raise ValueError("five-view prompt must refer to the input as <Picture 1>")

    resized, ref_width, ref_height = _resize_reference(ref_image, size)
    encoded = video_vae.encode(resized)
    expected_ref_shape = (1, 24, 1, ref_height // 16, ref_width // 16)
    if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected_ref_shape:
        raise ValueError(
            "MiniMax H3 video VAE must encode the five-view reference as "
            f"{expected_ref_shape}; got {getattr(encoded, 'shape', None)}"
        )
    ref_blocks = [{
        "kind": "image",
        "latent_h": ref_height // 16,
        "latent_w": ref_width // 16,
        "latent": encoded,
    }]
    tokens = clip.tokenize(
        conditioned_prompt,
        minimax_ref_items=[{"type": "image", "data": resized}],
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    text_len = int(conditioning[0][0].shape[1])
    assert_still_layout_contract(
        text_len, FIVE_VIEW_SLOTS, size // 16, size // 16, FIVE_VIEW_AUDIO_T, ref_blocks
    )
    conditioning = node_helpers.conditioning_set_values(
        conditioning, {"minimax_refs": ref_blocks}
    )

    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        (1, 24, FIVE_VIEW_SLOTS, size // 16, size // 16), device=device
    )
    audio = torch.zeros((1, 32, 2, FIVE_VIEW_AUDIO_T), device=device)
    if audio.device != video.device:
        audio = audio.to(device)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "t8_five_view_protocol": FIVE_VIEW_PROTOCOL,
    }
    report = json.dumps({
        "protocol": FIVE_VIEW_PROTOCOL,
        "status": "experimental",
        "view_count": FIVE_VIEW_SLOTS,
        "view_size": size,
        "reference_size": [ref_width, ref_height],
        "video_latent_shape": list(video.shape),
        "audio_latent_shape": list(audio.shape),
        "required_model": "MiniMax H3 Ref2VA pruned with a five-view turnaround LoRA",
        "recommended_sampling": "stock res_multistep / simple, 28 steps, denoise 1.0",
        "warnings": [
            "Five positions are image slots, not ordinary video frames.",
            "Only use the five-view LoRA in this experimental graph; it degrades normal video motion.",
            "Review identity and angle consistency before reusing views as project references.",
            *prompt_warnings,
        ],
    }, ensure_ascii=False, indent=2)
    return conditioning, latent, conditioned_prompt, report


def decode_five_view_sheet(av_latent: dict, video_vae):
    """Decode each slot as its own legal 2-token / 5-frame VAE clip."""
    if not isinstance(av_latent, dict) or av_latent.get("t8_five_view_protocol") != FIVE_VIEW_PROTOCOL:
        raise ValueError("expected a five-view protocol latent, not ordinary video latent")
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or video.shape[2] != FIVE_VIEW_SLOTS:
        raise ValueError("five-view target requires exactly five 24-channel image slots")
    if video.shape[3] != video.shape[4] or video.shape[3] < MIN_SIZE // 16:
        raise ValueError("five-view target requires square image slots of at least 512 pixels")
    if audio.shape != (1, 32, 2, FIVE_VIEW_AUDIO_T):
        raise ValueError("five-view target audio latent shape changed")

    views: list[torch.Tensor] = []
    for index in range(FIVE_VIEW_SLOTS):
        slot = video[:, :, index:index + 1]
        decoded = video_vae.decode(torch.cat((slot, slot), dim=2))
        if not isinstance(decoded, torch.Tensor):
            raise ValueError(f"video VAE returned no IMAGE tensor for five-view slot {index + 1}")
        if decoded.ndim == 5:
            if decoded.shape[0] != 1 or decoded.shape[1] != 5:
                raise ValueError(f"video VAE changed the five-frame decode contract at slot {index + 1}")
            image = decoded[0, 0]
        elif decoded.ndim == 4:
            if decoded.shape[0] != 5:
                raise ValueError(f"video VAE changed the five-frame decode contract at slot {index + 1}")
            image = decoded[0]
        else:
            raise ValueError(f"video VAE returned invalid rank at five-view slot {index + 1}")
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"video VAE returned invalid RGB image at five-view slot {index + 1}")
        views.append(image)

    if any(view.shape != views[0].shape for view in views[1:]):
        raise ValueError("five-view VAE decoded inconsistent view sizes")
    batch = torch.stack(views)
    strip = torch.cat(views, dim=1).unsqueeze(0)
    report = json.dumps({
        "protocol": FIVE_VIEW_PROTOCOL,
        "views": FIVE_VIEW_SLOTS,
        "per_view_shape": list(batch.shape[1:]),
        "strip_shape": list(strip.shape),
        "selection": "first decoded frame from each independently duplicated latent slot",
    }, ensure_ascii=False, indent=2)
    return batch, strip, report
