"""Public, independently prepared LOW/HIGH inputs for native H3 progressive AV.

The caller owns media/accepted-parent provenance. This boundary validates tensor
geometry and native conditioning without inventing a parent or resizing time.
"""
from __future__ import annotations

import math
import torch

from .patch_stack_policy import warn_patch_stack
from .progressive_sampling_contract import _plan_progressive


def _tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Prepared progressive {name} must be finite with shape {tuple(shape)}")


def validate_conditioning(value, video_shape, frames):
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("Prepared progressive requires one native conditioning entry")
    item = value[0]
    if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[1], dict):
        raise ValueError("Invalid prepared progressive conditioning")
    embedding, metadata = item
    if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or embedding.shape[0] != 1
            or not embedding.is_floating_point() or not bool(torch.isfinite(embedding).all())):
        raise ValueError("Prepared progressive embeddings must be finite batch-1 tensors")
    if metadata.get("t8_long_video_schema", 1) != 1:
        raise ValueError("Unsupported native long-video conditioning schema")
    tags = metadata.get("minimax_token_tags")
    if tags is not None and (not isinstance(tags, torch.Tensor) or tags.dtype != torch.long
            or tuple(tags.shape) not in {(embedding.shape[1],), (1, embedding.shape[1])}
            or not bool(((tags == 0) | (tags == 1)).all())):
        raise ValueError("Prepared progressive token tags do not match embeddings")
    keyframes = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, list):
        raise ValueError("Prepared progressive keyframes must be a list")
    if keyframes and metadata.get("minimax_frame_count") != frames:
        raise ValueError("Prepared progressive keyframe time differs from the AV time")
    for frame in keyframes:
        if not isinstance(frame, dict):
            raise ValueError("Invalid prepared progressive keyframe")
        index = frame.get("resolved_frame_index")
        if type(index) is not int or not 0 <= index < frames:
            raise ValueError("Prepared progressive keyframe index is outside the clip")
        motion = frame.get("t8_long_video_frame_index")
        if motion is not None and (isinstance(motion, bool) or not isinstance(motion, (int, float))
                or not math.isfinite(motion) or not 0 <= motion < frames or index != 0):
            raise ValueError("Invalid native motion-keyframe position")
        _tensor(frame.get("latent"), (1, 24, 1, *video_shape[-2:]), "keyframe latent")
    refs = metadata.get("minimax_refs", [])
    if not isinstance(refs, list):
        raise ValueError("Prepared progressive references must be a list")
    user_refs, motion_refs = [], []
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("Invalid prepared progressive reference")
        (motion_refs if "t8_long_video_audio_end_frame" in ref else user_refs).append(ref)
    from .progressive_references import describe_references
    if user_refs:
        describe_references(user_refs)
    if len(motion_refs) > 1:
        raise ValueError("Prepared progressive has multiple motion audio sources")
    for ref in motion_refs:
        end, length = ref["t8_long_video_audio_end_frame"], ref.get("ref_audio_t")
        if (ref.get("kind") != "audio" or type(length) is not int or length <= 0
                or isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(end) or not 0 < end < frames):
            raise ValueError("Invalid prepared progressive motion audio clock")
        _tensor(ref.get("audio_latent"), (1, 32, 2, length), "motion audio")
    known = {"pooled_output", "guidance", "minimax_keyframes", "minimax_frame_count",
             "minimax_token_tags", "minimax_refs", "t8_long_video_schema"}
    if set(metadata) - known:
        warn_patch_stack("Prepared progressive preserves additional conditioning/hooks; composition unverified")


def prepare_stage_inputs(high, low, positive, negative, positive_low, negative_low,
                         sigmas, *, low_evaluations, low_scale, task):
    from .core import nested_av_parts
    from .progressive_masking import normalize_av_masks
    video, audio = nested_av_parts(high)
    plan = _plan_progressive(video, audio, sigmas, low_evaluations=low_evaluations,
        low_scale=low_scale, task=task, noise_mask=None, initialized=True, prepared=True)
    low_video, low_audio = nested_av_parts(low)
    low_shape = (*video.shape[:-2], plan.low_height // 16, plan.low_width // 16)
    _tensor(low_video, low_shape, "LOW video")
    _tensor(low_audio, audio.shape, "LOW audio")
    frames = 5 + (video.shape[2] - 2) // 5 * 17
    for value, shape in ((positive, video.shape), (negative, video.shape),
                         (positive_low, low_shape), (negative_low, low_shape)):
        validate_conditioning(value, shape, frames)
    low_mask = normalize_av_masks(low.get("noise_mask"), low_video, low_audio)
    high_mask = normalize_av_masks(high.get("noise_mask"), video, audio)
    return plan, low_mask, high_mask
