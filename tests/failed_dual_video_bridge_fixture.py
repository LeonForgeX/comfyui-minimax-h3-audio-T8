"""Historical failed experiment for regression reproduction ONLY; never runtime."""
import math

import torch
from comfy.nested_tensor import NestedTensor

HIGH_VIDEO_LATENT_BRIDGE_VERSION = 1
HIGH_VIDEO_LATENT_BRIDGE_TOKENS = 5


def bridge_high_video_boundary(
    latent,
    context,
    *,
    chain_id,
    segment_index,
    context_frames,
    bridge_tokens=HIGH_VIDEO_LATENT_BRIDGE_TOKENS,
):
    """FAILED EXPERIMENT: retained only for controlled diagnostic reproduction.

    Never call this from the production segment runner. Human review rejected
    the ghosting it introduces; an endpoint metric is not visual acceptance.

    A hard 0→1 mask boundary can leave the first generated token structurally
    unrelated to the accepted predecessor even though every locked prefix token
    is exact.  Align the first free token to the accepted endpoint, then decay
    that full latent-space offset with a cosine window.  This spreads the scene
    transition over native H3 time without blending RGB frames or touching audio.
    """
    from h3_audio_t8_pkg.native_masked_context_advanced import _validated_context
    from h3_audio_t8_pkg.core import nested_av_parts
    from h3_audio_t8_pkg.long_video import CONTEXT_FRAME_STEPS, pixel_frames_from_latent_t

    if segment_index <= 0 or context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError('High video latent bridge requires a supported continuation context')
    bridge_tokens = int(bridge_tokens)
    if bridge_tokens < 2:
        raise ValueError('High video latent bridge needs at least two tokens')
    video, audio = nested_av_parts(latent)
    steps = CONTEXT_FRAME_STEPS[context_frames]
    available = int(video.shape[2]) - steps
    if available < bridge_tokens:
        raise ValueError('High video latent bridge does not fit in the generated region')
    tail = _validated_context(
        context,
        chain_id=chain_id,
        segment_index=segment_index,
        context_frames=context_frames,
        target_video=video,
    ).to(video)
    prefix_delta = float((video[:, :, :steps].float() - tail.float()).abs().max())
    if not math.isfinite(prefix_delta) or prefix_delta > 1e-5:
        raise ValueError('High video latent bridge requires the exact sampled locked prefix')

    output_video = video.clone()
    previous_endpoint = tail[:, :, -1]
    first_free_before = output_video[:, :, steps].clone()
    offset = previous_endpoint - first_free_before
    positions = torch.linspace(
        0.0,
        1.0,
        bridge_tokens,
        device=output_video.device,
        dtype=output_video.dtype,
    )
    weights = torch.cos(positions * torch.pi / 2).square()
    output_video[:, :, steps : steps + bridge_tokens] += (
        offset[:, :, None] * weights[None, None, :, None, None]
    )
    if not bool(torch.isfinite(output_video).all()):
        raise ValueError('High video latent bridge produced non-finite values')
    first_free_after = output_video[:, :, steps]
    result = {
        **latent,
        'samples': NestedTensor((output_video, audio)),
    }
    bridge_frames = pixel_frames_from_latent_t(steps + bridge_tokens) - context_frames
    return result, {
        'schema_version': HIGH_VIDEO_LATENT_BRIDGE_VERSION,
        'method': 'post_sampling_endpoint_align_cosine_decay',
        'source': 'accepted_completed_high_video_endpoint',
        'context_steps': steps,
        'bridge_tokens': bridge_tokens,
        'bridge_frames': bridge_frames,
        'weights': [float(value) for value in weights.detach().cpu()],
        'locked_prefix_max_absolute_delta': prefix_delta,
        'first_free_rms_before': float(
            (first_free_before.float() - previous_endpoint.float()).square().mean().sqrt()
        ),
        'first_free_rms_after': float(
            (first_free_after.float() - previous_endpoint.float()).square().mean().sqrt()
        ),
        'maximum_absolute_applied_offset': float(offset.float().abs().max()),
        'audio_touched': False,
        'rgb_frames_blended': False,
        'scope': 'post-sampling video latent only; human seam quality still requires review',
    }
