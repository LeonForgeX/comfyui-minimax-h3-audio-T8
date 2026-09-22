"""Typed HyperFlow 8-interval AV plan and explicit dual-clock sampler.

The author grid is not the existing T8 native-flow or standard two-pass grid.
Intervals retain absolute indices, so 1+7/4+4/7+1 continuation can be checked
without silently initializing a new noise trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import comfy.samplers
import torch

from .core import nested_av_parts
from .hyperflow_runtime_advanced import (
    ATTACHMENT_KEY,
    HyperFlowBinding,
    HyperFlowStep,
    pop_step,
    push_step,
)
from .sampling import _make_sampling, model_uses_raw_audio_velocity, sample_minimax_h3_dual_clock_euler


def shifted_grid(raw: tuple[float, ...], shift: float) -> torch.Tensor:
    grid = torch.tensor(raw, dtype=torch.float32, device="cpu")
    return shift * grid / (1.0 + (shift - 1.0) * grid)


@dataclass(frozen=True)
class HyperFlowPlan:
    owner: str
    source_sha256: str
    raw_sigmas: tuple[float, ...]
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]
    start_interval: int
    stop_interval: int
    recipe: str = "hyperflow8_full_v1"

    @property
    def nfe(self) -> int:
        return self.stop_interval - self.start_interval

    @property
    def video_segment(self) -> torch.Tensor:
        return torch.tensor(
            self.video_sigmas[self.start_interval:self.stop_interval + 1], dtype=torch.float32
        )

    @property
    def audio_segment(self) -> torch.Tensor:
        return torch.tensor(
            self.audio_sigmas[self.start_interval:self.stop_interval + 1], dtype=torch.float32
        )

    def as_report(self) -> dict:
        return {
            "schema": "t8.minimax_h3.hyperflow.plan.v1",
            "recipe": self.recipe,
            "source_sha256": self.source_sha256,
            "raw_sigmas": list(self.raw_sigmas),
            "video_sigmas": list(self.video_sigmas),
            "audio_sigmas": list(self.audio_sigmas),
            "absolute_interval": [self.start_interval, self.stop_interval],
            "nfe": self.nfe,
            "note": "Segment plans carry x_sigma, not clean x0. Generic sampler restarts re-noise partial segments.",
        }


def build_hyperflow_plan(model, start_interval: int = 0, stop_interval: int = 8) -> HyperFlowPlan:
    binding = model.get_attachment(ATTACHMENT_KEY)
    if not isinstance(binding, HyperFlowBinding):
        raise ValueError("HyperFlow Plan requires a MODEL from the dedicated HyperFlow Loader")
    start = int(start_interval)
    stop = int(stop_interval)
    if not 0 <= start < stop <= 8:
        raise ValueError("HyperFlow absolute interval must satisfy 0 <= start < stop <= 8")
    video = shifted_grid(binding.raw_sigmas, binding.video_shift)
    audio = shifted_grid(binding.raw_sigmas, binding.audio_shift)
    if not (bool(torch.isfinite(video).all()) and bool(torch.isfinite(audio).all())):
        raise ValueError("HyperFlow shifted AV grid is non-finite")
    return HyperFlowPlan(
        owner=binding.owner, source_sha256=binding.sha256,
        raw_sigmas=binding.raw_sigmas,
        video_sigmas=tuple(float(x) for x in video),
        audio_sigmas=tuple(float(x) for x in audio),
        start_interval=start, stop_interval=stop,
        recipe="hyperflow8_full_v1" if (start, stop) == (0, 8) else "hyperflow8_continuation_exp_v1",
    )


def _match_plan(model, plan: HyperFlowPlan) -> HyperFlowBinding:
    binding = model.get_attachment(ATTACHMENT_KEY)
    if not isinstance(binding, HyperFlowBinding):
        raise ValueError("HyperFlow sampler MODEL has no dedicated loader attachment")
    if (binding.owner, binding.sha256) != (plan.owner, plan.source_sha256):
        raise ValueError("HyperFlow MODEL and PLAN belong to different adapter instances")
    if id(model.model) != binding.model_identity:
        raise ValueError("HyperFlow MODEL identity changed since adapter installation")
    return binding


class _StepProxy:
    """Pass the absolute interval to the owned model for exactly one forward."""

    def __init__(self, model, plan: HyperFlowPlan, *, skip_start_rebase: bool = False):
        self._model = model
        self._plan = plan
        self._count = 0
        if skip_start_rebase:
            # The caller supplied an already advanced x_sigma. Core's generic
            # partial-restart noise/latent terms must not rebase its audio clock.
            self.noise = None
            self.latent_image = None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, x, sigma, **extra_args):
        index = self._plan.start_interval + self._count
        if index >= self._plan.stop_interval:
            raise RuntimeError("HyperFlow model evaluated more than the typed plan's NFE")
        expected = self._plan.video_sigmas[index]
        actual = float(torch.as_tensor(sigma).flatten()[0])
        if not math.isfinite(actual) or abs(actual - expected) > 2e-6:
            raise ValueError("HyperFlow sampler sigma disagrees with the trained absolute grid")
        step = HyperFlowStep(
            owner=self._plan.owner, absolute_index=index,
            sigma_video=expected, sigma_video_next=self._plan.video_sigmas[index + 1],
        )
        token = push_step(step)
        try:
            return self._model(x, sigma, **extra_args)
        finally:
            pop_step(token)
            self._count += 1

    def assert_complete(self):
        if self._count != self._plan.nfe:
            raise RuntimeError(f"HyperFlow performed {self._count} forwards, expected {self._plan.nfe}")


def sample_hyperflow_continuous(
    model_callable,
    x_sigma: torch.Tensor,
    plan: HyperFlowPlan,
    *,
    video_values: int,
    packed_values: int,
    audio_velocity_is_raw: bool,
    extra_args=None,
    callback=None,
    disable=None,
    skip_start_rebase: bool = False,
):
    """Continue the exact x_sigma trajectory; do not add noise or treat x as x0."""
    proxy = _StepProxy(model_callable, plan, skip_start_rebase=skip_start_rebase)
    output = sample_minimax_h3_dual_clock_euler(
        proxy, x_sigma, plan.video_segment.to(x_sigma.device),
        extra_args=extra_args, callback=callback, disable=disable,
        video_values=video_values, packed_values=packed_values,
        shift_video=12.0, shift_audio=3.0,
        audio_velocity_is_raw=audio_velocity_is_raw,
    )
    proxy.assert_complete()
    return output


def setup_hyperflow_sampler(model, av_latent: dict, plan: HyperFlowPlan, *,
                            internal_continuation: bool = False,
                            new_noise_restart: bool = False,
                            x_sigma_override=None,
                            endpoint_capture=None):
    """Return MODEL/SAMPLER/SIGMAS for an ordinary full 8-NFE Comfy graph.

    Partial plans are private to the controlled continuous-state split or the
    explicit new-noise HIGH refinement route. The public full8 node never
    accepts a bare partial plan as an untyped generic restart.
    """
    binding = _match_plan(model, plan)
    if new_noise_restart and (not internal_continuation or plan.start_interval == 0):
        raise ValueError("HyperFlow new-noise restart needs a typed partial absolute plan")
    if new_noise_restart and x_sigma_override is not None:
        raise ValueError("HyperFlow new-noise restart cannot inject a continuous x_sigma override")
    if (x_sigma_override is not None or endpoint_capture is not None) and not internal_continuation:
        raise ValueError("HyperFlow state override/capture is private to typed continuous sampling")
    if x_sigma_override is not None and plan.start_interval == 0:
        raise ValueError("HyperFlow captured x_sigma is only valid for a nonzero absolute restart")
    if (plan.start_interval, plan.stop_interval) != (0, 8) and not internal_continuation:
        raise ValueError("Partial HyperFlow plans require continuous x_sigma sampling, not a generic restart")
    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError("HyperFlow sampler requires native packed MiniMax H3 video24/audio32x2 latents")
    patched = model.clone()
    original_sampling = model.get_model_object("model_sampling")
    patched.add_object_patch(
        "model_sampling",
        _make_sampling(model, original_sampling, binding.video_shift, binding.audio_shift, False),
    )
    options = patched.model_options.get("transformer_options", {}).copy()
    options["minimax_h3_sigma_shift_video"] = binding.video_shift
    options["minimax_h3_sigma_shift_audio"] = binding.audio_shift
    patched.model_options["transformer_options"] = options
    video_values = math.prod(video.shape[1:])
    packed_values = video_values + math.prod(audio.shape[1:])
    raw_audio = model_uses_raw_audio_velocity(model)

    def sampler_function(model_wrap, x, sigmas, extra_args=None, callback=None, disable=None):
        if sigmas.numel() != plan.nfe + 1 or not bool(torch.allclose(sigmas.cpu(), plan.video_segment, rtol=0.0, atol=1e-7)):
            raise ValueError("HyperFlow sampler received a grid different from its typed trained plan")
        if x_sigma_override is not None:
            state = x_sigma_override()
            if not isinstance(state, torch.Tensor) or state.shape != x.shape or not bool(torch.isfinite(state).all()):
                raise ValueError("HyperFlow continuation requires a finite, shape-matched x_sigma state")
            x = state.to(device=x.device, dtype=x.dtype, copy=True)
        output = sample_hyperflow_continuous(
            model_wrap, x, plan, video_values=video_values, packed_values=packed_values,
            audio_velocity_is_raw=raw_audio, extra_args=extra_args,
            callback=callback, disable=disable,
            skip_start_rebase=internal_continuation and plan.start_interval > 0 and not new_noise_restart,
        )
        if endpoint_capture is not None:
            endpoint_capture(output.detach().to(device="cpu", dtype=torch.float32, copy=True))
        return output

    sampler_function.__name__ = "sample_minimax_h3_hyperflow_two_time_euler"
    sampler = comfy.samplers.KSAMPLER(sampler_function)
    return patched, sampler, plan.video_segment
