"""Isolated HyperFlow partial-4+4 long-film stage runner.

The accepted legacy 4+4 runner and its ``dual_stages`` cache are intentionally
untouched.  Only the durable delivery loop and reviewed picture-context helper
are shared; all new stage receipts live in ``hyperflow_stages``.
"""
from __future__ import annotations

import json
import time

from comfy.nested_tensor import NestedTensor

from .. import long_video_dual_picture_context as picture_context
from ..execution_timing import WallTimings
from ..hyperflow_sampling_advanced import build_hyperflow_plan, setup_hyperflow_sampler
from ..learned_latent_upscale_advanced import learned_upscale_h3_av_latent
from ..long_video_delivery import _resolve_inside, _write_context_candidate
from ..long_video_dual_model_runner import DualModelSegmentRunner
from ..long_video_dual_model_stages import sample_model_stage
from ..long_video_dual_residency import release_stage_residency
from ..long_video_dual_stage_cache import AVStageCache
from ..preview_execution_context import preview_scope


RECIPE = "hyperflow_long_video_partial4plus4_exp_v1"


class HyperFlowLongVideoSegmentRunner(DualModelSegmentRunner):
    """Private adapter for the old delivery-loop stage-runner protocol.

    We inherit only pure conditioning/reconciliation/context helpers.  The
    legacy ``run`` and native-flow ``_stage_sampling`` are never called.
    """

    def __init__(self, model_low, model_high, *, contract, low_width, low_height,
                 upscaler_model, color_match=True):
        super().__init__(
            model_low, model_high, contract=contract,
            low_width=low_width, low_height=low_height,
            upscaler_model=upscaler_model, coarse_steps=4, refine_steps=4,
            first_shift_video=12.0, first_shift_audio=3.0,
            second_shift_video=12.0, second_shift_audio=3.0,
            prompt_relay_mode="disabled", query_chunk_rows=256,
            second_audio_source="auto", second_audio_strength=0.0,
            eav_config={"mode": "disabled"}, color_match=color_match,
            video_context_mode="reference_only",
            low_context_source=picture_context.NAME,
        )
        self.contract = contract

    @staticmethod
    def _setup(model, latent, first):
        start, stop = (0, 4) if first else (4, 8)
        plan = build_hyperflow_plan(model, start, stop)
        patched, sampler, sigmas = setup_hyperflow_sampler(
            model, latent, plan,
            internal_continuation=True,
            new_noise_restart=not first,
        )
        return patched, sampler, sigmas, plan.as_report()

    def run(self, *, root, chain_id, job_sha256, segment, candidate_id,
            base_candidate_id, high_context, parent_candidate_id,
            parent_revision, projected_plan, inputs):
        if projected_plan is not None:
            raise ValueError("HyperFlow long video does not accept Prompt Relay")
        if inputs["task_type"] != "T2VA" or inputs["audio_mode"] != "native":
            raise ValueError("HyperFlow long video is T2VA/native only")
        if any(inputs.get(name) is not None for name in (
                "drive_audio", "final_audio", "first_frame", "last_frame",
                "ref_images", "ref_videos", "ref_video_audios", "ref_audios",
                "persistent_identity_image")):
            raise ValueError("HyperFlow long video does not accept external conditioning")
        if getattr(segment.plan, "context_frames", None) is None:
            raise ValueError("HyperFlow long-video segment has no context frame count")

        started = time.perf_counter()
        timings = WallTimings()
        low_context, low_parent_sha = self._low_context(
            root, chain_id, segment, high_context, parent_candidate_id, job_sha256)
        cache_root = _resolve_inside(
            root, root / "hyperflow_stages" / f"segment_{segment.index:05d}" / base_candidate_id)
        cache = AVStageCache(cache_root)
        stage_contract = {
            "schema": RECIPE,
            "job": job_sha256,
            "segment": segment.index,
            "parent_high": parent_candidate_id,
            "parent_revision": parent_revision,
            "parent_low_sha256": low_parent_sha,
            "seed_low": segment.seed,
            "seed_high": (segment.seed + 1) % (2**64),
            "audio_policy": "joint_low_x0_then_high4_native_av",
            "intervals": [[0, 4], [4, 8]],
            "source_sha256": self.contract["hyperflow"]["source_sha256"],
        }
        picture_media = None
        picture_source = None
        if segment.index > 0:
            # Always validate the selected accepted predecessor, even on a hit.
            picture_media, picture_source = picture_context.accepted_source(
                root, parent_candidate_id, segment.index, chain_id)
            stage_contract["low_picture_context"] = picture_source

        low_hit = cache.load("low_x0", stage_contract)
        if low_hit is None:
            picture_report = None
            if picture_source is not None:
                low_context, picture_report = timings.call(
                    "accepted_picture_context", picture_context.prepare_context,
                    low_context, picture_media, picture_source,
                    inputs["video_vae"], *self.low_size)
            low_inputs = {**inputs, "width": self.low_size[0], "height": self.low_size[1]}
            low_model, positive, low_latent, _mux, _prompt, condition_report, relay = timings.call(
                "first_conditioning", self._conditions,
                self.models[0], low_context, low_inputs, None)
            if relay.get("status") != "disabled":
                raise RuntimeError("HyperFlow LOW unexpectedly received a Relay route")
            condition_release = release_stage_residency(
                inputs["clip"], inputs["video_vae"], inputs["audio_vae"])
            low_model, sampler, sigmas, plan = self._setup(low_model, low_latent, True)
            with preview_scope(phase="low", segment=segment.index, global_offset=0, global_total=8):
                low_x0, low_report = timings.call(
                    "first_sampling", sample_model_stage, low_model, positive, low_latent,
                    sampler=sampler, sigmas=sigmas, seed=segment.seed,
                    segment_index=segment.index, output_kind="denoised_x0")
            low_report.update(
                hyperflow_plan=plan, conditioning=json.loads(condition_report),
                relay=relay, conditioning_residency_release=condition_release,
                sampling_residency_release=release_stage_residency(low_model),
                handoff="predicted_clean_x0_not_nonterminal_x_sigma",
            )
            if picture_report is not None:
                low_report["accepted_picture_context"] = picture_report
            low_receipt = cache.save("low_x0", stage_contract, low_x0, low_report)
        else:
            low_x0, low_receipt = low_hit
            low_report = low_receipt["report"]

        high_inputs = dict(inputs)
        high_model, positive, template, mux, prompt, condition_report, relay = timings.call(
            "second_conditioning", self._conditions,
            self.models[1], high_context, high_inputs, None)
        if relay.get("status") != "disabled":
            raise RuntimeError("HyperFlow HIGH unexpectedly received a Relay route")
        condition_release = release_stage_residency(
            inputs["clip"], inputs["video_vae"], inputs["audio_vae"])
        high_contract = {**stage_contract, "low_tensor_sha256": low_receipt["tensor_sha256"]}
        high_hit = cache.load("high_output", high_contract)
        if high_hit is None:
            prepared_hit = cache.load("high_input", high_contract)
            if prepared_hit is None:
                enlarged, width, height, upscale_json = timings.call(
                    "learned_upscale", learned_upscale_h3_av_latent, low_x0,
                    self.upscaler_model, "target_dimensions", 2.0, 1.0,
                    inputs["width"], inputs["height"], "honor_dimensions_exp",
                    1.05, "fp16", "offload_after")
                if (width, height) != (inputs["width"], inputs["height"]):
                    raise RuntimeError("HyperFlow learned-upscale dimensions changed")
                prepared, positive, reconcile_json = timings.call(
                    "reconcile", self._reconcile, enlarged, template, positive)
                prepare_report = {
                    "upscale": json.loads(upscale_json),
                    "reconcile": json.loads(reconcile_json),
                }
                cache.save("high_input", high_contract, prepared, prepare_report)
            else:
                prepared, prepare_receipt = prepared_hit
                prepare_report = prepare_receipt["report"]
                prepared, positive, _ = timings.call(
                    "cached_input_reconcile", self._reconcile,
                    prepared, template, positive)
            high_model, sampler, sigmas, plan = self._setup(high_model, prepared, False)
            with preview_scope(phase="high", segment=segment.index, global_offset=4, global_total=8):
                output, high_report = timings.call(
                    "second_sampling", sample_model_stage, high_model, positive, prepared,
                    sampler=sampler, sigmas=sigmas,
                    seed=(segment.seed + 1) % (2**64),
                    segment_index=segment.index, output_kind="zero_sigma_output")
            high_report.update(
                hyperflow_plan=plan,
                preparation=prepare_report,
                conditioning=json.loads(condition_report),
                audio_delivery={"source": "completed_second_pass_output",
                                "policy": "joint_native_av_not_frozen_low_audio"},
                conditioning_residency_release=condition_release,
                sampling_residency_release=release_stage_residency(high_model),
            )
            cache.save("high_output", high_contract, output, high_report)
        else:
            output, high_receipt = high_hit
            high_report = high_receipt["report"]

        context_record = None
        if segment.plan.save_context:
            low_video, low_audio = low_x0["samples"].unbind()
            final_audio = output["samples"].unbind()[1]
            if final_audio.shape != low_audio.shape or final_audio.dtype != low_audio.dtype:
                raise RuntimeError("HyperFlow completed audio cannot populate LOW context")
            continuation = {**low_x0, "samples": NestedTensor((low_video, final_audio))}
            low_path = _resolve_inside(
                root, root / "candidates" / f"segment_{segment.index:05d}"
                / candidate_id / "low.context.safetensors")
            context_record = _write_context_candidate(
                continuation, low_path, chain_id, segment.index,
                self.contract["first_model"]["sha256"], job_sha256)
            context_record.update(
                path=low_path.relative_to(root).as_posix(),
                audio_source="completed_second_pass_output",
                audio_policy_version=3)
        return {
            "sampled": output, "mux_audio": mux, "conditioned_prompt": prompt,
            "conditioning_report_json": condition_report, "relay_report": relay,
            "sampling_report": {
                "mode": RECIPE,
                "hyperflow": {
                    "source_sha256": self.contract["hyperflow"]["source_sha256"],
                    "intervals": [[0, 4], [4, 8]],
                    "nfe": 8, "cache_namespace": "hyperflow_stages",
                    "human_quality": "unverified",
                },
                # Existing immutable context reader uses this documented
                # long-video report slot; it is not a legacy cache namespace.
                "dual_model": {
                    "low_context": context_record,
                    "first_pass": low_report, "second_pass": high_report,
                    "low_reused": low_hit is not None,
                    "high_reused": high_hit is not None,
                    "low_context_source": picture_context.NAME,
                    "audio_policy": "joint_native_av_not_frozen_low_audio",
                    "segment_compute_seconds": time.perf_counter() - started,
                    "execution_timings": timings.report(),
                },
            },
        }
