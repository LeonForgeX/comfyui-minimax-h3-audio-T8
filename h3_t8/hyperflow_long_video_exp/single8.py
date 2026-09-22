"""Isolated full-eight-NFE HyperFlow long-film comparison runner.

Each delivery segment makes one native 0:8 pass at final resolution.  There is
no learned upscaler, fresh HIGH noise, or second MODEL branch in this route.
"""
from __future__ import annotations

import json
import time

from .. import long_video_dual_picture_context as picture_context
from ..execution_timing import WallTimings
from ..hyperflow_sampling_advanced import build_hyperflow_plan, setup_hyperflow_sampler
from ..long_video_delivery import _resolve_inside
from ..long_video_dual_model_runner import DualModelSegmentRunner
from ..long_video_dual_model_stages import sample_model_stage
from ..long_video_dual_residency import release_stage_residency
from ..long_video_dual_stage_cache import AVStageCache
from ..preview_execution_context import preview_scope


RECIPE = "hyperflow_long_video_native_single8_exp_v1"


class HyperFlowSingle8SegmentRunner(DualModelSegmentRunner):
    """Use reviewed conditioning/color helpers, never their dual-stage run."""

    def __init__(self, model, *, contract, width, height, color_match=True):
        super().__init__(
            model, model, contract=contract, low_width=width, low_height=height,
            upscaler_model=None, coarse_steps=4, refine_steps=4,
            prompt_relay_mode="disabled", query_chunk_rows=256,
            second_audio_source="auto", second_audio_strength=0.0,
            eav_config={"mode": "disabled"}, color_match=color_match,
            video_context_mode="reference_only",
            low_context_source=picture_context.NAME,
        )
        self.contract = contract

    def run(self, *, root, chain_id, job_sha256, segment, candidate_id,
            base_candidate_id, high_context, parent_candidate_id,
            parent_revision, projected_plan, inputs):
        if projected_plan is not None:
            raise ValueError("Native single8 long video does not accept Prompt Relay")
        if inputs["task_type"] != "T2VA" or inputs["audio_mode"] != "native":
            raise ValueError("Native single8 long video is T2VA/native only")
        if any(inputs.get(name) is not None for name in (
                "drive_audio", "final_audio", "first_frame", "last_frame",
                "ref_images", "ref_videos", "ref_video_audios", "ref_audios",
                "persistent_identity_image")):
            raise ValueError("Native single8 long video does not accept external conditioning")
        started = time.perf_counter()
        timings = WallTimings()
        stage_contract = {
            "schema": RECIPE, "job": job_sha256, "segment": segment.index,
            "parent": parent_candidate_id, "parent_revision": parent_revision,
            "seed": segment.seed, "interval": [0, 8],
            "source_sha256": self.contract["hyperflow"]["source_sha256"],
        }
        picture_media = None
        picture_source = None
        if segment.index > 0:
            # Check the accepted immediate predecessor even when the stage is cached.
            picture_media, picture_source = picture_context.accepted_source(
                root, parent_candidate_id, segment.index, chain_id)
            stage_contract["picture_context"] = picture_source
        cache_root = _resolve_inside(
            root, root / "hyperflow_single8_stages" /
            f"segment_{segment.index:05d}" / base_candidate_id)
        cache = AVStageCache(cache_root)
        # Reuse the cache's reviewed final-output kind, under our distinct
        # namespace and full-8 contract; never touch P7 or legacy receipts.
        cached = cache.load("high_output", stage_contract)
        context = high_context
        picture_report = None
        if cached is None and picture_source is not None:
            context, picture_report = timings.call(
                "accepted_picture_context", picture_context.prepare_context,
                high_context, picture_media, picture_source,
                inputs["video_vae"], inputs["width"], inputs["height"])
        conditioned_model, positive, latent, mux, prompt, condition_json, relay = timings.call(
            "conditioning", self._conditions, self.models[0], context, inputs, None)
        if relay.get("status") != "disabled":
            raise RuntimeError("Native single8 unexpectedly received a Relay route")
        condition_release = release_stage_residency(
            inputs["clip"], inputs["video_vae"], inputs["audio_vae"])
        if cached is None:
            plan = build_hyperflow_plan(conditioned_model, 0, 8)
            sampled_model, sampler, sigmas = setup_hyperflow_sampler(
                conditioned_model, latent, plan)
            with preview_scope(phase="single8", segment=segment.index,
                               global_offset=0, global_total=8):
                output, sample_report = timings.call(
                    "single8_sampling", sample_model_stage,
                    sampled_model, positive, latent, sampler=sampler, sigmas=sigmas,
                    seed=segment.seed, segment_index=segment.index,
                    output_kind="zero_sigma_output")
            sample_report.update(
                hyperflow_plan=plan.as_report(),
                conditioning=json.loads(condition_json),
                conditioning_residency_release=condition_release,
                sampling_residency_release=release_stage_residency(sampled_model),
                audio_delivery={"source": "native_single8_joint_av"},
            )
            if picture_report is not None:
                sample_report["accepted_picture_context"] = picture_report
            cache.save("high_output", stage_contract, output, sample_report)
        else:
            output, receipt = cached
            sample_report = receipt["report"]
        return {
            "sampled": output, "mux_audio": mux, "conditioned_prompt": prompt,
            "conditioning_report_json": condition_json, "relay_report": relay,
            "sampling_report": {
                "mode": RECIPE,
                "hyperflow": {
                    "source_sha256": self.contract["hyperflow"]["source_sha256"],
                    "interval": [0, 8], "nfe": 8,
                    "cache_namespace": "hyperflow_single8_stages",
                    "learned_upscaler": "not_used",
                    "human_quality": "unverified",
                },
                "single8": {
                    "pass": sample_report, "reused": cached is not None,
                    "picture_context_source": picture_context.NAME,
                    "segment_compute_seconds": time.perf_counter() - started,
                    "execution_timings": timings.report(),
                },
            },
        }
