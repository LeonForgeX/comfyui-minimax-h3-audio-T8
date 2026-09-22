"""One isolated HyperFlow long-video EXP output node.

Registered outside ``h3_t8/nodes.py`` so that the accepted legacy long-video
source-file cache identity remains byte-for-byte unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import folder_paths
from comfy_api.latest import io

from ..hyperflow_runtime_advanced import ATTACHMENT_KEY, install_hyperflow
from ..hyperflow_weights_advanced import load_hyperflow_original
from ..learned_latent_upscale_advanced import learned_upscale_geometry, PIXELS_PER_H3_LATENT
from ..long_video_delivery import _sha256_file
from ..long_video_dual_identity import content_identity, stage_model_identity
from ..long_video_in_node_loop_effects_advanced import run_long_video_in_node_loop_effects
from ..nodes_hyperflow_advanced import _resolve, _weight_options
from ..nodes_long_video_dual_model import _component_identity
from ..nodes_long_video_in_node_loop_effects_advanced import (
    MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced,
    _preview_video,
)
from .runner import HyperFlowLongVideoSegmentRunner, RECIPE


_OMIT = {
    "model", "model_id", "steps", "shift_video", "shift_audio", "sampler_name", "scheduler",
    "prompt_relay_mode", "query_chunk_rows", "prompt_relay_plan",
    "eav_mode", "eav_tau", "eav_start_video_progress", "eav_end_video_progress",
    "eav_max_workspace_mib", "eav_g_hard_limit", "long_video_sampling_plan",
    "task_type", "context_audio", "audio_mode", "audio_denoise_strength",
    "add_source_as_reference", "prompt_primary_audio_ordinal", "strict_prompt_tags",
    "ref_image_size", "reference_video_policy", "first_frame_reuse",
    "persistent_identity_strategy", "persistent_identity_interval",
    "drive_audio", "final_audio", "first_frame", "last_frame", "persistent_identity_image",
    "ref_images", "ref_videos", "ref_video_audios", "ref_audios", "source_motion",
    "semantic_bridge",
}


def require_safe_dual_branch_core():
    from comfy.cli_args import args
    import comfy.memory_management

    if not args.disable_comfy_compiler and comfy.memory_management.aimdo_enabled:
        raise ValueError(
            "HyperFlow dual-branch long video is unsafe with the active comfy-aimdo compiler. "
            "Use an isolated Core started with --disable-comfy-compiler."
        )


class MiniMaxH3HyperFlowLongVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
        inherited = [item for item in original.inputs if item.id not in _OMIT]
        for item in inherited:
            if item.id == "chain_id":
                item.default = "h3_hyperflow_long_video_exp"
            elif item.id == "total_duration_seconds":
                item.default = 8.0
            elif item.id == "width":
                item.default = 896
            elif item.id == "height":
                item.default = 448
            elif item.id == "filename_prefix":
                item.default = "H3_HyperFlow_Long_Video_EXP"
        return io.Schema(
            node_id="MiniMaxH3HyperFlowLongVideoEXPT8",
            display_name="MiniMax H3 HyperFlow 双采长片 / Long Video (EXP/T8)",
            category=original.category, is_experimental=True, is_output_node=True,
            description=(
                "Separate T2VA/native HyperFlow 0:4 predicted x0 → learned 3D → "
                "new-noise 4:8 long-video experiment. Two content-LoRA MODEL branches "
                "must come from one full H3 base; select the original HyperFlow file "
                "separately. Uses its own stage-cache namespace and a new chain_id. "
                "Only a total-eight-second/two-segment film is planned for qualification; "
                "visual/audio seam quality is not inherited from legacy 4+4."
            ),
            inputs=[
                io.Model.Input("model_pass1"), io.Model.Input("model_pass2"),
                io.Combo.Input("hyperflow_file", options=_weight_options()),
                io.Int.Input("low_width", default=448, min=32, max=16384, step=32),
                io.Int.Input("low_height", default=224, min=32, max=16384, step=32),
                io.Combo.Input("upscaler_model", options=folder_paths.get_filename_list("latent_upscale_models")),
                io.Boolean.Input("color_match", default=True),
                *inherited,
            ],
            outputs=original.outputs,
        )

    @classmethod
    def execute(cls, model_pass1, model_pass2, hyperflow_file,
                low_width, low_height, upscaler_model, color_match=True, **kwargs):
        require_safe_dual_branch_core()
        allowed = {item.id for item in cls.define_schema().inputs}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise ValueError(f"HyperFlow long-video EXP rejects unsupported inputs: {sorted(unexpected)}")
        if model_pass1.model is not model_pass2.model:
            raise ValueError("Both HyperFlow stages must clone the same full H3 base")
        if any(model.get_attachment(ATTACHMENT_KEY) is not None for model in (model_pass1, model_pass2)):
            raise ValueError("Supply content-LoRA MODEL branches before the dedicated HyperFlow Loader")
        target_width, target_height = int(kwargs["width"]), int(kwargs["height"])
        if (low_width < 32 or low_height < 32 or low_width % 32 or low_height % 32
                or target_width < 32 or target_height < 32
                or target_width % 32 or target_height % 32):
            raise ValueError("HyperFlow long-video LOW/HIGH dimensions must be positive multiples of 32")
        geometry = learned_upscale_geometry(
            low_width // PIXELS_PER_H3_LATENT, low_height // PIXELS_PER_H3_LATENT,
            "target_dimensions", 2.0, 1.0, target_width, target_height,
            "honor_dimensions_exp", 1.05,
        )
        if (geometry["output_width"], geometry["output_height"]) != (target_width, target_height):
            raise ValueError("Final dimensions are not representable by learned 3D upscale")
        if low_width >= target_width or low_height >= target_height:
            raise ValueError("HIGH dimensions must both exceed LOW dimensions")
        if int(kwargs["render_window_frames"]) != 124 or int(kwargs["context_frames"]) != 22:
            raise ValueError("First HyperFlow long-video EXP qualification requires window124/context22")
        if float(kwargs["total_duration_seconds"]) != 8.0:
            raise ValueError("First HyperFlow long-video EXP qualification requires total eight seconds")
        weight_path = _resolve(hyperflow_file)
        upscaler_path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", upscaler_model))
        started = time.perf_counter()
        first_identity = stage_model_identity(model_pass1)
        second_identity = stage_model_identity(model_pass2)
        weights = load_hyperflow_original(weight_path)
        low_model, low_binding, low_report = install_hyperflow(model_pass1, weights)
        high_model, high_binding, high_report = install_hyperflow(model_pass2, weights)
        if (low_binding.sha256 != high_binding.sha256
                or low_binding.raw_sigmas != high_binding.raw_sigmas
                or low_binding.model_identity != high_binding.model_identity):
            raise RuntimeError("HyperFlow stage owners do not share one original source and base")
        implementation = {
            path.relative_to(Path(__file__).resolve().parents[1]).as_posix(): _sha256_file(path)
            for path in (Path(__file__).resolve(),
                         Path(__file__).with_name("runner.py"),
                         Path(__file__).with_name("__init__.py"))
        }
        contract = {
            "schema": RECIPE,
            "settings": {
                "identity_version": 2,
                "low_width": int(low_width),
                "low_height": int(low_height),
                "color_match": bool(color_match),
                "audio_seam_policy": str(kwargs["audio_seam_policy"]),
                "bridge_ms": float(kwargs["bridge_ms"]),
                "bit_depth": int(kwargs["bit_depth"]),
                "crf": int(kwargs["crf"]),
                "filename_prefix": str(kwargs["filename_prefix"]),
            },
            "first_model": first_identity,
            "second_model": second_identity,
            "hyperflow": {
                "source_sha256": weights.source_sha256,
                "metadata": weights.metadata.raw,
                "low_loader": {k: v for k, v in low_report.items() if k != "source_path"},
                "high_loader": {k: v for k, v in high_report.items() if k != "source_path"},
                "absolute_intervals": [[0, 4], [4, 8]],
                "joint_native_audio": True,
            },
            "upscaler_sha256": _sha256_file(upscaler_path),
            "geometry": geometry,
            "low_context_source": "accepted_picture_low_context_v1",
            "stage_cache_namespace": "hyperflow_stages",
            "media": content_identity({"external_media": None}),
            "components": {name: _component_identity(kwargs[name]) for name in ("clip", "video_vae", "audio_vae")},
            "implementation_sha256s": implementation,
            "legacy_runtime_source_sha256s": {
                path.name: _sha256_file(path)
                for path in sorted(Path(__file__).resolve().parents[1].glob("*.py"))
            },
        }
        runner = HyperFlowLongVideoSegmentRunner(
            low_model, high_model, contract=contract,
            low_width=low_width, low_height=low_height,
            upscaler_model=upscaler_model, color_match=bool(color_match))
        identity_seconds = time.perf_counter() - started
        model_id = first_identity["sha256"][:16] + ":" + second_identity["sha256"][:16] + ":hf"
        video_path, manifest_path, completed, status, report = run_long_video_in_node_loop_effects(
            high_model, kwargs["clip"], kwargs["video_vae"], kwargs["audio_vae"],
            chain_id=kwargs["chain_id"], total_duration_seconds=kwargs["total_duration_seconds"],
            render_window_frames=kwargs["render_window_frames"], context_frames=kwargs["context_frames"],
            global_prompt=kwargs["global_prompt"], segment_prompts_json=kwargs["segment_prompts_json"],
            prompt_relay_mode="disabled", query_chunk_rows=256,
            eav_mode="disabled", eav_tau=4.0, eav_start_video_progress=.15,
            eav_end_video_progress=.90, eav_max_workspace_mib=32, eav_g_hard_limit=1.5,
            minimum_free_vram_mib=kwargs["minimum_free_vram_mib"],
            base_seed=kwargs["base_seed"], seed_policy=kwargs["seed_policy"],
            steps=8, shift_video=12.0, shift_audio=3.0,
            sampler_name="dual_clock_euler", scheduler="native_flow",
            width=target_width, height=target_height,
            task_type="T2VA", context_audio="video_and_audio",
            audio_mode="native", audio_denoise_strength=.35,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True, ref_image_size="match",
            reference_video_policy="official_2_to_15s", first_frame_reuse="segment0_only",
            persistent_identity_strategy="single_reference", persistent_identity_interval=1,
            resume_existing=kwargs["resume_existing"], filename_prefix=kwargs["filename_prefix"],
            audio_seam_policy=kwargs["audio_seam_policy"], bridge_ms=kwargs["bridge_ms"],
            bit_depth=kwargs["bit_depth"], crf=kwargs["crf"], model_id=model_id,
            _stage_runner=runner,
        )
        payload = json.loads(report)
        payload["hyperflow_identity_seconds"] = identity_seconds
        payload["hyperflow_recipe"] = RECIPE
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video, video_path, manifest_path, completed, status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False), ui=preview)
