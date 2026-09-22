"""Opt-in native single-eight-step long-film comparison, separate from P7."""
from __future__ import annotations

import json
from pathlib import Path
import time

from comfy_api.latest import io

from ..hyperflow_runtime_advanced import ATTACHMENT_KEY, install_hyperflow
from ..hyperflow_weights_advanced import load_hyperflow_original
from ..long_video_delivery import _sha256_file
from ..long_video_dual_identity import content_identity, stage_model_identity
from ..long_video_in_node_loop_effects_advanced import run_long_video_in_node_loop_effects
from ..nodes_hyperflow_advanced import _resolve, _weight_options
from ..nodes_long_video_dual_model import _component_identity
from ..nodes_long_video_in_node_loop_effects_advanced import (
    MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced, _preview_video,
)
from .nodes import _OMIT
from .single8 import HyperFlowSingle8SegmentRunner, RECIPE


class MiniMaxH3HyperFlowSingle8LongVideoEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema()
        inherited = [item for item in original.inputs if item.id not in _OMIT]
        for item in inherited:
            if item.id == "chain_id":
                item.default = "h3_hyperflow_single8_long_video_exp"
            elif item.id == "total_duration_seconds":
                item.default = 8.0
            elif item.id == "width":
                item.default = 1024
            elif item.id == "height":
                item.default = 576
            elif item.id == "filename_prefix":
                item.default = "H3_HyperFlow_Single8_Long_Video_EXP"
        return io.Schema(
            node_id="MiniMaxH3HyperFlowSingle8LongVideoEXPT8",
            display_name="MiniMax H3 HyperFlow 原生8步长片 / Single8 (EXP/T8)",
            category=original.category, is_experimental=True, is_output_node=True,
            description=(
                "T2VA/native only. Each segment runs one complete trained HyperFlow "
                "0:8 AV trajectory at final dimensions. No dual pass or learned "
                "upscaler. Separate cache and chain_id; picture/audio seam needs review."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("hyperflow_file", options=_weight_options()),
                io.Boolean.Input("color_match", default=True),
                *inherited,
            ],
            outputs=original.outputs,
        )

    @classmethod
    def execute(cls, model, hyperflow_file, color_match=True, **kwargs):
        from comfy.cli_args import args
        import comfy.memory_management

        if not args.disable_comfy_compiler and comfy.memory_management.aimdo_enabled:
            raise ValueError(
                "HyperFlow long-video experiment requires an isolated Core started "
                "with --disable-comfy-compiler")
        allowed = {item.id for item in cls.define_schema().inputs}
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise ValueError(f"Native single8 rejects unsupported inputs: {sorted(unexpected)}")
        if model.get_attachment(ATTACHMENT_KEY) is not None:
            raise ValueError("Connect a bare content-LoRA MODEL before the dedicated HyperFlow loader")
        width, height = int(kwargs["width"]), int(kwargs["height"])
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Native single8 dimensions must be positive multiples of 32")
        if int(kwargs["render_window_frames"]) != 124 or int(kwargs["context_frames"]) != 22:
            raise ValueError("Single8 comparison requires window124/context22")
        if float(kwargs["total_duration_seconds"]) != 8.0:
            raise ValueError("Single8 comparison requires total eight seconds")
        started = time.perf_counter()
        model_identity = stage_model_identity(model)
        weights = load_hyperflow_original(_resolve(hyperflow_file))
        installed_model, binding, loader_report = install_hyperflow(model, weights)
        if binding.sha256 != weights.source_sha256:
            raise RuntimeError("Installed HyperFlow owner differs from original weight")
        package = Path(__file__).resolve().parents[1]
        implementation = {
            path.relative_to(package).as_posix(): _sha256_file(path)
            for path in (Path(__file__).resolve(), Path(__file__).with_name("single8.py"))
        }
        contract = {
            "schema": RECIPE,
            "settings": {
                "identity_version": 1, "color_match": bool(color_match),
                "audio_seam_policy": str(kwargs["audio_seam_policy"]),
                "bridge_ms": float(kwargs["bridge_ms"]),
                "bit_depth": int(kwargs["bit_depth"]),
                "crf": int(kwargs["crf"]),
                "filename_prefix": str(kwargs["filename_prefix"]),
            },
            "model": model_identity,
            "hyperflow": {
                "source_sha256": weights.source_sha256,
                "metadata": weights.metadata.raw,
                "loader": {k: v for k, v in loader_report.items() if k != "source_path"},
                "absolute_interval": [0, 8], "joint_native_audio": True,
            },
            "picture_context_source": "accepted_picture_low_context_v1",
            "stage_cache_namespace": "hyperflow_single8_stages",
            "media": content_identity({"external_media": None}),
            "components": {name: _component_identity(kwargs[name])
                           for name in ("clip", "video_vae", "audio_vae")},
            "implementation_sha256s": implementation,
            "legacy_runtime_source_sha256s": {
                path.name: _sha256_file(path)
                for path in sorted(package.glob("*.py"))
            },
        }
        runner = HyperFlowSingle8SegmentRunner(
            installed_model, contract=contract, width=width, height=height,
            color_match=bool(color_match))
        identity_seconds = time.perf_counter() - started
        video_path, manifest_path, completed, status, report = run_long_video_in_node_loop_effects(
            installed_model, kwargs["clip"], kwargs["video_vae"], kwargs["audio_vae"],
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
            width=width, height=height, task_type="T2VA", context_audio="video_and_audio",
            audio_mode="native", audio_denoise_strength=.35,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True, ref_image_size="match",
            reference_video_policy="official_2_to_15s", first_frame_reuse="segment0_only",
            persistent_identity_strategy="single_reference", persistent_identity_interval=1,
            resume_existing=kwargs["resume_existing"], filename_prefix=kwargs["filename_prefix"],
            audio_seam_policy=kwargs["audio_seam_policy"], bridge_ms=kwargs["bridge_ms"],
            bit_depth=kwargs["bit_depth"], crf=kwargs["crf"],
            model_id=model_identity["sha256"][:16] + ":hf:single8", _stage_runner=runner,
        )
        payload = json.loads(report)
        payload["hyperflow_identity_seconds"] = identity_seconds
        payload["hyperflow_recipe"] = RECIPE
        video, preview = _preview_video(video_path)
        return io.NodeOutput(
            video, video_path, manifest_path, completed, status,
            json.dumps(payload, ensure_ascii=False, allow_nan=False), ui=preview)
