"""Explicit MiniMax HyperVAE 2x loader; legacy VAE nodes remain unchanged."""

from __future__ import annotations

import json
from pathlib import Path

from comfy_api.latest import io


class MiniMaxH3HyperVAE2xLoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        names = folder_paths.get_filename_list("vae")
        return io.Schema(
            node_id="MiniMaxH3HyperVAE2xLoaderEXPT8",
            display_name="MiniMax H3 HyperVAE 2× 加载 (T8 EXP)",
            category="T8/MiniMax H3/VAE",
            is_experimental=True,
            description=(
                "加载 HyperVAE Krea2+MiniMax v2 2× 视频 VAE。编码仍为 H3 16×，解码为 32×；"
                "输出接原 video_vae，音频 VAE 不变。仅对显式选择的权重生效。"
            ),
            inputs=[
                io.Combo.Input(
                    "vae_name",
                    options=names or ["填写绝对路径"],
                    tooltip="直接列出 ComfyUI/models/vae 的文件；选择 HyperVAE 2× 权重。下方路径仅作旧图兼容覆盖。",
                ),
                io.String.Input(
                    "absolute_path",
                    default="",
                    tooltip="可选兼容项，通常留空；填写后优先于 vae_name。",
                ),
            ],
            outputs=[io.Vae.Output("video_vae"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, vae_name, absolute_path=""):
        import folder_paths

        from .hyper_vae_2x import load_hyper_vae_2x

        override = str(absolute_path).strip().strip('"')
        if override:
            path = Path(override)
            if not path.is_absolute():
                raise ValueError("HyperVAE absolute_path 必须是绝对路径")
        else:
            if vae_name == "填写绝对路径":
                raise ValueError("请填写 HyperVAE 2× 文件的绝对路径")
            path = Path(folder_paths.get_full_path_or_raise("vae", vae_name))
        vae, report = load_hyper_vae_2x(path)
        return io.NodeOutput(vae, json.dumps(report, ensure_ascii=False, indent=2))


HYPER_VAE_2X_NODE_CLASSES = [MiniMaxH3HyperVAE2xLoaderEXPT8]
