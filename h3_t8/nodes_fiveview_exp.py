"""Dedicated five-view contact-sheet nodes; never replace the regular Still path."""

from __future__ import annotations

from comfy_api.latest import io

from .fiveview_sheet import build_five_view_conditioning, decode_five_view_sheet


FIVE_VIEW_CATEGORY = "T8/MiniMax H3/Still/Experimental"


class MiniMaxH3FiveViewConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiveViewConditioningEXPT8",
            display_name="MiniMax H3 Five-View Character Sheet (EXP/T8)",
            description=(
                "One reference to five independently decoded image slots. Requires a "
                "Ref2VA pruned base and the dedicated turnaround LoRA; not a video workflow."
            ),
            category=FIVE_VIEW_CATEGORY,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.String.Input(
                    "prompt",
                    default="The camera orbits the subject of <Picture 1> clockwise; "
                    "five coordinated views, consistent identity and clothing.",
                    multiline=True,
                    dynamic_prompts=True,
                ),
                io.Image.Input("ref_image", tooltip="Exactly one reference image, <Picture 1>."),
                io.Int.Input(
                    "size", default=512, min=512, max=2048, step=32,
                    tooltip="Square size of each of five images, not the total strip size.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="five_view_latent"),
                io.String.Output(display_name="conditioned_prompt"),
                io.String.Output(display_name="report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, video_vae, prompt, ref_image, size):
        return io.NodeOutput(*build_five_view_conditioning(
            clip, video_vae, prompt, ref_image, size
        ))


class MiniMaxH3FiveViewDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3FiveViewDecodeEXPT8",
            display_name="MiniMax H3 Five-View Decode (EXP/T8)",
            description=(
                "Decode five image slots separately through five legal two-token VAE clips; "
                "reject ordinary video latents."
            ),
            category=FIVE_VIEW_CATEGORY,
            inputs=[
                io.Latent.Input("five_view_latent"),
                io.Vae.Input("video_vae"),
            ],
            outputs=[
                io.Image.Output(display_name="views_batch"),
                io.Image.Output(display_name="horizontal_sheet"),
                io.String.Output(display_name="report_json"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, five_view_latent, video_vae):
        return io.NodeOutput(*decode_five_view_sheet(five_view_latent, video_vae))


FIVE_VIEW_EXP_NODE_CLASSES = [
    MiniMaxH3FiveViewConditioningEXPT8,
    MiniMaxH3FiveViewDecodeEXPT8,
]
