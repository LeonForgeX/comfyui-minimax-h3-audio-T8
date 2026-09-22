"""Build the isolated HyperFlow eight-second/two-segment EXP API graph.

This is a graph builder only. It does not queue Core or assert seam quality.
"""
from __future__ import annotations

from copy import deepcopy


NODE = "MiniMaxH3HyperFlowLongVideoEXPT8"
LORA = "MiniMaxH3LoRACompatibilityLoaderT8Advanced"


def _defaults(info: dict) -> dict:
    values = {}
    for name, spec in info["input"]["required"].items():
        choices, options = spec
        if "default" in options:
            values[name] = deepcopy(options["default"])
        elif isinstance(choices, list):
            values[name] = choices[0]
    return values


def build_prompt(info: dict, *, chain_id: str,
                 prompt: str = "A steady cinematic shot of a quiet candle-lit room, natural motion and room tone.",
                 low_lora: str = "disabled", high_lora: str = "disabled",
                 low_strength: float = 0.0, high_strength: float = 0.0,
                 width: int = 896, height: int = 448,
                 low_width: int = 448, low_height: int = 224) -> dict:
    if not chain_id or chain_id == "h3_hyperflow_long_video_exp":
        raise ValueError("Choose a unique HyperFlow EXP chain_id; never reuse the sample or a legacy chain")
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": LORA, "inputs": {
            "model": ["1", 0], "lora_name": low_lora, "strength_model": low_strength}},
        "3": {"class_type": LORA, "inputs": {
            "model": ["1", 0], "lora_name": high_lora, "strength_model": high_strength}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    }
    graph["8"] = {"class_type": NODE, "inputs": {
        **_defaults(info[NODE]),
        "model_pass1": ["2", 0], "model_pass2": ["3", 0],
        "clip": ["4", 0], "video_vae": ["5", 0], "audio_vae": ["6", 0],
        "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors",
        "upscaler_model": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "chain_id": chain_id, "filename_prefix": "H3_HyperFlow_Long_Video_EXP",
        "low_width": low_width, "low_height": low_height, "width": width, "height": height,
        "total_duration_seconds": 8.0, "render_window_frames": 124,
        "context_frames": 22, "global_prompt": prompt,
    }}
    return graph
