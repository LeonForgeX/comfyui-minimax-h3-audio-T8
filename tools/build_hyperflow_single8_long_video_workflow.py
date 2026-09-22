"""Build a final-resolution native single8 comparison graph; never queue Core."""
from __future__ import annotations

from copy import deepcopy


NODE = "MiniMaxH3HyperFlowSingle8LongVideoEXPT8"


def build_prompt(info: dict, *, chain_id: str, prompt: str,
                 width: int = 1024, height: int = 576) -> dict:
    if not chain_id or chain_id == "h3_hyperflow_single8_long_video_exp":
        raise ValueError("Choose a unique single8 EXP chain_id")
    values = {}
    for name, spec in info[NODE]["input"]["required"].items():
        choices, options = spec
        if "default" in options:
            values[name] = deepcopy(options["default"])
        elif isinstance(choices, list):
            values[name] = choices[0]
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_int8_convrot.safetensors",
            "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3LoRACompatibilityLoaderT8Advanced", "inputs": {
            "model": ["1", 0], "lora_name": "disabled", "strength_model": 0.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "6": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "8": {"class_type": NODE, "inputs": {
            **values, "model": ["2", 0], "clip": ["4", 0],
            "video_vae": ["5", 0], "audio_vae": ["6", 0],
            "hyperflow_file": "hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors",
            "chain_id": chain_id, "filename_prefix": "H3_HyperFlow_Single8_Long_Video_EXP",
            "width": width, "height": height, "global_prompt": prompt,
            "total_duration_seconds": 8.0, "render_window_frames": 124,
            "context_frames": 22,
        }},
    }
    return graph
