"""Derive the one-sample, same-latent HyperVAE canvas graph from TRT comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/workflows/30-trt-vae/2026-09-10_H3_TRT_VAE_Same_Latent_Compare_EXP.json"
TARGET = ROOT / "examples/workflows/58-hypervae-2x/2026-09-27_H3_HyperVAE_2x_Same_Latent_5s_Compare_EXP.json"


def build() -> dict:
    graph = json.loads(SOURCE.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert len(nodes) == 22 and len(graph["links"]) == 32
    assert nodes[17]["type"] == "MiniMaxH3TRTVAEDecoderEXPT8"
    assert nodes[16]["type"] == "SamplerCustomAdvanced"
    assert [link[:5] for link in graph["links"] if link[3] == 18] == [
        [25, 10, 2, 18, 0], [26, 17, 0, 18, 1]
    ]
    graph["id"] = "2e725f45-f9b6-4818-ac13-f7f839bde293"
    graph["revision"] = 0
    nodes[4]["widgets_values"][1:4] = [512, 288, 124]
    nodes[5]["widgets_values"] = [
        "One continuous cinematic street-level shot of an adult woman in a red raincoat "
        "cycling slowly past warm shop windows on a wet evening street. Natural realistic "
        "motion, subtle reflections and soft city ambience, no speech, no captions, no cuts."
    ]
    # DurationPlanner keeps H3's 4n+1 latent-frame contract. OutputTrim cuts to 5.0s.
    nodes[6]["widgets_values"] = [0.0, 5.0, 0.0, 0.0, False, 0.0]
    # The TRT template trims to render_duration (4n+1 = 124 frames, 5.167s).
    # This 5-second demo must instead use the planner's requested final duration.
    nodes[6]["outputs"][1]["links"] = None
    nodes[6]["outputs"][5]["links"] = [14, 29]
    for link in graph["links"]:
        if link[0] in (14, 29):
            assert link[1:3] == [6, 1]
            link[2] = 5
    nodes[12]["widgets_values"][0] = "MiniMaxH3/HyperVAE-2x/5s_native"
    nodes[17]["type"] = "MiniMaxH3HyperVAE2xLoaderEXPT8"
    nodes[17]["title"] = "HyperVAE 2x 视频VAE（仅右侧）"
    nodes[17]["properties"] = {"Node name for S&R": "MiniMaxH3HyperVAE2xLoaderEXPT8"}
    nodes[17]["widgets_values"] = [
        "hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors",
        "",
    ]
    nodes[20]["widgets_values"][0] = "MiniMaxH3/HyperVAE-2x/5s_hyper2x"
    nodes[22]["widgets_values"] = [
        "# HyperVAE 2× · 同一次采样的5秒画布对照 (EXP)\n\n"
        "运行前将 HyperVAE Krea2+MiniMax v2 权重放入 ComfyUI/models/vae，"
        "在加载节点的 vae_name 下拉框选择它；absolute_path 通常留空。"
        "左路为原生视频VAE 512×288，右路是同一个 video_latent 经 HyperVAE 解码后1024×576；"
        "两路共用一次8步采样、同一条原生音频和 5 秒裁剪。右路并非二次采样。\n\n"
        "点击画布 Queue 即实际生成和保存两个 MP4。此实验不替换旧工作流、不自动宣称画质提升。"
    ]
    nodes[22]["title"] = "先读这里 / HyperVAE 2×"
    graph["extra"]["workflow_title"] = "H3 HyperVAE 2x Same Latent 5s Comparison EXP"
    graph["extra"].pop("trt_vae_delivery_status", None)
    graph["extra"]["hyper_vae_2x_status"] = "canvas_run_120_frames_5_seconds_strict_decode_pass_single_sample"
    return graph


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_hyper_vae_2x_workflow.py [output_path]")
    target = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
