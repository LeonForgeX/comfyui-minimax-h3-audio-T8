# OpenVDN 二次采样 / Two-pass refinement

v1.75.0 提供四张二采工作流，原有单采工作流保留。原生打开、另存、API 回读、回归和隔离版真实模型测试已通过；实际验证范围见下文。

| 工作流 / Workflow | 用途 / Purpose |
|---|---|
| `2026-09-08_H3_OpenVDN_T2VA_vdn_TwoPass_EXP.json` | 文字生成：VDN 8 步 → 潜空间放大 → VDN 4 步 |
| `2026-09-08_H3_OpenVDN_I2VA_vdn_TwoPass_EXP.json` | 首图生成，二采仍使用 VDN |
| `2026-09-08_H3_OpenVDN_T2VA_native_h3_TwoPass_EXP.json` | 文字生成，二采使用独立原生 H3 + EMA B |
| `2026-09-08_H3_OpenVDN_I2VA_native_h3_TwoPass_EXP.json` | 首图生成，二采使用独立原生 H3 + EMA B |

## 怎么用

打开一张工作流，选择本机模型；I2VA 还需选择首图。修改共用的提示词和时长节点即可，不需要分别修改两遍。默认先生成低分辨率画面，再用学习型 2x 潜空间放大和第二遍采样细化画面。保存节点会保留第一遍的声音，不需要再接一个 SaveVideo 重编码。

两条路线的 0.52MP 对照已有人审：画面差不多；古典音乐、“你在哪里”的人声和两边口型正常。I2VA 的环境声也没有杂音。没有选出更好的路线，二采也不保证所有素材都比单采更清晰。

## 模型与依赖

复用已有模型，不需要新转换的底模或新训练的 adapter：

- `ComfyUI/models/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`
- `ComfyUI/models/diffusion_models/OpenVDN/vdn-minimax-h3/`：保留模型包内部完整目录
- `ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`
- 仅原生 H3 二采：`ComfyUI/models/loras/minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors`
- 原有 H3 文本编码器、视频/音频 VAE，以及可从 PATH 调用的 FFmpeg/ffprobe

OpenVDN 安装说明和模型链接见[同目录说明](README.md#openvdn-minimax-h3)。节点不会自动下载模型或安装软件。VDN 分支不要再叠加通用 Turbo/EMA LoRA；原生二采必须使用工作流里独立的 MODEL 分支。

全局 `--use-sage-attention` 可以保留，不代表 VDN 内部改成了 Sage。已识别的 Core 稀疏补丁只在 VDN 分支避让；未知模型/注意力替换仍会报告冲突。完整兼容范围见[兼容说明](../../../docs/CORE_VDN_COMPATIBILITY.md)。

## English

Version1.75.0 provides four two-pass workflows. Choose T2VA for text or I2VA for a first image, then choose VDN or independent native H3 refinement. Both routes finish the VDN 8-step first pass, apply the existing learned 2x latent upscaler, rebuild high-resolution conditioning and refine the video. Shared prompt/duration controls drive both stages. First-pass audio is retained.

Use the existing assets listed above. Only native H3 refinement needs the new EMA B LoRA; do not add it to the VDN branch. FFmpeg and ffprobe must be available on PATH. There is no automatic installation, download or newly converted model requirement.

One reviewed T2VA pair passed music, Mandarin speech and both candidates' lip-sync, with no visual winner. A separate I2VA pair passed ambient-audio review and was visually similar. These results do not guarantee every input, GPU or patch combination. Global Sage can remain enabled, but VDN retains its own attention algorithm. Existing single-pass workflows remain available.
