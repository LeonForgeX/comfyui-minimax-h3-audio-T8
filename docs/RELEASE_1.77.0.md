# 1.77.0 — 渐进采样与 DLSS 插帧 / Progressive sampling and DLSS FI

## 新增功能

- 渐进首采：原生 H3 小画幅 6 步 → 学习型潜空间放大 → 大画幅 2 步，总共 8 步。提供独立 T2VA/I2VA EXP 工作流，不是 VDN 8+4 二采，不替换旧图。
- DLSS 视频插帧：独立 EXP 文件节点，固定 2x，保持分辨率、播放时长与原音轨；支持显式切镜位置。不是超分或 H3 生成加速。
- 共 320 个节点、226 份工作流 JSON。新增工作流在 `examples/workflows/28-progressive-sampling` 与 `29-dlss-fi`。

## 模型和安装

渐进首采复用原有 H3 模型、新版 Turbo EMA B 和学习型放大模型：

`ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`

不需要 SelfLift 插件，也没有新训练或转换的模型。插帧不需要额外 H3 模型，但需要用户自行准备匹配的 `dlssg-worker.exe` 和 `nvngx_dlssg.dll`，放入 `ComfyUI/models/DLSS-FI/310.7.0` 或指定实际目录。详情、文件哈希和 Python 依赖见 [插帧说明](../examples/workflows/29-dlss-fi/README.md)。运行文件不随节点分发，不自动下载或安装。

## 实测和限制

- RTX 4060 Ti 16GB、本机 Core、1024×512、约 3 秒短片热测：T2VA 中位数 126.15→79.03 秒，I2VA 139.41→86.50 秒，端到端耗时分别减少37.35%和37.95%。每路只有两个正式热测样本，不是通用承诺，也没有证明显存一定更少。
- 已审画面相近可用，人物组音乐、人声和口型正常。三组游戏有声音偏轻或缺脚步声反馈，第一组还存在选项与备注矛盾；保留问题，不默认归一化音量，也不声称所有音频通过。
- 插帧的游戏、固定字幕、闪光低纹理、显式硬切短片人审可接受；不保证所有快动作、文字或 HUD 无伪影。只支持当前限定的 Windows 单 RTX、8bit SDR、CFR、偶数尺寸文件输入；资源检查和其他限制见节点说明。
- 32秒渐进路线没有通过：原生基线触发资源保护，没有合格长片。TRT 解码尚未实现。完整浏览器 API 导出往返未验证；文件工作流和后台 API 检查不能替代该证据。
- 开发基线全仓 CPU 回归3779项通过；本次发布不重新生成盲测视频。EXP 发布不代表未通过的路径已转为正式质量保证。

## English summary

This release adds two optional EXP nodes: native H3 progressive first sampling (6 low-resolution evaluations plus a learned latent upscale and 2 high-resolution evaluations) and file-based DLSS 2x frame interpolation. Three workflows are included in folders28 and29; existing workflows remain available. Total:320 nodes and226 workflow JSONs.

The tested approximately three-second T2VA/I2VA warm runs saved37.35%/37.95% end-to-end time on one RTX4060Ti16GB, with only two measured runs per route. This is not a universal performance or VRAM guarantee. Reviewed visuals were comparable; scored portrait music, speech and lip-sync passed. Game audio concerns and one conflicting A/B answer remain unresolved. The32-second progressive route has not passed, TRT decoding is deferred, and browser API-export round-trip verification remains incomplete.

FI preserves duration, resolution and source audio, not video pixels after H.264 encoding. It requires separately obtained matching DLSSG runtime files and optional Python dependencies; none are downloaded, installed or redistributed by the node. See the workflow guide for paths, hashes, platform/input limits and conservative resource guards. Acceptance of the four short FI cases is not a guarantee of artifact-free motion, text or HUDs on every source.
