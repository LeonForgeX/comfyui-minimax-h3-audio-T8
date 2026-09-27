# v1.86.0 — HyperVAE 2× 视频 VAE

新增 `MiniMax H3 HyperVAE 2× 加载 (T8 EXP)`，只在用户显式连接时使用。普通 H3 视频 VAE、音频 VAE、现有节点接口和旧工作流不迁移。模型文件请自行放入 `ComfyUI/models/vae/`，然后从 `vae_name` 下拉框选择；`absolute_path` 默认留空，仅作为旧工作流兼容选项，填入时仍优先。模型权重和本地测试视频不随插件分发。

[5 秒同潜空间对照工作流](../examples/workflows/58-hypervae-2x/README.md)只采样一次，把同一潜空间送到原生视频 VAE 和 HyperVAE 2×，共用同一音频。示例使用指定用户权重、512×288 原生输出和 1024×576 HyperVAE 输出。修订图在真实 ComfyUI 画布以 `vae_name=hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors`、`absolute_path=""` 提交并成功执行，生成两条各 120 帧、24 fps、精确 5.000 秒的 MP4；严格完整音画解码通过，两路解码 PCM SHA-256 相同。此目录选择运行与先前绝对路径运行是不同的请求，不混作同一证据。

该功能仍为 EXP。它是最终 VAE 解码像素宽高各 2×，不是采样间 latent 超分；实际输出内存约随像素量增大。单片通过不表示画质优于原生 VAE、所有素材和双采图通用、长视频稳定或旧图逐帧等价。具体原理和限制见[HyperVAE 说明](HYPERVAE_2X_EXP.md)。

公开发布以 Registry API 显示本版本 `Active`、节点公开 `latest_version` 指向本版本和实际可安装为准。GitHub 代码／Release 或 Registry 上传成功不能代替该验收；若官方安全审核仍拦截，必须保留为未完成状态并通过既有[人工复核单](https://github.com/Comfy-Org/registry-backend/issues/233)处理，不重复覆盖旧版本归档。
