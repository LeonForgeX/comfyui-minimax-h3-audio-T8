# HyperVAE Krea2+MiniMax v2 2× 视频 VAE（EXP）

`MiniMax H3 HyperVAE 2× 加载 (T8 EXP)` 是独立、显式选择的视频 VAE 加载节点，不替换项目原有视频／音频 VAE，不修改旧工作流或 ComfyUI 核心。将权重放进 `ComfyUI/models/vae/` 后，可像普通 VAE 加载节点一样从 `vae_name` 下拉选择；`absolute_path` 默认留空，仅保留给已有绝对路径工作流兼容使用，填写时仍优先。权重不随代码发布。

权重的 H3 编码器仍是空间 16×，解码头输出 12 个相位打包 RGB 通道，PixelShuffle 后宽高各为普通 H3 视频 VAE 的 2×。这是最终视频 VAE 解码的像素放大，**不是**采样器之间的 2× latent upscaler。音频 VAE 仍走旧连接。普通 ComfyUI VAE Loader 无法直接使用这个 12 通道解码头。

示例：[5 秒同潜空间双路画布工作流](../examples/workflows/58-hypervae-2x/README.md)。它只用一个 8 步 sampler，原生 AV Decode 后将同一 `video_latent` 分给 HyperVAE 解码，音频只解码一次；对照标签明确，不作为匿名盲测或普适画质推荐。使用到其它二采／长视频图时，预留 4 倍输出像素内存，使用新的缓存身份／输出前缀，不复用旧结果，并单独验证最终音画。

本机用户提供的 v2 文件 SHA-256 为 `84DA7F476F3732D2B7F6CD51B03063979E8CDCE225032A6D4ED16542775A2DE2`。复制到 `models/vae` 的文件哈希完全相同。真实权重已完成单帧与 5 帧极小输入的编码／解码及单帧 tiled 解码，形状、有限值、16×编码合同通过。另从真实 ComfyUI 画布打开上述保存图并点击运行：512×288 原生路与 1024×576 HyperVAE 路各产出 120 帧、24 fps、精确 5.000 秒 H.264/AAC；双路严格音画解码通过，解码音轨 SHA-256 相同，且可在画布预览。修订后的目录下拉图再次从画布运行，实际提交的 `absolute_path` 为空、`vae_name` 为 `models/vae` 中的 HyperVAE 文件；两路再次各 120 帧、5.000 秒、严格音画解码通过、解码音轨 SHA-256 相同。第一次模板裁剪误接渲染时长而产出 124 帧／5.167 秒，仅保留为本地失败证据；正式图已改接最终时长。

这只是一个短片、一个 seed 和一套本机权重的真实运行，不证明原尺寸、长片、所有双采路线、主观画质提升或旧图逐帧等价。

来源：[作者模型页](https://civitai.com/models/2894736/hypervaekrea2minimax?modelVersionId=3301496)、[作者的 MiniMax 2× VAE 解码实现](https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler/blob/main/vae_decode.py)。仅使用本机权重，没有打包第三方代码或模型。
