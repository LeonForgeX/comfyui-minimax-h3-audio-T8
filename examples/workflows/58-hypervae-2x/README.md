# HyperVAE 2× 同潜空间对照（EXP）

打开 `2026-09-27_H3_HyperVAE_2x_Same_Latent_5s_Compare_EXP.json`。这是一张可直接在 ComfyUI 画布运行的 5 秒 T2VA 图：真实 FL2VA／Qwen／Turbo 8 步只采样一次，原生 H3 视频 VAE 与 HyperVAE 分别解码同一个 `video_latent`，共用原生音频 VAE 和同一条裁剪后的 AUDIO，保存 `5s_native` 与 `5s_hyper2x` 两个 MP4。

示例使用 512×288 生成尺寸、24 fps、输出裁到 5 秒；HyperVAE 分支应输出 1024×576。把 HyperVAE 权重放入 `ComfyUI/models/vae/`，在加载节点的 `vae_name` 下拉框选择实际文件；示例预选 `hyperVAEKrea2Minimax_v20MinimaxX2Upscale.safetensors`，`absolute_path` 默认留空。若本机文件名不同，只需更改下拉选择，无需填写机器专属路径。权重不会随图发布。不要将右侧误认作二次采样或 latent 超分。

旧图和原生 VAE 默认路线完全保留。HyperVAE 节点属于可选 EXP，先看 [接入说明](../../../docs/HYPERVAE_2X_EXP.md)；单例实际跑通不等于画质优于原生 VAE，也不证明长视频、双采或其它画幅均已验证。
