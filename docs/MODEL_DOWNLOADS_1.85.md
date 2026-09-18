# 1.85 模型下载与资源准备

[TAEH3](https://huggingface.co/t8star/Taeh3-Comfy) · [Meridian + Omega](https://huggingface.co/t8star/Meridian-Comfy)

两个模型仓库都按 `ComfyUI/models/` 目录整理。合并目录而非只把所有文件放到同一个文件夹。
完整 SHA256、上游版本、每个文件的许可与社媒／节点链接均见仓库 README。

## TAEH3：只需要两个小权重中的一个

时序版 `models/vae_approx/taeh3.safetensors`，22,709,752 字节，默认推荐。
逐潜帧 2D 版 `models/vae_approx/taeh3_2d_kijai.safetensors`，9,791,388 字节，手动选择。
不是新训练／量化权重；2D 仅另存文件名，不能覆盖时序版。两者不生成声音，不替换最终 VAE。
使用 [一采预览节点](TAEH3_SAMPLING_PREVIEW_EXP.md)，原 sampler／sigmas 接线保持。

## Meridian：先准备源码，再合并权重目录

主模型 `models/meridian/meridian_dmd_int8_convrot_comfy.safetensors` 是合并 DMD 后的原生 ConvRot INT8。
正确 Omega 为 `models/meridian/vggt-omega/checkpoints/vggt_omega_1b_512.pt`，原始 PT，不是 INT8、普通 VGGT 或 256 版。
两者已提供，不需要用户再申请 Omega 原仓库权限才能从上述镜像下载。
Meridian 仍遵守 MiniMax H3 Community License；Omega 单独遵守 FAIR 非商用研究许可及 AUP。

权重不是完整程序，另外需要：

1. 可复用的 H3 视频 VAE：`models/vae/minimax_h3_video_vae_fp16.safetensors`。
2. [Meridian 固定源码／assets](https://huggingface.co/Viggle/Meridian/tree/2083d059d8544ff7eaaf86966b83e4964a904737)：保留 `recam/` 与 `assets/`，放 `models/meridian/source/`。不需要下载 14 个原 BF16 teacher 分片；73 帧示例需要 `fixed_embed_73.pt` 和 `silence_audio_73.pt`。其他长度使用对应配对文件，不切片凑数。
3. [Omega 源码](https://github.com/facebookresearch/vggt-omega/tree/b2c61f6631d9f344a2d914bfba5d9529d6fc1d35)，已核验的版本是 `b2c61f6631d9f344a2d914bfba5d9529d6fc1d35`。默认根目录 `models/meridian/vggt-omega/` 必须实际含 `vggt_omega/__init__.py`；按照其说明安装依赖。不要重新安装／替换整个 ComfyUI 的 Torch 环境。

**先取得源码，再合并模型目录**，避免 `git clone` 遇到已经放有权重的非空目录。
如果已经下载权重，保留它；把源码放到另一个空目录，在「Meridian 1 · 模型与资源」分别填写
`omega_source_directory` 和 `omega_checkpoint` 的实际路径，无需重下或复制 34GB 主模型。

使用[正式四节点工作流](../examples/workflows/37-meridian/README.md)。DMD 已合并，不能再叠加同一份 DMD。
磁盘文件大小不是显存需求；指定短片通过，不保证任意素材、远距离新视野或通用 16GB 性能。

## 本次改动边界

节点说明与模型参数 tooltip 增加下载入口；模型选择、输入／输出、默认值、采样和最终声音不变。
首页已简化，旧介绍与历史更新分别保留在[详细说明](README_DETAILS_ZH.md)和[更新日志](../CHANGELOG.md)。
