# v1.81.0 — H3 独立低显存节点与语音边界加固

## 完成内容

- 新增 `MiniMaxH3LowVRAMAttentionT8Advanced`：在当前原生 H3 block 公式内提前释放中间量，并按 `head_chunks` 分组调用已有 attention backend。
- 新增 `MiniMaxH3ChunkFeedForwardT8Advanced`：只在 packed token 数严格超过 `seq_threshold` 时按 token 轴分块执行同一套 SwiGLU；`chunks=1` 是不克隆、不打补丁的真实旁路。
- 两个节点不导入也不要求 KJNodes，可单独使用或任意顺序串联。普通权重 LoRA 保留；重叠的 KJ/第三方 block、attention、MLP forward 所有者和 DiT block replacement 会明确拒绝。
- H3 block 合同同时接受当前不带 `attention` 形参的 Core 和仍保留该可选形参的旧 Core；不会因 Core 签名升级误报不兼容。
- Speech Studio 新建节点默认提供 `auto_reference_voice`：参考音色模式解析为保守能量边界裁切，描述音色解析为不裁切；已有工作流保存的 `none` 或 `conservative_energy` 不会被迁移或覆盖。
- faster-whisper 边界验证启用 VAD 以获得更可信的首尾词时间戳，但完整目标文本仍须唯一、逐字符匹配才允许 ASR 精确裁切，不引入模糊匹配。
- Sol 验证探针按当前 Core 的认证 packed H3 layout 和 `sink_blocks` / `sink_q` 接口运行；这是验证工具修复，不把外部独立 Sol 节点打包进本仓库。

## 默认参数与短片证据

- `Low VRAM Attention`: `head_chunks=4`
- `Chunk FeedForward`: `chunks=2`, `seq_threshold=4096`
- 固定本机 H3 T2VA、1024×512、73 帧、4 步、同模型/LoRA/提示词/seed、KJ 全局 Sage `auto` 的严格串行短片：基线峰值 14922.98 MiB、79.94 秒；组合候选峰值 14616.25 MiB、82.27 秒。
- 两路均产生可严格解码的视频和音频。候选在该样本少用 306.73 MiB（2.06%）峰值显存，耗时增加 2.33 秒（2.92%）；调用形状变化会带来浮点舍入差异，不声明逐像素一致。
- 参考音色边界诊断只复核已有生成音频，没有重跑 GPU。5.875 秒解码音频经保守边界得到 3.9944375 秒；ASR 因 `いい`/`良い` 非逐字符一致而正确拒绝精确文本裁切。

## 兼容与限制

- 旧 334 个节点保持原顺序，新节点仅追加为第 335、336 个节点；现有工作流无需迁移。
- 当前短片只证明绑定环境和参数，不承诺其他模型、分辨率、时长、显卡或 allocator 获得相同显存收益，也不承诺提速。
- `Low VRAM Attention` 会保留并分组调用已有 callable attention override；外部 backend 仍应按具体版本跑短片验证。不要叠加另一个占用同一 H3 forward 的低显存节点。
- 双模型节点内长循环仍有独立 MODEL 身份合同，本次未把两个新节点加入该专用链；要支持它需另做断点恢复、身份和接缝回归。
- 语音自动边界只处理低能量对齐留白，不能修复音色、语义、口型、音乐或主动非语音内容。

详细接线和参数边界见 [H3 低显存节点说明](H3_MEMORY_NODES_EXP.md)；语音诊断见 [Speech validation report](SPEECH_VALIDATION_REPORT.md)。
