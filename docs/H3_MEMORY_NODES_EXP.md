# MiniMax H3 独立低显存节点（EXP）

这两个节点只修改传入的 `MODEL` 克隆，不修改 ComfyUI 全局设置，也不依赖或导入
ComfyUI-KJNodes：

- `MiniMax H3 Low VRAM Attention / 低显存注意力 (Advanced EXP/T8)`
- `MiniMax H3 Chunk FeedForward / 分块前馈 (Advanced EXP/T8)`

推荐接线：

```text
H3 模型加载 / 可选普通权重 LoRA
  → Low VRAM Attention（head_chunks=4）
  → Chunk FeedForward（chunks=2, seq_threshold=4096）
  → H3 采样器
```

两个 T8 节点也可以反向串联，或只接其中一个。不要再同时接 KJ 的同名 LowVRAM、
ChunkFFN 或 H3 Memory Efficient Sage 节点；它们会争用同一组模块 forward，T8 节点会
明确拒绝，不会静默覆盖。

## 参数

### Low VRAM Attention

- `head_chunks=4`：把注意力头分为最多 4 组，每组单独调用当前 Core attention，降低
  单次内核临时量；实际组数不会超过模型的 head 数。
- `head_chunks=1`：仍然启用 block 输入提前释放，但 attention 只调用一次，不做头分组。
- 分组越多通常临时显存越低，调用开销也越高；不是越大越好。

节点保留已有的 callable `optimized_attention_override`，每个 head group 都会访问它；
这只表示链路没有被丢弃，不等于所有第三方后端都已证明支持分组。普通 Core backend
是直接范围；Sol/Sage 等外部 backend 需要按具体版本做一次短片验证。节点还发布
`sol_take_forward`，供后接的 ComfyUI-SolAttn_triton 在合格调用上保留本低显存 forward。

### Chunk FeedForward

- `chunks=2`：当 packed token 数超过阈值时，把 token 轴分为两块执行同一套 SwiGLU。
- `seq_threshold=4096`：只有 `packed_rows > 4096` 才分块；等于 4096 仍走一次原生 FFN。
- `chunks=1`：返回完全相同的原 `MODEL` 对象，不克隆、不检查、不打补丁。

FFN 节点不改变 attention，可与不替换 DiT block/MLP forward 的 attention backend 共存。

## 已知兼容边界

直接支持：

- 当前原生 ComfyUI MiniMax H3；
- 在节点前加载的普通权重 LoRA；
- 两个 T8 节点单独使用或任意顺序串联。

明确拒绝：

- 同一 T8 节点重复连接；
- KJ 或其他节点已经占用相同 block/attention/MLP forward；
- `patches_replace["dit"]` 的 DiT block 替换；
- 不完整或被篡改的 T8 ownership receipt。

运行时还会检查补丁函数和私有 receipt 是否仍属于本节点；下游如果覆盖 forward，会在
进入 diffusion 前报错，而不是悄悄变成另一种算法。

## 性能与画质声明

前置原型在一台本机、固定 3 秒 1024×512、72 帧、4+4 采样条件下，组合候选相对
基线峰值显存从 12457.30 MiB 降到 11651.57 MiB，即减少 805.73 MiB（6.47%），
耗时约为 1.0678 倍。用户明确认为该收益值得进入节点开发。

独立节点候选随后用标准 H3 T2VA 采样链、同模型、同 LoRA、同提示词、同 seed、
1024×512、73 帧、4 步和 KJ 全局 Sage `auto` 做了严格串行实测：

- 原生基线：峰值 14922.98 MiB，端到端 79.94 秒；
- `head_chunks=4 + chunks=2`：峰值 14616.25 MiB，端到端 82.27 秒；
- 此固定样本少 306.73 MiB（2.06%），耗时增加 2.33 秒（2.92%）；
- 两路均输出 73 帧、1024×512、24fps 的 H.264 MP4，视频与音频严格解码通过；
- 两路是不同的生成结果，不声明逐像素或逐样本一致，仍需按生成任务正常审片。

这不是所有模型、分辨率、帧数、GPU、Core 或 attention backend 的通用保证。两个节点
保持数学公式，但 GEMM/attention 的调用形状变化仍可能改变浮点舍入，因此除
`Chunk FeedForward(chunks=1)` 的真实旁路外，不声明 bit-exact。实际收益还取决于序列
长度、量化融合、allocator 状态和其他 GPU 进程。

## 最小验收建议

先用同模型、同提示词、同 seed、同采样器跑 3 秒对照；GPU 任务严格串行。至少记录：

1. 原生 baseline；
2. `head_chunks=4 + chunks=2 + seq_threshold=4096`；
3. 峰值显存、端到端时间、完整视频/音频解码；
4. 如果叠加第三方 attention backend，再单独验证对应顺序和实际 backend 调用。

短片通过只证明该固定条件，没有证明长视频、OpenVDN、Relay、EAV、TST、SLA、FastH3
或双模型长循环兼容。

当前双模型长循环还有独立的 MODEL 内容身份白名单，会主动拒绝这两个新包装节点；本次
没有扩大范围去修改该长循环。要在该专用工作流中使用，需另做身份适配、断点恢复和接缝
回归验证，不能用本次标准采样链结果替代。
