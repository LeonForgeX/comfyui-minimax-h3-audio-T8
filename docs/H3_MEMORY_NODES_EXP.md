# MiniMax H3 独立低显存节点（EXP）

这两个节点只修改传入的 `MODEL` 克隆，不修改 ComfyUI 全局设置，也不依赖或导入
ComfyUI-KJNodes：

- `MiniMax H3 Low VRAM Attention / 低显存注意力 (Advanced EXP/T8)`
- `MiniMax H3 Chunk FeedForward / 分块前馈 (Advanced EXP/T8)`

推荐接线：

```text
H3 模型加载 / 可选普通权重 LoRA
  → Low VRAM Attention（双模型长视频当前候选用 head_chunks=4）
  → Chunk FeedForward（chunks=2, seq_threshold=4096）
  → H3 采样器
```

两个 T8 节点也可以反向串联，或只接其中一个。不要再同时接 KJ 的同名 LowVRAM、
ChunkFFN 或 H3 Memory Efficient Sage 节点；它们会争用同一组模块 forward，T8 节点会
明确拒绝，不会静默覆盖。

## 参数

### Low VRAM Attention

- `head_chunks=1`：仍然启用 block 输入提前释放，但 attention 只调用一次，不做头分组。
- `head_chunks=4`：把注意力头分为最多 4 组，每组单独调用当前 Core attention，降低
  单次内核临时量；实际组数不会超过模型的 head 数。它会改变内核调用形状和浮点舍入。
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

短片通过只证明该固定条件，没有证明 OpenVDN、EAV、TST、SLA 或 FastH3 等其他组合兼容。

## 双模型长视频内循环

`2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json` 已把这两个节点接入
双模型 4+4 内循环。LOW 与 HIGH 必须保持完全独立：

```text
LOW 底模 → LOW LoRA → LOW LowVRAM(h4) → LOW ChunkFFN(c2) → LOW MODEL
HIGH 底模 → HIGH LoRA → HIGH LowVRAM(h4) → HIGH ChunkFFN(c2) → HIGH MODEL
```

专用身份合同会逐路绑定实际权重、LoRA、`head_chunks`、`chunks`、`seq_threshold`、object
patch、runtime token、wrapper 所有权和源码 SHA。切换任一值后旧阶段缓存必须失效；缺失、
伪造或被下游覆盖的 receipt 会在采样前失败。Prompt Relay 可以与这两个 T8 wrapper 共存，
但不能再叠加占用相同 forward 的 KJ LowVRAM/ChunkFFN。

本机已严格串行完成两次两段共 8 秒真实 H3 运行。两个候选均为每段 LOW4→学习型 latent
放大→HIGH4，最终 896×448、192 帧、24fps，H.264 视频和 32kHz 双声道 AAC 音频严格
解码通过。`head_chunks=1 + ChunkFFN=2` 端到端约 520.79 秒，最高观察约 12.95GiB 已用；
虽然 CPU 亮度指标较小，但用户完整播放后确认其续段接缝明显，因此已淘汰，不作为默认。

`head_chunks=4 + ChunkFFN=2` 首片端到端约 627.85 秒，最高观察约 12.48GiB 已用、最低约
3.52GiB 空闲；每段 LOW/HIGH 均有完整 h4/c2 回执。用户认为它的约 5.17 秒续段接缝暂时
正常，但两个候选在原音乐厅提示词下都有移动光影。

随后只追加了一条不同内容的 h4+c2 八秒复核，没有再跑 h1 或长片。复核使用均匀浅灰平面
背景、五条固定定位线和跨段连续移动的橙色卡片，并明确禁止渐变、高光、阴影、反射、辉光、
光斑、曝光／色温变化及亮度呼吸。成片仍出现移动彩色光圈，因此该现象不只是提示词描绘的
光照，更可能是模型／风格生成伪影；这不自动证明接缝失败。机器分析测得第 124 帧全局平均
亮度跳变约 0.50%、角落背景约 0.71%，但固定线条、接缝观感和环境底噪仍由人工播放确认。
两种配置都不是通用 16GiB、安全、省显存或提速保证。

用户再用 2:3 外滩 I2VA 内容复核时确认：人物和其余画音正常，但旧
`high_native_mask_exp` 硬边界会在约 5.17 秒让背景突然更换。为此新增可选
`high_native_mask_ramp_exp`：先精确锁定 HIGH 的 7 个上下文 latent 单元，再让随后三个单元
按 `0.25 → 0.50 → 0.75` 逐步恢复可生成区域；音频 latent 和音频 mask 保持原对象和值。
它不会改变旧节点默认 `reference_only`，也不会迁移旧 JSON。正式 T8 低显存工作流现显式选择
`accepted_picture_low_context_v1 + high_native_mask_ramp_exp`，并使用新的 `chain_id`，避免误命中
硬边界缓存。

同一首帧、提示词、seed、4+4、h4+c2、512×768 和 22 帧上下文的渐释候选已串行完成：
192 帧、24fps、H.264/AAC 严格解码通过；两段 LOW/HIGH 都记录 4 次 forward、800 次 Relay
路由以及 h4/c2 回执。媒体 SHA 为
`b8659a058204e917d8bcab808c6abcfb9a86f6e86f10f87159419ed9381b23e9`，机械审计 SHA 为
`708371b4a2843d41077ec19074be36e6581db2ae0ab013ee4862797a67b154bd`。逐帧机器检查未再看到
旧硬边界候选的瞬时斑块，背景楼体和栏杆构图连续；最终仍须用户在完整播放中确认 5.17 秒
附近的背景、人物、声音和口型，不能把该检查写成人审通过。

两版前 124 帧解码 RGB 逐像素相同，因此背景诊断只比较第二段。相对旧硬边界，渐释版在
123→130窗口内的背景MAD峰值降低33.9%、95分位变化降低26.4%、大变化像素比例降低45.8%、
光流峰值降低13.3%、边缘突变降低16.4%，直接段界全局SSIM从0.85283升至0.86496。
回执为 `artifacts/t8-memory-dual-8s-h4c2-bund-i2va-apc22-ramp-gpu-20260916-v2/`
`background-seam-comparison.json`，但这些非预注册诊断仍不能替代人审。

同一渐释链还在全新隔离 ComfyUI 进程中执行了缓存续跑：返回
`returned_verified_existing_final`，重新核验两段 audit，30 个既有输出／阶段缓存文件的大小和
SHA-256 全部保持不变，最终 MP4 仍为上述 SHA，且没有重采样。回执位于
`artifacts/t8-memory-dual-8s-h4c2-bund-i2va-apc22-ramp-resume-20260916-v1/terminal.json`
（SHA-256 `692dd8d9592be713bd5da017678185118b5342453009b111b43a5e6843d9dacb`）。
这证明精确测试链能在新进程续跑，不代表任意被修改的模型、LoRA、提示词或缓存应被复用。

### 2026-09-16 渐释后的轻微颜色跳变

用户确认 HIGH 渐释版的背景连续性明显改善，但仍看到轻微颜色跳变。执行报告显示既有 Color
Match V2 已启用并实际应用；它并非缺失，而是只做跨段色彩／空间分布匹配，未抑制续段第2、3帧
的短促低频色调摆动。追加选项 `bounded_spatial_temporal_exp` 在 V2 后只处理续段前12帧：
用前段尾部与本段头部的5帧时间中值估计 RGB 均值目标，每通道最多校正0.015并线性渐退。
它不混帧、不生成细节、不改几何／latent／音频。旧图仍默认 `bounded_spatial_v2`；切换必须换
`chain_id`。复用现有8秒成片的A/B只改变第124、125、126、129帧，其他帧解码像素相同，AAC
包载荷逐包哈希相同；相邻RGB均值峰值降低约85.4%。用户复核B表示“基本可以”。进一步的
`bounded_motion_color_exp` 仅修正有可信运动对应和前后支持的局部低频异常，不替换B；
用户已接受局部C并要求发布，轻微变色留待后续。推荐图保存LOW256×384→HIGH512×768的2:3首帧
控制组合及新模式／chain；旧默认不变。该组合不是新图完整GPU复跑或逐位复现保证。
完整边界见[局部修色说明](MOTION_COLOR_EXP.md)。
