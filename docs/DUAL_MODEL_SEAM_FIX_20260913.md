# 双采长视频接缝：2026-09-13 修复与防回归

## 结论及证据边界

用户完整看过本次 8 秒候选，反馈“非常好没问题了，可以收尾发布github了，保存工作流，有问题的不要保存”。
这取代此前“仅机械通过／等待人审”的状态。它是指定样片的验收，不是任意素材、后端、时长都无接缝的保证。

通过样片 SHA256：`3ff583bc817dd845fa288aaf0b16c2d2be24a6ce282f50c1b37a823edc91ad73`。
本地诊断目录名：`dual-seam-accepted-picture-0p4mp-8s-gpu-v1`。
最终文件：`assembled/Dual_Model_Pilot_r0002_cosine_bridge.mp4`。
这里文件名的 `cosine_bridge` 仅指 **5ms 音频交叉过渡**，不是视频补帧、叠化或潜空间桥接。

## 问题在哪里

原流程下一段一采使用上一段 **低分辨率 partial x0** 的视频尾部作为参考；二采使用上一段完成后的高分辨率尾部。
partial x0 是完整 8 步表前 4 步给出的预测，并不等同于用户最终看到的成片。
本例两个阶段续接所依据的画面存在差异。控制变量实验将一采的视频参考改为上一段实际成片的尾部，其他采样逻辑不变，接缝及后续画面获得用户认可。

这不是“看了中间 x0 很差，所以认定根因”：中间 x0 解码不能单独证明原因，该早期推断已撤回。
我们能确认的是这次**接受画面 → 一采参考**的干预解决了这份固定样例，不能推论所有项目的 low-x0 continuation 都有错误。

## 修复的精确路径

1. 找到当前链中被选择的直接前段，校验 candidate、chain、段号、路径及视频 SHA256。
2. 用 PyAV 将该段 MP4 解码为 RGB24，取真实最后 **39 帧**。不取另一个试验的尾帧，不取粗采临时预览。
3. 使用项目已有 `core.resize_image` 缩小到一采尺寸，再由当前原生视频 VAE 编码。
4. 只替换传给**下一段一采条件构建器**的 `video_tail`。原音频 tensor、时间元数据不动；高分辨率条件和二采遮罩不动。
5. 其余仍为一采 4 步 → 已有 learned 3D latent upscaler → 二采 4 步。生成的视频不再做潜空间端点偏移或视频叠化。

39 帧是保存的尾部容量（12 个视频 latent 时间单元），不是强行将工作流 `context_frames` 改成 39。
通过样例仍用 `context_frames=22`，条件构建按原逻辑选末尾 7 个单元；39 上下文选择 12 个单元。
22/39 的时间坐标与原生 Core 对照已通过；无需修改时序网格来实现本修复。

本例额外 VAE 准备约 3.74 秒，**额外扩散采样 NFE 为 0**。整次测试约 645.95 秒，最低剩余显存约 2.97 GiB。
耗时仅是该机器该样例的记录，不是通用性能承诺。
第一段 low_x0 / high_input / high_output 的三个 tensor SHA 与原始基线完全一致：

| 阶段 | SHA256 |
| --- | --- |
| low_x0 | `cebe96823275d8c51a7a429a21181d70d59aa3dae360c0d24a4f0bad35944c07` |
| high_input | `44368fdce916e0d188972978aafce504f30ab32af32fa5b918d40faeb3f79eba` |
| high_output | `46e4c0899e8a28a5e14727fe538be0455019cd16ced23dfde3be27bc6c87bdb6` |

## 正式用法

导入 [通过的 0.4MP／8 秒／KJ／4+4 示例](../examples/workflows/04-long-video/2026-09-13_H3_Dual_4plus4_Accepted_Picture_KJ.json)。

- `low_context_source=accepted_picture_low_context_v1` 开启此修复。旧 JSON 缺省仍用 `independent_low_x0`，避免静默改变旧任务。
- 通过组合：448×224 → 896×448；8 秒／24fps／192 帧；window=124；context=22；4+4；两路 EMA B 强度 1；KJ Memory Efficient Sage；Relay 开，EAV 关；auto 原生联合音频；color_match 开。
- 3D upscaler 放在 `models/latent_upscale_models/`；KJNodes 和对应 Sage 环境需另行安装。示例不依赖诊断 monkeypatch。
- 额外LoRA两槽按已验收图保留 `minimax_h3_turbo_4步加速_comfyui.safetensors`、强度0。强度0不增加权重效果，但兼容加载器仍读文件与metadata；所选文件必须存在。更换它也属于新配置，不声称与该样片逐位一致。
- `chain_id` 用新名称开始。改策略、模型、VAE、LoRA、提示词、尺寸等后换名称；不要把失败试验的阶段文件搬进新链。
- `resume_existing=true` 用于参数不变的中断恢复。阶段缓存身份包含策略、前段视频 SHA、VAE／模型内容及实现代码；命中一采缓存时不重复解码和 VAE 编码，但仍校验前段视频身份。
- 全局提示词仅写贯穿全片的场景、人物、衣着、声音氛围。只说一次的台词只写进对应局部事件，不能写进全局反复发送。局部每行一个事件，时间模式要与 `time_ranges` 单位一致。
- 改时长需同步 Relay 的 `length` 覆盖全片。本例用 193 帧计划覆盖 192 帧输出。8 秒两段不等于两段各 4 秒，接缝在 frame124，约 5.17 秒。
- 39 上下文可配置，但本次最终人审是 22 上下文的 8 秒样片。未运行最终策略的 12 秒 GPU 测试；用户接受后取消追加生成。不可声称该组合已完成长时长验证。

## 禁止再次混入的“修复”

- 失败的 post-sampling latent endpoint bridge 会产生重影；不纳入发布代码或正式示例。
- 只增加 39 上下文、只关闭高采 prefix mask、只移除高采 motion VIDEO guides 的本次对照均未解决画面问题，不保存为推荐工作流。
- 不为修视频接缝而冻结未完成的一采音频，不改变原生音频 mask 分区，不删除 Relay 的时间偏置。
- 不顺手改 4+4 步数、sigma 表、3D upscaler、时间长度、参考坐标、颜色算法或生成后的视频 latent。
- 不把一项接缝差值下降、mask 精确、文件能解码、CPU 测试通过，当成人物和后段画面正常。

## 后续更新门禁

涉及 LOW/HIGH 条件、上下文、VAE、3D upscaler、Relay、遮罩、缓存或拼接时：

1. 先读本文件；固定通过示例及旧基线，明确单变量假设；保留原视频和 SHA。
2. 跑 `tests/test_dual_picture_context.py`、`tests/test_dual_high_video_prefix.py`、`tests/test_long_video_dual_model_runner.py`、`tests/test_long_video_dual_stage_cache.py`、`tests/test_nodes_long_video_dual_model.py` 及工作流校验。
3. 必须覆盖：只影响后段 LOW；第一段不变；HIGH 条件不变；音频对象和已知区域不变；0 额外扩散步；正常／中断恢复一致；策略与源视频改变不可误命中；坏哈希、错误前段、越界路径、错帧数、NaN 拒绝。
4. 算法/数值路径改变时，串行复跑同条件 0.4MP／8 秒两段；检查实际 4+4、后端无静默回退、媒体完整解码、阶段 tensor、缓存与音频。
5. 从接缝前一直看到片尾，人工确认无跳切、虚影、后段崩坏和音频问题。机器审计和真人验收分别记录，缺一不能写“质量通过”。

生产移植只允许把已接受诊断路径改成显式节点选项并加入身份校验、报告和缓存边界；RGB24、取尾、resize、VAE encode 的数值顺序须保持。测试不包含需要模型权重的隐式下载。

## 上游参考与差异

- [latent Upscaler Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus)：研究其逐块状态、目标条件与遮罩处理；不是用它替换已有放大器。
- [LongMedia two-pass guide](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/blob/main/docs/TWO_PASS_LATENT_HIRES_REFINER_GUIDE.md)：同样存在 low-x0 续接路线，不能用本例指责其实现普遍错误。
- [H3 Continuation](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation)：参考已完成 AV 尾部的原生续接；其 two-pass 含义不等同本节点的低分辨率＋高分辨率细化。

以上是设计参考。本修复的验收依据是固定条件实验、阶段身份和用户完整审片，不是“别的仓库这么写，所以应该有效”。
