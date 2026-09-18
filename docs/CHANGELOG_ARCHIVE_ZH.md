# Historical README release entries

历史记录，仅用于溯源；旧版本的待开发／未发布状态不代表当前版本。当前状态以首页及最新发布说明为准。

# MiniMax H3 Audio T8

简体中文 | [English](../README_EN.md)

2026-09-18 **v1.85.0：Meridian、Avatar、音色／情绪与可选采样预览**。
指定样片已获用户验收，含直接狂怒声音与修正后的平行横移；正式工作流保存在节点目录：
[Avatar／原生声音／双采／长视频](../examples/workflows/36-avatar-voice/README.md)、
[Meridian 运镜与源时间编辑](../examples/workflows/37-meridian/README.md)、
[来源诊断与可选开头音频处理](../examples/workflows/38-diagnostics-preview/README.md)。
Meridian 使用独立 merged-DMD ConvRot INT8、授权 Omega 几何和四个节点，不用原全精度模型跑测试。
TAEH3 预览支持时序与独立2D tiny，只读观察LOW x0及定向取消，不修改采样数学。
旧接线／默认和已验收接缝配方不迁移；接缝NA不当作通过，不承诺精确声纹／字词时间、通用16GB性能。
已复现的Core同进程LoRA驻留差异仍为已知边界。更新后自行重启ComfyUI，详见[版本说明](RELEASE_1.85.0.md)。

2026-09-17 **H16 已完成部分：GitHub 源码更新，版本仍为1.84.0**。
官方 ConvRot INT8 VAE 使用原生 Load VAE；新增 H3→LTX 标准视频 LATENT Adapter，
已有 Prepared 两节点增加 Tao 两请求接续。四组八片人审通过，取消／故障后普通 H3 恢复也完成。
[安装、接线与范围](H16_SOURCE_UPDATE_20260917.md)；
[Adapter 转换保存模板](../examples/workflows/35-h3-ltx-latent/README.md)。
仍为带外部资产前提的 EXP，不是自动下载／一键精修或任意16GB配置保证。
H16-3 与 Meridian 未发布；不修改已验收双采、Topaz、Sol／Sage 或旧图。更新后重启 ComfyUI。

2026-09-17 **v1.84.0：Semantic Bridge / BUNNY 与验收双采工作流**：
两个独立节点支持普通条件、Prompt Relay、单模型内循环及双模型分阶段配置。
[模型下载与安装](https://huggingface.co/t8star/Semantic-Bridge-Comfy)：保留
`ComfyUI/models/semantic_bridge/t8_compat/` 目录。默认强度0.10，关闭或0完全旁路；
未连接的新输入不迁移旧工作流，不串联两份Bridge。重启ComfyUI后使用
[五份示例](../examples/workflows/34-semantic-bridge/README.md)，详见
[接线、参数与验证边界](SEMANTIC_BRIDGE_EXP.md)。模型不是LoRA或提速器；
指定两段8秒双采已验收：640×320→896×448，4+学习放大+4，两份Bridge均0.10；
[正式推荐工作流](../examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json)。
其他素材、歌唱及参考声音仍需自行验证；不迁移旧图，不带入H16／Meridian。
详见[发布说明](RELEASE_1.84.0.md)，Registry可用状态以发布服务实际结果为准。

2026-09-17 GitHub 源码更新（不另发 Registry 版本）：合入下述已完成修复，保留线上已有
Topaz／FastH3 V2／Qwen 缓存身份改进；未完成的其他任务与本地交接文件不进入提交。
本次更新的范围和后续部署门禁见[源码同步说明](GITHUB_SOURCE_SYNC_20260917.md)。
以下 CPU／GPU 数字各自绑定原测试范围，不合并冒充整仓或全组合画质认证。

2026-09-17 接口修复：SelfLift 正式执行器补齐真实 LOW 断点、
producer 身份、独立 HIGH MODEL 和逐阶段 TST／EAV／Prompt Relay 执行。
此前单列的5项接口错误已修复，不是忽略未知参数；旧默认路线及已验收接缝不迁移。
前置 Sage／Sol／LoRA 保留，未验证组合只警告，真实错误仍传出。
最终受影响 CPU 792 项通过、2 个子进程 worker 入口条件跳过；现有 EAV／TST／
Guide Mean 图通过实际 Core 校验，不是新增整模型 GPU 画质验收。
需重启 ComfyUI 加载本地改动。
原因、用法与核验边界见[SelfLift 接口修复](SELFLIFT_INTERFACE_REPAIR_20260917.md)。

2026-09-17 策略更新：全部自有节点不再仅因已有／后加 LoRA、
KJ Sage、SageAttention、Sol 或其他 callable 补丁、组合未验证而硬性禁止运行。
保留／委托原补丁，覆盖不足明确报告，风险由用户承担；真实坏输入、内核异常、
配对和缓存完整性检查保留。未知组合不能跨运行误用旧缓存。
使用及测试边界见[补丁组合策略](PATCH_STACK_POLICY.md)，开发规则已保存到 SKILL.md。
这不是全组合质量认证，旧默认采样／放大／音频／接缝配方不迁移。

2026-09-17 更新：`Chunked Two-Pass Plan` 补齐倍率／目标面积模式，
从一采源 LATENT 读取原尺寸，复用普通学习型放大节点的比例与32像素对齐规则；
新增 INT `width/height` 输出，可连接 HIGH 条件自动同步尺寸。
[非PDD标准4＋4时间分块示例](../examples/workflows/13-latent-upscale/2026-09-17_H3_NonPDD_Standard_4plus4_Chunked_EXP.json)
已连好2×倍率，采样和接缝算法不变；使用需重启，详见[接线说明](CHUNKED_STANDARD_4PLUS4_EXP.md)。

按用户要求一并发布[私人改装工作流（EXP）](../examples/workflows/13-latent-upscale/2026-09-17_H3_Chunked_4plus4_接线修正版（低显存双分块双采样）.json)：
保留Ref2VA、Turbo0.7及KJ/Sol/LoRA链；依赖、实际2倍尺寸和时长已写入画布备注。
用户组合不被硬性禁止，但不借用纯净示例的GPU画质资格。

2026-09-17 更新：独立 `Chunk FeedForward` 和 `MLP Activation Chunk`
取消对已有 KJ／Sage／Sol、block/attention/MLP owner 的兼容性硬拦截，改为警告并保留／
委托调用，组合风险由使用者承担；输入有效性和严格缓存／恢复门禁不变。现有
`sol_attn_minimax_v2.py` 原样直接注册为 `Patch Sol-Attn (MiniMax)`，不改实现或参数。
重启 ComfyUI 生效，详见[节点说明](H3_MEMORY_NODES_EXP.md)。

2026-09-17 **v1.83.0**：新增三个独立 FastH3 V2 EXP 节点，支持完整学生模型、固定八步 AV 配方、learned-gate VSA、T8 内存桥接与双 MODEL 4+放大+4 内循环。训练配方／官方模板及两条 8 秒循环已获指定范围验收；Sol 非 video Q/KV 保护后的 B 也已明确验收，旧失败 A 不进入推荐。六份通过配方保存在[加速工作流目录](../examples/workflows/10-speed/FAST_H3_V2_README.md)，真实原生 UI/API 往返已核验。移除人为帧数上限，但必要对齐、上下文与实际内存约束保留。旧节点和工作流不迁移。固定 832×480／73 帧冷／热对照中，旧 EMA-B 整图 103.35／99.90 秒，V2 为 78.20／71.92 秒；仅这组更快，显存观察未下降，不是同模型或等画质证明。详见[版本说明](RELEASE_1.83.0.md)和[模型、连接与兼容边界](FAST_H3_V2_EXP.md)。

2026-09-16 **v1.82.0**：完成 Core H3 VAE、语音／时间线、缓存身份、PDD 生命周期与前端工作流往返加固；新增一份不依赖 KJNodes 的双模型 4+4 长视频内循环工作流，让 LOW/HIGH 各自连接 T8 `LowVRAM(head_chunks=4)` 与 `ChunkFFN(chunks=2)`，并保留 Prompt Relay。两段共 8 秒真实 GPU 机械验证通过；用户已淘汰接缝明显的 `head_chunks=1`，并确认旧 HIGH 硬边界会在人物正常时令背景突然更换。当前工作流改用 `accepted_picture_low_context_v1 + high_native_mask_ramp_exp`：精确锁定 HIGH 前缀后，再用三个 latent 单元按 `0.25→0.50→0.75` 渐进释放，音频不改。用户复核认为背景连续性明显改善，但仍有轻微颜色跳变；核查确认原 Color Match V2 已执行，残余主要是续段第2／3帧的短促偏暗—回亮。节点因此追加可选 `bounded_spatial_temporal_exp`，只稳定续段开头12帧的低频RGB均值，不混帧、不改结构、不碰音频；旧工作流仍保持 `bounded_spatial_v2`。CPU复用成片的 A/B 已将该峰值降低约85.4%。进一步的 `bounded_motion_color_exp` 仅修正可信运动对应的局部低频颜色异常；用户已确认 C 验收通过并要求发布，轻微接缝变色保留为已知限制。新增推荐图保存通过的 2:3 首帧控制组合（LOW256×384→HIGH512×768、h4+c2、4+4、8秒）；旧图和节点默认不迁移。见[局部修色说明](MOTION_COLOR_EXP.md)。不作通用 16GiB、省显存、提速或画质保证。详见[更新说明](RELEASE_1.82.0.md)。

2026-09-15 **v1.81.0**：新增两个互不依赖 KJNodes 的 H3 低显存 EXP 节点：按头分组并提前释放中间量的 `Low VRAM Attention`，以及仅在长 packed token 序列上启用的 `Chunk FeedForward`。固定 3 秒短片中，默认 `head_chunks=4 + chunks=2` 相对基线少用约 306.73 MiB 峰值显存（2.06%），耗时增加约 2.92%；不是通用显存或提速承诺。Speech Studio 新建参考音色工作流会自动保守裁掉低能量对齐留白，旧工作流保存的显式设置不变；Sol 验证探针同步适配当前 `sink_blocks` 接口。详见[更新说明](RELEASE_1.81.0.md)。

2026-09-14 **v1.80.0**：新增 SelfLift LOW4→学习型 latent 放大→HIGH4、独立 LOW/HIGH MODEL、TST 和节点内长视频组合，并交付 9 份已完成绑定样片验收的 SelfLift/TaoMate EXP 工作流。EAV 默认采用已审的 `tau=8、15%–90%`；想要更高动态可逐步提高，但必须复查身份、形变、闪烁和声音。所有结论仅限绑定短片，不作通用提速、显存或画质承诺。详见[更新说明](RELEASE_1.80.0.md)。

2026-09-14 **v1.79.6**：完成星光以外的正式Topaz闭环。14个高清模型和5个插帧模型均已走通真实官方生产worker；新增明确的10-bit SDR→HEVC Main10保持路径、多卡GPU序号和“先高清再插帧”串联工作流，并修复无损母版被8-bit审计误拒绝。HDR、VFR、隔行和旋转元数据仍明确拒绝，不静默转换；商业权重只下载到本机，不随仓库发布。详见[更新说明](RELEASE_1.79.6.md)。

2026-09-14 **v1.79.5**：继续修复Topaz交付边界。手动参数现在按模型定义过滤，固定预设模型不再因不支持的滑块报错；权重检查遵循正式网络模板，修复Rhea、Nyx XL和Theia误报未下载；运行前按音频编码选择MP4/MKV；10bit/HDR不再静默降成8bit；ComfyUI进度条显示推理帧与审计阶段。Apollo已完成真实短片机械验证，画质仍待人审。详见[更新说明](RELEASE_1.79.5.md)。

2026-09-14 **v1.79.4**：修复正式Topaz把15秒视频保存成12.9GB无损MOV、再接SaveVideo写满磁盘的问题。高清节点现在默认一次性GPU编码高质量H.264 MP4，禁止再接SaveVideo；超大无损母版保留为高级审计选项。新增模型用途说明、自动/手动参数及独立中文滑块，并新增正式`tvai_fi` 2x/4x插帧节点和工作流。使用时无需保持Topaz软件打开，但须已登录并先下载模型。详见[更新说明](RELEASE_1.79.4.md)。

2026-09-13 GitHub 主线更新：指定LTX成片、Tao新对白5秒恢复片及最新[Dance 4+4／前段成片LOW上下文8秒样片](DANCE_ACCEPTED_PICTURE_20260913.md)均已完整人审接受，不重复生成或审片。Dance正确示例与接缝防回归说明已保存；旧Dance与两条深度失败实验没有进入正式工作流。Tao原生成任务收尾触发内存保护仍保留失败，媒体接受不等于新封装完整GPU可靠性认证。[可选深度参考](DEPTH_REFERENCE_EXP.md)仅作负实验说明；另见[资源清理、AdaLN、HJL与音频边界](LOCAL_RELIABILITY_SCOPE_20260913.md)。

本次更新 **v1.79.0**：新增[双模型4＋潜空间放大＋4内循环](DUAL_MODEL_LONG_VIDEO_EXP.md)、[正式Topaz高清后处理](TOPAZ_EXP.md)和[R1可靠性修复](R1_RELIABILITY_20260911.md)。新双模型模板默认两段8秒，已评样片的画面、声音、口型和接缝可接受，旧工作流不变。OpenVDN不再因上游Sol的注意力override单独报错；VDN仍使用自身注意力计算，不宣称两种算法叠加提速。星光暂停，未作为已完成能力发布。详见[更新说明](RELEASE_1.79.0.md)。

这是一个面向 MiniMax H3 的 ComfyUI 节点包。它不只做文生视频，还把图生视频、首尾帧、参考图、参考音频、长视频、口型、加速和成片修复整理成可以直接使用的工作流。

当前版本：**1.85.0** · 354 个节点 · GPL-3.0-or-later

1.78.1 整理了仓库目录：实现代码集中到 `h3_t8/`，首页不再堆满 Python 文件。旧工作流、模型位置、节点参数和三个 TRT 命令入口保持不变；不需要重新下载模型。详见 [目录结构与更新说明](REPOSITORY_LAYOUT.md)。

1.78.0 新增的 [TRT VAE 可选后端](TRT_VAE_EXP.md) 和 [5 份工作流](../examples/workflows/30-trt-vae)继续保留：安装检查、本机编译、仅视频解码、完整编解码，以及同潜空间双路对照。已审短片和文字对照获得整体认可，保留 EXP；长片未验证。需要独立 TensorRT 环境和本机编译引擎，旧工作流、生成模型和音频 VAE 不变。

**不保证整流程更快。** 本机短片包含加载、解码和保存时，原生约 30.54 秒、TRT 约 31.33 秒，没有总耗时收益；不能把裸解码提速当成整条生成提速。安装与限制见 [1.78.0 发布说明](RELEASE_1.78.0.md)。

1.77.0 新增的渐进首采和 DLSS 2x 插帧仍作为独立 EXP 节点提供，不替换旧工作流。视频扩画默认联合解码，部分素材可能出现条带、重复纹理或接缝，不承诺无缝扩画。

[渐进首采 T2VA / I2VA](../examples/workflows/28-progressive-sampling)先小画幅采 6 步，学习型潜空间放大后再采 2 步。本机约 3 秒短片热测少用约 37%～38% 端到端时间；已审画面评价相近，对白组声音和口型正常，但游戏组仍有音量和音效内容差异，32 秒路线未通过，不保证所有素材同样提速或保音。

[DLSS 视频插帧 2x](../examples/workflows/29-dlss-fi)把 24fps 变成 48fps、30fps 变成 60fps，保持时长、分辨率和原音轨；不是超分，也不加速 H3 生成。已审短片可接受，不保证所有快动作和 HUD 都无伪影。需要用户自行准备 DLSSG 运行文件和依赖，见目录 README；节点不自动下载或安装。更多变更见 [1.77.0 发布说明](RELEASE_1.77.0.md)。

