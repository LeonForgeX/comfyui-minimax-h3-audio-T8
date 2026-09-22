# HyperFlow 导演台组合审核（2026-09-22）

本页审核的是**当前源码和现有实测证据**，不是为未跑通的组合放行。HyperFlow 原始权重必须走专用 Loader；普通加速 LoRA、标准 4+4 和独立 P7 长片不能代替导演台 HyperFlow 配方。源码入口为 `h3_t8/director_generation.py`、`h3_t8/director_hyperflow.py` 和 `h3_t8/director_sampling_settings.py`。

## 任务／声音矩阵

四种导演台配方（单 8、连续 4+4、低清 8+高采 4、低清 4+高采 4）均受同一任务门禁控制。`tests/test_hyperflow_director_combination_audit.py` 对下表除首行外的 11 种任务／声音组合逐一运行四种变体，共 44 个拒绝用例；它们在读取权重或改动图前拒绝。原有 `tests/test_director_generation.py` 另检查四种 T2VA/native 图的接线、步数和各阶段 LoRA。**CPU 编译与拒绝测试不等于 GPU 成片或人审。**

| 导演台任务 | 模型生成声音（native） | 锁定上传录音（lock_source） | 现阶段结论 |
| --- | --- | --- | --- |
| 文字 T2VA | 四种变体可编译；仅列明的尺寸／配方有隔离 GPU 音画实测 | 拒绝 | 仅 native 保留 EXP 入口，不能把不同尺寸、LoRA 或长片资格互相继承 |
| 首帧 I2VA | 拒绝 | 拒绝 | 未完成导演台 HyperFlow 首帧图和真实 GPU 验收 |
| 尾帧 L2VA | 拒绝 | 拒绝 | 未完成尾帧锚定与真实 GPU 验收 |
| 首尾 FL2VA | 拒绝 | 拒绝 | 专用手工节点的 256²／22 帧单 8 音画实跑，不等于导演台四配方通过 |
| 多参考 Ref2VA（图／视频／音色） | 拒绝 | 拒绝 | 专用手工节点的 256²／22 帧单 8 实跑，不等于导演台素材映射与参考质量通过 |
| 混合 Hybrid | 拒绝 | 拒绝 | 包括首帧＋原音驱动；未完成导演台 HyperFlow 条件与原声交付验证 |

选择“参考音色”会把已上传音频加入参考素材：无首尾帧时编译为 Ref2VA，有首尾帧时编译为 Hybrid，而非借 T2VA/native 名称绕过门禁。选择“原音驱动”会进入 lock_source／Hybrid；不能因音频已上传就视作原生生成声音。首尾帧、参考和原音现有普通采样路线不受本审核改动。

## D3、素材与采样叠加

| 组合 | 当前行为 | 要升格为可用所需证据 |
| --- | --- | --- |
| Prompt Relay、FastH3 V2 | 编译前拒绝；新增回归覆盖两条拒绝路径 | 单独实现原生条件／模型接线，Core API 校验、隔离 GPU 完整音画及时间线人审 |
| Semantic Bridge、LowVRAM Attention、ChunkFFN | 保留用户选择并明确警告；CPU 图检查过接线 | 每个开启配置的隔离 GPU 前向、成片、恢复／失败回滚与人审；当前不得称组合已认证 |
| 多内容 LoRA | LOW/HIGH 分别接链；短镜已有两种内容 LoRA 同时出片 | 逐阶段单变量对照，确认各 LoRA 实际生效和质量；P7 长片样例 LoRA 为 disabled，不能借短镜结论 |
| 软／二值 mask、EAV、Drive Audio、Hybrid、参考视频 | 无导演台 HyperFlow 完整 GPU 资格；Drive Audio／Hybrid 被任务门禁拦下 | 分别验证条件与音频时钟、实际输入、媒体全解码、失败安全和人审，不能只靠手工节点或 tiny CPU |
| P7 两段 8 秒 | 独立节点、独立缓存；不是导演台现有四配方 | 完整观看／试听接缝和人物连续性；不能把短镜 GPU 结果当 P7 或其它任务认证 |

本机历史证据（尺寸、模型、运行回执、失败样本和限制）详见 [HyperFlow 运行说明](HYPERFLOW_RUNTIME_EXP.md) 与 [P7 长片说明](HYPERFLOW_LONG_VIDEO_EXP.md)。本次审核新增的是 44 个任务／声音拒绝和 2 个 D3 拒绝回归；**没有新增 GPU 生成或用户主观验收**。审核时用户 Core 在 8189 运行，显卡剩余显存不足以安全并行完整 H3；未停止或抢占它。

后续逐项开放时，每一行都应先实现单独接线，再在独立 Core 记录精确权重 SHA、变体／尺寸／素材、真实前向次数、完整视频与音频解码、失败恢复和人工画面／声音结果。未满足这些门禁的行保持拒绝或显式 EXP 警告，不能写成“全部组合通过”。
