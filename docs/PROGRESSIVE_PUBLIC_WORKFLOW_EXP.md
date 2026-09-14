# 渐进双模型工作流：接线、参数与验收边界（开发候选）

本页对应 `MiniMaxH3ProgressiveSetupEXPT8`、`MiniMaxH3ProgressiveLongVideoEXPT8` 和可选的 `MiniMaxH3TSTModelEXPT8`。这些接口不替换旧双 MODEL 长视频节点，不改变旧图默认值。新候选尚未完成全部完整模型组合与人审；不能据此承诺更快、显存更低或画质不退步。

## 两次采样如何各接自己的 LoRA

```text
H3 底模 → LOW 的兼容 LoRA 链 → LOW 的一个注意力后端 ─→ Setup.model
H3 底模 → HIGH 的兼容 LoRA 链 → HIGH 的一个注意力后端 → Setup.model_hires

Setup.model       → [可选 LOW TST]  → LongVideo.model
Setup.model_hires → [可选 HIGH TST] → LongVideo.model_hires
Setup.sampler                      → LongVideo.sampler
Setup.sigmas                       → LongVideo.sigmas
                    └──────────────→ 两个 TST 的 sigmas（如使用）
```

两个 LoRA 链可以从同一个加载器分支，也可以来自各自加载器，但模型结构、潜空间和音视频时钟必须匹配。节点接得上不代表不同模型任务可以混用。没有接 `model_hires` 时复用 LOW 源配置；这不是独立二采 LoRA。

`Setup.steps=8`，`LongVideo.low_evaluations=4` 表示完整时间表的前 4 步在低分辨率执行，原 3D latent upscaler 放大后继续后 4 步。不是重新开始两次完整的 4 步，也不是 8+4。两路 TST 必须连接同一张**完整** SIGMAS 表，不给 HIGH 单独截取后四步并从零计时。此路线使用原生 Euler/CFG1，不把别的采样器或三步 TaoMate 配置直接套入。

## 关键参数

| 参数 | 含义和用法 |
| --- | --- |
| `width` / `height` | HIGH 最终画布；按 32 网格设置。LOW 由 `low_scale` 和有效网格共同确定。 |
| `low_scale` | LOW 空间缩放比例；不是帧率或时长比例。896×448、0.5 对应 448×224。 |
| `upscaler_model` | 选择原有 learned 3D latent upscaler 权重；不是 Topaz、RGB 放大或自动下载入口。 |
| `guide_resize` | `legacy_bilinear` 保留原缩放方式；`preserve_mean` 是可选逐帧 guide 均值保持，不混合时间帧，不替换 HIGH 原 guide。仍需同素材画面对照。 |
| `render_window_frames` | 内部窗口为124–362帧、17n+5网格（124、141……362），不等于成片总帧数。即使输出3秒，也生成至少124帧再按目标时长裁出72帧；不能填73。 |
| `context_frames` | 续段实际选用的上下文长度，22 或 39；保存尾部容量 39 帧不意味着必须填 39。 |
| `base_seed` / `seed_policy` | 基础 seed 与分段 increment/fixed 策略；保存/重载时还需保留前端 seed 控件的运行后行为。 |
| `resume_existing` | 同一 chain 与输入不变时，校验内容身份后复用阶段/成片。关闭会拒绝已有链，不删除旧文件。 |
| `chain_id` | 本次链的独立名称。改模型、LoRA、提示、素材、后端或音频策略后使用新名称，不把另一实验的缓存搬进来。 |
| `eav_mode` | 内部组合开关：disabled 关闭、report_only 仅诊断、apply_exp 实际生效。诊断值不是画质评分。 |
| `eav_tau` / 生效窗口 | 已审工作流候选使用 tau=8、视频进度15%–90%，表现偏克制。希望更高动态时可小幅逐步提高 `eav_tau`，但数值越高越可能放大身份漂移、局部形变、闪烁或联合AV带来的间接声音变化；每次调整后都要重看完整画面和声音。不要把 TST 的 tau 当作 EAV 强度。`g_hard_limit` 仍保持1.5。节点本身保留更保守的 tau=4 默认值，不把单一样片外推为 MiniMax H3 通用最优值。 |
| TST `mode` / `tau` | 独立 MODEL 上的可选时序 Query 修正。默认关闭；tau 增大不等于更好。TST 不是作者实现的输出增益算法，具体差异见 TST 说明。 |
| `max_workspace_mib` | TST 显式临时 tensor 预算；不是整卡显存上限，也不是不会 OOM 的保证。 |

8 秒、24fps、window124/context22 的测试输出为 192 帧，接缝约在第 124 帧、5.17 秒；不是每段各 4 秒。短片通过不等于 24/32 秒长片通过。

## global、local 与时间线

- global 只写贯穿全片的人物、衣服、场景与持续声音。只说一次的台词写在对应 local 事件，不能放入 global 反复发送。
- Prompt Relay 的局部事件和时间线必须一一对应。百分比范围使用 `percent`，不要把秒数写成百分比。
- 8 秒输出 192 帧时，现有候选用 193 帧 Relay 计划覆盖；改总时长需同步检查计划覆盖长度。
- `joint_av_exp` 是联合生成对白的实验路由；锁定源音频时使用适合该策略的 `video_only_paper`，不改写已知音频区域来追求视觉效果。
- `segment_prompts_json` 是高级分段配置，不等同任意粘贴多段文字；优先使用已有 Relay 计划节点，避免维护两份相互冲突的时间表。

## Sage、KJ、Sol 不是“显示已连接就算加速”

每路使用一个已支持的注意力配置入口。Core 全局 Sage 通过 ComfyUI 的 `--use-sage-attention` 启用；本次所用 Core 的 `ModelAttentionBackend` 下拉中没有 Sage，不能手造 `sage` 枚举。也不要留下显式 PyTorch 节点覆盖全局 Sage。

KJ 普通 Sage 选择器与 H3 memory-efficient attention/FFN 是不同补丁；支持某一个不能推论另一个已执行。检查报告中的实际后端计数及 FFN 配置。LOW/HIGH 独立记录，不能只查一边。

Sol 的稀疏内核有形状和偏置约束。Relay 添加时间偏置时允许按语义回退，必须报告实际原因和调用；不能删除 Relay 偏置来伪造 Sol 加速。TST/EAV 使用内部已定义的组合顺序，不叠加未认证的外部包装。

## 音频输入和保存

`native` 无外部音轨时交付模型生成的音频。`lock_source` 需要 `drive_audio`，按源音频语义交付；`remix_source` 需要源音频参与生成，但不等同原声逐样本保留。

交付优先级必须明确：显式 `final_audio` 优先；否则 native/lock_source 且接有 `drive_audio` 时交付该源音轨；其余交付生成音轨。因此希望听新生成对白时，不要无意接入覆盖它的 `final_audio` 或 native 的 `drive_audio`。

内部的 reference_only 用于参考语义，不等于锁源；当前公共节点的 audio_mode 下拉并未提供该值，不要手改 JSON 冒充公开支持。短音轨不能靠自动补静音掩盖缺失。AAC 重编码不是 PCM bit-exact。

`audio_seam_policy=cosine_bridge`、`audio_bridge_ms=5` 指 5ms 音频交接，不是视频叠化或潜空间接缝修复。封装或保存失败应保留可验证的采样结果，恢复交付不应重新扩散；失败结果不记成功回执。

## 交付边界

当前公开接口支持 T2VA 或单首帧 I2VA，不据此宣称通用多参考/首尾帧/任意编辑兼容。旧 accepted-picture 样片的人工通过仅属于其固定配置，不自动转移到本开发候选。

每份新候选分别记录实际 LOW/HIGH 次数、LoRA/后端、源与媒体身份、完整解码和音轨，再集中人工检查画面、接缝及声音。CPU 图验证、随机小模型 CUDA 或某个诊断指标改善不能替代完整模型和人审。只有通过的候选才能晋级为推荐示例；失败与未评模板保留在开发证据目录。

相关约束：[接缝回归](DUAL_MODEL_SEAM_FIX_20260913.md)、[TST 方法差异](TST_SOURCE_CONTRACT_EXP.md)、[完整模型测试入口](PROGRESSIVE_TRAINED_VALIDATION_EXP.md)。
