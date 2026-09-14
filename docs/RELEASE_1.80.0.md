# v1.80.0 — SelfLift 4+4、EAV/TST 与 TaoMate 实验工作流

## 完成内容

- 新增 `MiniMaxH3ProgressiveSetupEXPT8` 与 `MiniMaxH3ProgressiveLongVideoEXPT8`，支持 LOW/HIGH 两条独立 MODEL 链，把完整 8 步拆为 LOW 4 步 → 学习型 3D latent 放大 → HIGH 4 步；不是先完成 8 步后再额外采 4 步。
- 新增独立 `MiniMaxH3TSTModelEXPT8`，LOW/HIGH 可分别接入，且使用同一完整 SIGMAS。首窗 TST 与 Prompt Relay 的安装顺序已修复，未知注意力所有者仍会拒绝。
- Prompt Relay、EAV、guide 均值保持、KJ 省显存 Sage、FFN 分块和 Sol 可按已验证的组合边界接入。恢复链继续校验输入身份，修改模型、提示、时长或音频后应更换 `chain_id`。
- 新增 9 份正式前端 EXP 工作流：Core/Sage 基准、Guide Mean、TST、EAV、Sol、KJ+FFN+TST+EAV+Relay、两段 8 秒接缝，以及 TaoMate 原生 3/4 步 T2VA。
- 工作流从真实 Core `object_info` 规范化，移除了测试导出器遗留的未连接控件伪输入；节点、连线和采样参数未改变。

## EAV 默认值

EAV 工作流默认采用：

- `eav_tau = 8`
- `eav_start_video_progress = 0.15`
- `eav_end_video_progress = 0.90`
- `eav_g_hard_limit = 1.5`

绑定真人样片已通过画面、细节和环境声验收。希望更高动态的用户可以小幅逐步提高 `eav_tau`，但不要误调 TST 的 `tau`；提高后必须重新检查身份漂移、局部形变、闪烁以及联合 AV 可能带来的间接声音变化。`tau=12 / 0%–100%` 的真人样片因效果过强被否决，只作为负证据保留。

## 验收范围

- 渐进相关 CPU 回归为 46 个文件、957 项通过、3 项条件跳过、0 项失败；这不是整个仓库所有功能的总测试数。
- 9 个绑定媒体均完成机械审计和实际 Chrome 播放/定位/声音检查。两段共 8 秒的样片已确认接缝/事件、画面、音乐/人声和口型；TaoMate 3/4 步、Guide、TST、Sol 和最终 EAV 配置也在各自绑定范围内通过。
- 组合样片的画面、声音和事件衔接已通过；正式组合工作流随后仅把 EAV 控制替换为另一组已验收的 `tau=8 / 15%–90%`，没有重复运行 GPU。

## 限制

- 所有结论仅约束绑定的短片、参数和环境，不承诺所有素材、提示词、时长、插件版本或显卡都具有相同画质、速度或显存占用。
- Core `--use-sage-attention`、KJ Sage 和 Sol 是不同接入方式；遵循每个画布 NOTE，不要随意叠加多个未知注意力替换器。
- TaoMate 示例是原生单采 T2VA，不是 SelfLift 4+4，也不使用 latent 放大器。
- 第三方模型、LoRA 和插件由用户按各自许可安装；仓库不携带模型权重。Starlight 仍暂停，不在本次范围内。
