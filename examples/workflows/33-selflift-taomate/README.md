# SelfLift 渐进双采与 TaoMate（实验）

本目录包含 9 个已经完成绑定样片验收的前端工作流。它们保留 MiniMax H3 原生音视频联合生成，并覆盖 SelfLift 的 LOW 4 步 → 学习型 3D latent 放大 → HIGH 4 步、独立 LOW/HIGH 模型链、TST、EAV、Prompt Relay、KJ/Sol 后端，以及 TaoMate 原生 3/4 步 T2VA 对照。

## 先选哪一个

- `Core_Sage`：最少附加功能的 4+4 基准；ComfyUI 启动时使用 Core 的 `--use-sage-attention`。
- `Guide_Mean`：在基准上保持 LOW guide 的逐帧/逐通道均值。
- `TST`：LOW/HIGH 各接一个独立 TST MODEL 节点；TST 的 `tau` 与 EAV 强度无关。
- `EAV`：绑定真人样片通过的默认值是 `eav_tau=8`、起始 `15%`、结束 `90%`、`g_hard_limit=1.5`。
- `Sol`：LOW/HIGH 各自接 Sol；不要再叠加另一个显式注意力替换器。
- `KJ_FFN_TST_EAV_Relay`：功能组合图，适合确认完整接线；它不是单项性能基准。
- `KJ_Relay_Two_Segment_8s`：两段共 8 秒的接缝与 Prompt Relay 示例，边界约在 5.17 秒。
- `TaoMate_3step` / `TaoMate_4step`：原生单采 T2VA 对照，不是 4+4，也不使用 latent 放大器。

## EAV 强度

正式示例默认采用较克制的 `tau=8 / 15%–90%`。如果用户希望动作或光效更强，可自行小幅逐步提高 `eav_tau`；不要误调 TST 节点自己的 `tau`。提高后必须重新检查身份漂移、局部形变、闪烁以及联合 AV 可能带来的间接声音变化。先前的 `tau=12 / 0%–100%` 真人样片因效果过强被否决，只保留为负证据。

## 验收边界

2026-09-14 的绑定短片已完成机械检查、实际 Chrome 导入/播放检查和人工审片；其中两段 8 秒样片的接缝/事件、画面、声音与口型通过，TaoMate 3/4 步、Guide、TST、Sol 及最终 EAV 配置也在各自绑定范围内通过。组合图的画面、声音和事件衔接已通过，随后仅把 EAV 控制替换为另一组已验收的 `tau=8 / 15%–90%` 配置，没有重跑 GPU。

这些结果不代表任意素材、提示词、时长、模型版本、插件组合或显卡都能获得同样画质，也不构成通用提速或省显存承诺。更改模型、提示词、时长、音频或关键参数后请换新的 `chain_id`，并重新检查完整音画；不要把其他任务的恢复缓存直接搬入当前链。

依赖的 LoRA、底模、KJ/Sol 插件及商业或第三方权重都由用户按各自许可自行安装，仓库不携带模型文件。
