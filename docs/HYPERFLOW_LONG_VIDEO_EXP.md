# HyperFlow 双采长片 EXP：隔离实现与验收门禁

## 原生单次 8 步同规格对照（2026-09-22）

为回应 0.6 MP 成片“感觉模糊”的反馈，新增**独立 EXP 节点**
`MiniMaxH3HyperFlowSingle8LongVideoEXPT8`。它保留现有长片 8 秒／两段
124+68 帧的交付、续段、音频接缝和 `1024×576` 输出，但每段在最终尺寸
直接走原生完整 HyperFlow 绝对区间 0:8；没有 LOW 采样、learned latent
upscaler 或 HIGH 重噪声再采样。它的阶段缓存单独放在
`hyperflow_single8_stages/`，不命中双采 `hyperflow_stages/` 或旧缓存。
节点只放行 T2VA/native、window124/context22、总8秒，未扩展导演台
生成配方，也不改既有双采默认。

隔离 Core 实跑证据 `artifacts/development/hyperflow-single8-long-video-9c11d32211a7/`，
与上表明亮版使用相同提示词、基础 seed `123456789`、时长、帧率、画幅及
CRF18。两段回执分别记录 `completed_network_forwards=8`、训练网格
`absolute_interval=[0,8]`、`learned_upscaler=not_used`；第二段视频条件
重新编码自第一段**已接受 MP4** 最后39帧。最终片 SHA256
`672864f971aad082a7b3543cac98b34b3320815b1efaf6463647faa773e06ce7`；
`ffprobe` 显示 192 帧、24fps、1024×576、8秒 H.264
及 32kHz 双声道 AAC，视频/音频 `ffmpeg -xerror` 全解码均退出0。稳定审片副本
`artifacts/review/2026-09-22-0p6mp-fiveview/hyperflow-0p6mp-native-single8-8s.mp4`
逐 SHA 相同。2秒与6秒抽帧可见完整客厅细节，但两种路线生成的构图不同，
**不能仅凭静帧断言谁更清晰或把机器通过当作完整真人画质验收**；用户需并排看
动态清晰度、运动和约5.17秒接缝并听音。

首次实验在任何采样前被阶段缓存的非法 `full8_output` 名称拒绝，反证保留于
`hyperflow-single8-long-video-13d5ef333147/`；改用缓存支持的 `high_output`
阶段名并保持独立命名空间，17 项定向 CPU 测试与 Ruff 通过后才实跑成功。
两次只启动/停止自有 Core，没有碰用户 8189，也没有 GitHub 推送。新实验
路线不声称比双采更快、更锐或接缝更好。

状态：2026-09-22 正式安装的隔离 Core 已完成总 8 秒／两段 GPU 成片、全音画解码、跨 Core 完成链恢复及中断后只补 HIGH 4 步；用户反馈 0.4 MP 版本偏模糊后，另完成两条 0.590 MP 实片。**尚未完成真人从接缝前看到片尾并听音的质量验收**。不能继承旧 4+4 指定样片的质量结论。

## 0.6 MP 清晰度反馈复测（2026-09-22）

用户要求 8 秒版提高到约 0.6 MP。采用 HIGH `1024×576`（589,824 像素，约 0.590 MP）、LOW `512×288`，都是 32 网格且保持精确 2 倍学习型上采样；探针和图构造器新增可选尺寸参数，旧 `896×448` 默认不变。两次均用独立自有 Core、不同唯一链，旧 0.4 MP 原片与回执未覆盖。

| 提示词 | 隔离证据 | 成片 SHA256 | 机器结果 | 视觉边界 |
| --- | --- | --- | --- | --- |
| 旧烛光暗房提示词，作同提示词对照 | `artifacts/development/hyperflow-long-video-p7-ce32f2cbc7e8/` | `1a85afacade95194be0e5a93969b47d496a600de70aa331aa77553c273815363` | Core success；两段 124+68；各 LOW/HIGH 4 次真实前向；192 帧 1024×576 H.264 和 32kHz 音频全解码 | 首末帧仍为暗场；增像素不等于照明或人审画质合格。 |
| 明亮日光客厅提示词，供清晰度审片 | `artifacts/development/hyperflow-long-video-p7-c537088c4b4b/` | `3afca940a120729db6f9b744610afb0097cfc1d1291687846258875cc6e87785` | 同规格、两段及 4+4 前向；`media-audit/report.json` 全帧/全音频解码通过 | 抽查首末帧和 frame123/124：沙发布、木桌和植物可辨，接缝两帧构图连续；**尚非完整真人观看/听音的质量通过**。 |

明亮版 prompt 为 “A single continuous cinematic camera move through a bright sunlit living room in clear late-morning daylight. Large windows evenly illuminate the entire room. Crisp visible details in the sofa fabric, wooden table grain, green plant leaves and picture frames. Natural realistic colors, stable slow movement, deep focus, clear bright exposure, quiet natural room tone. No darkness, no haze, no intentional blur.” 它和旧暗场不构成仅分辨率单变量的画质 A/B；同提示词暗场复测才是像素数对照。两条 0.6 MP 片均需用户完整审看并听音，尤其是约 5.17 秒的接缝。

## 与旧长片的边界

新节点 `MiniMaxH3HyperFlowLongVideoEXPT8` 仅接受同一完整 H3 底模派生的两条内容 LoRA MODEL 分支，HyperFlow 原始权重在专用 `hyperflow_file` 输入选择。采样固定为 LOW 绝对区间 0:4 预测干净 x0 → 现有 learned 3D latent upscaler → HIGH 绝对区间 4:8 新噪声四步。两阶段音频按原生联合 AV 继续生成；不会把未完成 LOW 声音冻结为最终音频。LOW 和 HIGH 分别有自己的内容补丁和 HyperFlow owner；不得用通用 LoRA 加载器加载 HyperFlow 原文件。

首版强制 T2VA/native、总长 8 秒、window124、context22；节点默认且本次 GPU 验证的是 0.4 MP 896×448、LOW 448×224。节点允许其它可表示的 32 网格尺寸，但那些尺寸**未获 GPU/画质资格**；没有 Relay、Bridge、EAV、Drive Audio、参考图视频或 Hybrid。需要先在独立 Core 使用 `--disable-comfy-compiler` 避免已复现的 comfy-aimdo 双 owner 原生崩溃。原权重与 3D upscaler 必须已安装。只有新 `chain_id` 才能开始，切勿复用旧双采长片链名。图构造入口为 `tools/build_hyperflow_long_video_workflow.py` 的 `build_prompt(info, chain_id=...)`；该函数不排队。独立 P7 入口：在插件根目录运行 `python tools/hyperflow_long_video_gpu_probe.py --port 8867 --timeout 3600`。它仅停止自己启动的 Core，自动生成唯一链与证据目录；成片预期在 `artifacts/development/hyperflow-long-video-p7-<token>/output/minimax_h3_t8_long_video/hf_long_video_p7_<token>/assembled/`。

旧 `h3_t8/nodes.py`、`long_video_dual_model_runner.py`、`long_video_dual_stage_cache.py` 和已接受样例没有改动。新实现放在 `h3_t8/hyperflow_long_video_exp/` 子包，根入口只追加节点注册。HR1 音画完整度修复另改了顶层 `director_batch.py`，因此旧长片源码 glob 身份相对早先快照改变；旧链在不同源码下拒绝同 ID 恢复是预期安全行为，原已生成媒体不删除。新阶段回执位于每个新链目录的 `hyperflow_stages/`，复用不可变 AV 回执的完整 SHA、形状、合同校验，但不会命中 `dual_stages/`。冻结合同包含两个裸内容 MODEL 身份、HyperFlow 原始文件 SHA 与完整 metadata、固定双时钟网格/绝对区间、上采样器 SHA、CLIP/VAE 内容、几何、LOW 尺寸、色彩匹配、音频接缝及编码参数和实际运行源码 SHA。随机 owner token 不作为跨进程身份；坏回执、变化前段成片或变化配方必须拒绝／另建链。

续段 LOW 从上一段**已接受实际 MP4 最后 39 帧**的 RGB24，经既有 resize 与 VAE 重编码，只替换 LOW 的视频条件；HIGH 继续使用原高分辨率上下文，音频沿用已完成 HIGH 的声音。已有颜色匹配及 5ms 音频 cosine bridge 可选；它们不修复结构跳切，也不构成人审。总片 8 秒分两段，192 帧，第一段 124 帧，接缝约在 5.17 秒，并非两段各 8 秒。旧通过片 SHA `3ff583bc817dd845fa288aaf0b16c2d2be24a6ce282f50c1b37a823edc91ad73` 只作回归参照，不覆盖、不作为本算法结果。

## 当前机器验证与待验收

CPU/API 测试覆盖旧注册顺序／Schema、当前顶层源码快照在注册前后不变、Core 原生 `validate_prompt`、0:4/4:8 typed sampler 标志、串行阶段与独立缓存、完成音频续接、已接受前段媒体每次校验、失败后只续 HIGH、坏回执 fail closed、非 T2VA／非 native／Relay／外部素材拒绝；旧接缝专项已纳入本轮聚焦回归。

最终源码冻结后的 GPU 成片证据：`artifacts/development/hyperflow-long-video-p7-4ce4ba4e60c8/`，独立 Core prompt `b2ae3642-954c-45a3-bc56-a713a2cac317`，两段 124+68 帧、各 LOW/HIGH 4 次真实前向，总 192 帧、896×448、H.264/AAC 32 kHz；`media-audit/report.json` 全帧/音频解码通过。成片 SHA256 `edaacd41835efabe524334c57ea89df5d02b92df8096f0f49dadd59494f9594b`。第二段 LOW 以第一段**实际接受 MP4** 的末 39 帧为来源，前段视频 SHA `512105639f750c3386d68de69ff6326e83160071bc208c7d20c0ee935fc413ef` 与阶段回执一致；未改完成音频及 HIGH 原生上下文。

跨 Core 完成链副本 `artifacts/development/hyperflow-long-video-copy-resume-76555104c05c/`：27 个不可变文件逐 SHA 保持、无采样进度、返回相同最终 MP4；另一个丢弃用副本的坏阶段张量被 `AVStageCache.load` 在采样前拒绝，这**不是**完成态 Core 全链坏缓存测试。真正中断续算副本 `artifacts/development/hyperflow-long-video-interrupted-resume-b0c27d9414a2/` 从 rev1 已接受首段及第二段 LOW/HIGH-input 回执恢复，仅一次 HIGH 4 NFE，`low_reused=true/high_reused=false`，17 个保留文件 SHA 不变，最终 MP4 与未中断原片 SHA 完全相同。两个恢复探针都只停止自有 Core，未碰用户 Core 或原证据。配置/源码改变的旧链曾被 Core 拒绝，反证保留在 `hyperflow-long-video-copy-resume-b589c0111f87/`；不能把该拒绝说成新算法失败。

本次样例 LOW/HIGH 上游内容 LoRA 均为 disabled，仅证明两个独立 MODEL 分支及 HyperFlow owner；**不**证明 P7 新长片的多内容 LoRA 组合画质。整片仍需从接缝前看到片尾并听声音，人工检查跳切、重影、人物/场景稳定、台词、音乐和接缝；机器数值通过不能写成质量通过。原接缝回归门禁见 `docs/DUAL_MODEL_SEAM_FIX_20260913.md`。权重和 ROADMAP/SKILL 不随 GitHub 源码推送。
