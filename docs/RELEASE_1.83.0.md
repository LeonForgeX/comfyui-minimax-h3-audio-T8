# v1.83.0 — FastH3 V2 与帧数上限移除

新增三个独立 EXP 节点，注册节点从336增至339；旧336节点顺序、标识和旧255份发行工作流不变。

- `FastH3 V2 · 8-Step Recipe`：完整V2学生模型、固定DMD八步／AV双时钟10+3、learned-gate VSA。Dense兼容配方和官方Comfy模板独立标注，不混称训练配方。
- `FastH3 V2 · Actual Dispatch Audit`：采样后查看实际VSA／Dense执行，配置本身不算成功执行证据。
- `FastH3 V2 · Dual MODEL4+Upscale+4 Loop`：LOW／HIGH两个裸MODEL，可分别接普通权重LoRA及T8内存节点；沿用原learned3D潜空间放大与缓存／中断恢复。不要连接已绑定latent的单MODEL Recipe输出。

模型、安装和连接详见[FastH3 V2说明](FAST_H3_V2_EXP.md)。需要匹配的原生Core H3／Kitchen稀疏接口；不安装FastVideo分布式环境，不自动下载模型，不把旧EMA／Turbo LoRA当V2权重。

## 修复与交付

- 已认证Dense Sol、无Relay分支精确保留实际packed非video Q/KV，解决指定对照中人声偏轻问题。用户明确接受修复B；旧失败A保留，不进入推荐。没有后期增益、归一化或换音轨。旧PyTorch／KJ、训练VSA、官方模板和带时间bias的Relay路由不迁移。
- 移除38项公开schema及16项共享runtime的人为视频帧数／总时长／时间位置上限。869帧参考条件回归通过；仍保留最小长度、帧对齐、真实源范围、上下文容量和用户选择的分块窗口。语音参考音频边界不改变；不是无限内存或任意长片质量保证。见[帧数规则](FRAME_LIMITS.md)。
- [六份通过原生工作流](../examples/workflows/10-speed/FAST_H3_V2_README.md)包含训练h1/c1、h4/c2、官方模板、Dense Relay双MODEL8秒、训练VSA真实中断恢复8秒及Sol保护73帧。真实原生UI导入及12份完整／API导出已核验；接受范围只限绑定样片和已评分项目。

原P0–P4本地开发门禁已收口，实际官方comfy-cli包内339节点导入及受影响371项CPU通过。此次发布准备只更改版本、说明及本地构建工具兼容版本，不重生成接受样片，不改变六份图或生产采样数学。测试范围重合不相加为全仓结果。

## 固定配方性能与限制

RTX4060Ti16GB，同prompt／seed／832×480／73帧／24fps／CFG1／编码器和VAEs。两个独立进程分别首跑cold及cache-none hot；旧EMA-B原生Euler8／shift12+3／KJ Sage，与完整V2 DMD8／shift10+3／VSA／h1c1／min_tokens0比较。

| 配方 | cold整图／采样 | hot整图／采样 |
| --- | --- | --- |
| EMA-B＋KJ Sage | 103.35／73.03秒 | 99.90／72.85秒 |
| V2训练VSA | 78.20／49.83秒 | 71.92／45.76秒 |

本组整图耗时少约24%／28%；每次真实8NFE及400次选定后端调用，四份H.264+AAC完整解码通过。整卡周期观察没有显存下降；h4/c2本机更慢，不推荐为通用提速开关。不同模型／算法，不是等画质实验；cold未清磁盘缓存，hot重建loader不保证同一MODEL对象，墙时含一致观察器开销，整卡样本含桌面活动，不是独占精确峰值。

保持EXP，不承诺普遍更快、更省显存、通用16GB安全或任意LoRA／长片画质。轻微接缝变色留后续研究；Starlight授权仍暂停。正常重启ComfyUI加载更新，选择10-speed工作流；更改模型、LoRA、后端、参考、尺寸或代码后换新chain_id，不搬旧缓存。私人参考图、测试媒体、模型权重和本地roadmap不提交或入包。
