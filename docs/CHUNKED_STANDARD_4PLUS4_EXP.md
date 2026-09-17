# 非 PDD 标准 4＋4 时间分块（Advanced EXP）

这是独立的新执行合同，不修改已验收的双模型长视频循环，也不改变旧 Chunked Plan 的缺省行为。
必须使用包含本次实现的新代码并重启 ComfyUI；旧版本只有 JSON 没有实现，不能运行此模式。
本次实现随 GitHub 源码更新，不另发 Registry 版本；历史实机证据保持原范围。

## 用户自定义组合图（额外发布，EXP）

用户要求将 `2026-09-17_H3_Chunked_4plus4_接线修正版（低显存双分块双采样）.json`
一并公开。它不是下面纯净0.401MP／8秒图的另一份同参数已验收成片。

- 保留 Ref2VA INT8 底模、所选 FL2V Turbo alpha8 LoRA 强度0.7，以及
  `T8 ChunkFFN → KJ Memory Efficient Sage → ModelAttentionBackend → SolAttnMiniMax → LoRA`
  的自定义链。跨任务底模／LoRA搭配和组合执行效果由使用者判断，不硬性禁止，也不认证全部生效。
- 需安装 `ComfyUI-KJNodes`、`ComfyUI_Comfyroll_CustomNodes`、`comfyui-various`、
  `ComfyUI-Easy-Use`，及与本机版本匹配的 Sage/Sol依赖；本项目不安装这些第三方组件。
  新 Core 自带 `ResolutionSelector`、`ComfyMathExpression` 与 `ModelAttentionBackend`。
- LOW 尺寸来自 ResolutionSelector：16:9、0.4MP、32像素倍数，实际864×480。
  Plan `scale_by=2` 算得1728×960，面积约1.659MP，而不是整条输出0.4MP。
  修改LOW面积或倍率时看 Plan 的INT输出，HIGH宽高已连接，不用手改隐藏旧值。
- 时长来自 `JWFloat=5` 与17n+5对齐公式，得到124帧／24fps≈5.17秒；
  `temporal_chunk_frames=136` 下只有一个二采窗，一采4＋二采4＝8次前向。
  输入8秒会得到192帧／两窗／12次前向；增加视频时间分块与模型FFN分块是两件事。
- LOW／HIGH实际都消费 `CR Prompt Text` 的同一个文本；面板中旧缓存文本不代表另一套生效提示词。
  改说话内容只改这个共享提示词。图片 `10A.jpg` 是占位文件名，需用户自行选择有使用权的参考图，
  本次不上传人物图片、模型或任何私人媒体。
- 保留原参数与接线，仅纠正旧备注、语义相同的Ref2VA控件值和当前Core保存控件序列。
  未连线的第二个ResolutionSelector没有作用，可保留作辅助控件。

CPU实际Core API及外部原类schema检查不等于执行第三方patch／GPU内核，也不代表完整音画合格。
外部组合可能覆盖另一节点的优化；看运行警告、实际覆盖与成片，不依据节点串联就认定全部已生效。

## 与旧路线的区别

| Plan 合同 | 一采 | 二采 | 最终音频 |
| --- | --- | --- | --- |
| 原 Plan 默认 `video_only_legacy` | 沿用旧输入 | 沿用旧视频分块执行 | 原输入音频透传 |
| 旧低 sigma v3/v4 示例 | 完整8步至0 | 3步、denoise0.30 | 原输入音频透传 |
| 新 `standard_joint_4plus4_exp` | simple8前4步的denoised_output | 原始后4步，逐窗联合AV | 二采完成音频 |

新合同只开放完整空间画布 `full_frame_safe` 和 T8 双时钟 Euler，不开放独立空间tile。
Plan 原有15个参数位置和默认行为保留，新字段追加在末尾，不增加节点ID或改注册顺序。

## 正确接线

底模 → 正确alpha8 Turbo LoRA → Dual Clock的MODEL → Learned Two-Pass Parity Plan。
保留 `base_steps=8 / coarse_steps=4 / refine_steps=4 / shift_video=12 / shift_audio=3`。

- Parity `coarse_sigmas` → 第一采样器；第一采样器 **denoised_output** → Chunked Upscale `latent`。
- Parity `refine_sigmas` → Chunked Upscale `sigmas`，不另建低denoise调度器。
- Parity `report_json` → Chunked Plan `parity_report_json`，选择 `standard_joint_4plus4_exp`。
- 同一Dual Clock的MODEL和SAMPLER接Chunked Upscale；执行器按每窗放大后的AV形状重新绑定SAMPLER。
- HIGH Conditioning → Chunked Upscale `conditioning`。LOW/HIGH必须同提示词、媒体、首尾图片、时间长度，只改空间尺寸。
- 第一阶段和第二阶段连接独立RandomNoise。二采只生成一次完整目标AV噪声场，再按全片绝对坐标切片。
- Chunked Upscale的LATENT → AV Decode；使用Decode输出的二采音频保存，不旁路一采中间音频。

二采video sigma固定为 `0.9035, 0.8, 0.6316, 0.3158, 0`，恰好4个Euler区间。
Report用于验证配方与时钟，但不能证明外部用户图真的执行了一采4步；必须同时检查实际采样器接线和日志。

## 倍率、尺寸与 INT 输出

Chunked Plan 现与普通学习型放大节点共用 `learned_upscale_geometry`，不维护第二套尺寸算法。
除原手填模式外，可选择 `size_mode=scale_by`（倍率）或 `target_megapixels`（目标面积）。

- 一采 **denoised_output** 同时接 Plan 的 `source_latent` 与 Upscale 的 `latent`，Plan 从真实视频 latent 读取原始宽高，不从 HIGH 条件或手填目标推测。
- 倍率模式设 `scale_by=2.0`，例如源448×224得到896×448；`target_width/height` 在此模式不参与计算。
- Plan 新增 `width`、`height` 两个 **INT** 输出（原输出0/1不变），连接 HIGH Conditioning 的宽、高输入。先将其 `width/height` 控件转换为输入；最终二采条件与 Plan 使用同一计算结果，避免另行手填尺寸。
- 倍率范围与普通学习型节点一致：1～4。按原比例选取32像素对齐尺寸，并检查不缩小、任何单轴不超过4倍、非等比误差限制；不是将宽高分别随意取整。非整数倍率以输出 INT／`report_json.geometry` 为准。
- `target_megapixels` 按原比例与目标面积计算；`target_dimensions + preserve_source` 按手填面积保持源比例，`honor_dimensions_exp` 才按手填宽高分别对齐并检查 `max_anisotropy`（默认1.05）。

旧图仍默认 `target_dimensions`，不接 `source_latent` 时保留原手填尺寸及旧Plan内容；因此旧图无需额外连接源。
倍率／面积模式必须连接源。执行 Upscale 前会复核其输入的源尺寸与 Plan 记录相符，防止错误源或目标尺寸漂移。
输出仅为像素宽高，不携带 LATENT；Plan 不保存源张量，也不改变音频、sigma、步数或时间分块。
`tile_width/height` 不是最终画布，标准4＋4的 `full_frame_safe` 不用它们分割空间。
最新正式示例已连好源与 HIGH 宽高，调整 LOW 尺寸或倍率无需再维护 HIGH 的第二套尺寸。

## 时间分块的目的及计算量

3D学习型放大器先对整个联合输入的视频部分只放大一次，保持音频不变。
随后高分Transformer只看到当前时间窗，而非整段时间轴，旨在降低该阶段激活峰值。
它不是模型内部的MLP/FFN分块；无需因此追加那两个MODEL分块节点。
整段高分latent、噪声场、已完成输出、3D放大及模型权重仍占内存，不保证16GB安全、一定更快或更清晰。

本例448×224 → 896×448（0.401MP）、192帧／24fps＝8秒，窗口136帧、重叠34帧。
H3的5-token／17-frame网格产生两窗：[0,136)、[102,192)。
共享一采4次 + 每窗二采4次 = **12次模型前向**。
“4＋4”是每个区域的两阶段配方，**不是多窗整片总计8次**。
单窗才是整片4＋4共8次；缩窗增加重叠计算和独立预测轨迹。

## 接缝、音频和保护措施

后窗读取前窗已经完成二采的重叠video/audio latent，并将两路该区域noise_mask置0作为只读上下文。
执行后严格检查该前缀逐值相同，拼接时保留前窗结果，只追加后窗的新尾部。
原生Euler的最后一次float32加减本身可能产生几个ULP的误差。新路线在检查前只允许
`8*float32_eps*(1+abs(source))` 的逐元素界限，并精确恢复零mask值；大于该界限直接报错。
部分mask及可编辑值不改写。每窗报告恢复前最大误差和受影响值数，不靠全局大阈值隐藏真实漂移。
没有latent端点平移、RGB/latent叠化或音频增益，也不会把尚未完成的一采音频硬锁为全片最终音频。
全局噪声只生成一次；源latent不原地修改；原输入两路合法noise_mask继承；最后音频padding完整保留。

输入NaN/Inf、错误HIGH参考尺寸、错误sigma、错误双时钟、未知时间元数据、分块时间缺口或只读区域变化会明确拒绝。
执行过程中继续响应取消，不引入持久阶段缓存；本节点没有 `resume_existing` 功能。
已验收长视频循环中的22/39帧上下文参数并未改动；此节点使用自己的17帧网格overlap，不要将两个参数等同。

`anchor_strength`、`overlap_blend`和空间tile参数仍为旧合同保留，在新标准合同中不参与拼接。
上下文由 `temporal_overlap_frames` 控制；多窗要求非零重叠。不能靠增大重叠承诺自动消除内容漂移。

## 验证范围

必须区分CPU机制检查、整模型GPU完成／严格解码、以及完整人工音画质量验收。
本次新合同仍为Advanced EXP，不能仅凭mask相同或可解码宣称接缝、人脸、后段或声音正常。
EAV、Prompt Relay、Sol、额外串联LoRA及未知第三方MODEL patch不在本次组合资格承诺内。
旧接受样例、旧v1/v2/v3/v4路线和双模型长视频接缝策略不迁移。

回归：`tests/test_chunked_two_pass_parity.py`、旧Chunked全局噪声测试、Learned Upscale测试，
以及 `docs/DUAL_MODEL_SEAM_FIX_20260913.md` 指定的5组旧双模型防回归测试。
独立GPU工具：`tools/run_chunked_parity_probe.py`；工具只拥有隔离测试服务，保留失败证据，GPU串行且有资源保护。
GPU工具显式追加KJ全局Sage selector；正式JSON保留普通MODEL路线，两者均用原生H264保存。
GPUv2最初从VHS模板提取采样图并替换为原生保存；最终正式JSON同步为实测原生保存。
节点ID变化不改变执行合同；普通MODEL路线未额外运行GPU对照，不能冒称后端完全相同。

首轮GPU在第二窗触发严格逐值mask检查，未输出成片，失败记录保留。
CPU调用真实双时钟Euler并让微型model返回完全精确的锁定x0，200个保护值中34个发生变化，
最大绝对误差为一个float32 epsilon（1.1920928955078125e-7）；这验证了舍入机制，
不以CPU机制复现替代整模型误差记录或成片审查。

## 本地实机结果（2026-09-17，非人审验收）

最终数值执行路径的0.401MP、8秒两窗GPUv2已完成：896×448、24fps、192帧，H264视频和AAC音频均为8秒，
视频／音频／联合流分别严格完整解码通过。执行耗时约173.55秒，周期性整卡观察最低空闲约4.34GiB，
不是精确峰值或进程独占显存；不是与整段二采的显存／速度A/B。
WebSocket记录为一采4次、窗口一4次、窗口二4次采样回调，sigma与声明一致，3D放大调用一次。
后窗video376320个只读值中23038个发生舍入差，最大2.3841858e-7；audio3648个只读值中524个发生舍入差，
最大1.1920929e-7。全部通过逐元素小界限检查后精确恢复，未改可编辑部分。
CPU指定回归164项通过，注册339个唯一节点，旧位置断言保留；没有跑全仓或所有第三方组合。
最终正式JSON另经真实Core CPU验证：22个执行节点、38条边、75个显式参数，API验证通过，
未创建CUDA上下文或提交推理。保存节点采用当前原生DynamicCombo四控件序列，与已验收V2示例一致；
API嵌套format/codec/encoding结构也已验证。该检查不是原生浏览器导入／导出证明。

证据位于本地 `artifacts/development/chunked-standard4plus4-20260917/gpu-0p4mp-8s-v2`。
整片SHA256：`730c778286eac9a16eb9a22ebf4bb2945c3062799e52d2e378ec7c0f0c2aa51e`。
前一轮失败仍保留为v1，不交付为示例。测试服务已退出，无拥有的残留子进程。
人物、口型、时间窗内容漂移及接缝还需要完整人审，不根据边界抽帧宣称普遍质量通过。

### 倍率／输出补齐的本地检查（2026-09-17）

本轮只复用现有空间尺寸计算并补接线，不修改数值采样器、3D网络、上下文、音频或接缝策略。
195项指定CPU回归通过，覆盖倍率、目标面积／尺寸、32对齐、旧Plan兼容、实际INT输出、尺寸漂移检查、
二采音频／12次前向不变，以及上述5组双模型防回归门禁。
更新后的正式JSON经真实Core校验：340个节点ID、22个执行节点／41条边／78个显式参数，
API及序列化检查通过；无CUDA、未排队，不是浏览器导入／导出或本轮新增GPU画质验收。
2×示例仍计算为896×448；历史GPUv2证据不改写为新版JSON的额外GPU实跑。
证据保存在本地 `artifacts/development/chunked-plan-scale-20260917`。

## 旧测试副本覆盖正式节点的报错

若出现 `build_chunked_two_pass_plan() got an unexpected keyword argument 'parity_report_json'`，
先核对堆栈路径及 `/object_info/MiniMaxH3ChunkedTwoPassPlanT8Advanced` 的 `python_module`。
2026-09-17用户实例8189实际返回 `custom_nodes._minimax_h3_release_test_20260913`，只提供旧15个输入和2个输出；
该副本的builder不含 `sampling_contract/parity_report_json`，新工作流参数被传给旧函数因此报TypeError。
这不是Sage／Sol或倍率导致的错误，不能靠删除Parity参数退回旧执行路线来“修复”标准4＋4。

正式安装仅保留一份本项目节点。关闭ComfyUI后，可将旧测试副本目录
`_minimax_h3_release_test_20260913` 改名为 `_minimax_h3_release_test_20260913.disabled` 或移出custom_nodes；
开头下划线不代表禁用，当前Core扫描仅跳过 `.disabled` 后缀。
然后重启ComfyUI、刷新网页并重新导入更新示例；核对API来源为正式 `minimax-h3-audio-T8`，
有尺寸模式／源LATENT以及 `width/height` INT输出后再运行。
初次诊断只读核验了用户实例。随后用户要求继续收尾，已将这个精确旧副本改名为 `.disabled`，
没有删除内容或移动核心目录。改名前8189已不监听；未停止、启动或重启用户服务，也未提交任务。
合并两个计算分块节点／实际KJ微型组合／Sol注册／倍率／Relay缓存及接缝回归后，322项CPU通过，
新一轮正式Core340注册和22节点／41边／78参数校验通过。下一步须启动／重启真实用户实例，
确认API已指向正式目录；不能把隔离CPU资格检查写成用户实例已重启或本轮整模型GPU通过。
