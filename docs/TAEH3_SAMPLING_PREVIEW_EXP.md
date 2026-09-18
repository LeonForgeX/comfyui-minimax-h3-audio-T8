# 一采动态预览和定向取消（v1.85.0 EXP）

正式源码已在`h3_t8`／`web`，示例见
[Avatar预览接线](../examples/workflows/36-avatar-voice/README.md)。
原采样默认与旧图不迁移；下面资格／Core驻留和2D潜帧边界全部保留。

新增可选 MODEL 节点 `MiniMaxH3TAEH3SamplingPreviewEXPT8`，接在原 MODEL／Setup 与原采样器之间；原 sampler／sigmas保持原接线，旧图无需新节点。`enabled=false`直接返回原MODEL对象。

默认模型仍为 `ComfyUI/models/vae_approx/taeh3.safetensors`，使用Core原生24通道时序TAEH3 decoder。另支持 [Kijai的2D tiny H3模型](https://huggingface.co/Kijai/MiniMax-H3-TAE/blob/main/README.md)：把其 `vae_approx/taeh3.safetensors` **另存为** `ComfyUI/models/vae_approx/taeh3_2d_kijai.safetensors`，在本节点的checkpoint中主动选择，不能覆盖默认时序文件。

本机已添加该独立文件，9,791,388字节、SHA256 `f0f60fa072089997f817402098c2fd90777cb2660dd79cf5df42fc1e3e08e527`，上游固定revision `a213ac8bf2f148b4f32372279a7f207846978900`。原时序文件SHA与Core默认首选 `taeh3.safetensors`均保持。生产代码只使用Core的无随机初始化TAE基础层，不依赖KJNodes，也不随节点分发模型。

实际Core节点schema核对：仅checkpoint options新增该2D文件，原首选、所有其他输入／隐藏参数不变；这不是“350个节点的完整schema哈希恒等”。本地`taeh3-2d-schema-v3.json`通过。v1/v2测试脚本按旧combo形式读`INPUT_TYPES`而失败，实际Core返回`COMBO`加`options`元数据；v3仅修artifact读取方式，失败收据保留，不为审计断言改变节点或Core。

2D图严格核对完整已知布局：24通道、96→64宽度转换、四次2倍放大、3通道输出；逐潜帧串行解码并检查取消。它没有时序解码上下文，显示“2D逐潜帧预览”，事件的源fps为空、索引是潜帧索引，不能把7张预览冒称连续24fps视频。面板fps只是轮播速度。其他未知结构、缺失或不匹配tiny模型只使预览不可用，原采样继续；真实取消和致命CUDA错误仍传播。

观察真实callback里的x0，不是当前噪声、不是音频。只复制连续开头前缀：时序decoder的2／7／12 latent对应约5／22／39帧，保留时序解码上下文；2Ddecoder则分别输出2／7／12张潜帧近似图，不插值补成源视频帧数。默认7、最长边256、每2步更新、最小间隔500ms、最多12张JPEG、12fps播放。低分辨率和低频率可减少开销；没有额外扩散步，但有解码／传输耗时，不能承诺零显存或零性能开销。

LOW/HIGH、段号、累计步从实际生产者取值，不能从尺寸猜。原生单采或其他未标记sampler明确显示`unidentified`，不能把它冒称LOW。双采累计步不改变sigma表或原时间坐标。预览只看开头片段，不代表整片动作、最终VAE画质、音频或接缝质量。

## 取消和隔离

私有事件绑定client、prompt、preview node、sampler node、run、单调序列。迟到的旧帧／结束事件／取消回复不能覆盖新请求；节点移除时清理自己的计时器和监听器。面板暂停只暂停预览播放，不暂停采样。

取消调用Core `/api/jobs/{prompt_id}/cancel`，原子地取消该请求。不会调用全局`/interrupt`、清队列、杀进程或关闭服务。旧Core若没有按请求接口，面板明确说明失败，请用原生取消按钮，不自动退回全局取消。按下按钮不等于已经停止，等待真正中断／结束事件。

没有执行上下文／client时直接委托原采样器，不解码或广播。原callback先执行，参数对象／次数／顺序不变。真实采样错误／取消传播；普通预览文件或传输问题关闭本轮observer。缓存命中没有callback时不制造“实时”帧，已有画面标为此前请求。

## 回归范围

最新增量：93项CPU小范围通过，真实2D与安装的KJ解码数学逐值相同，原时序与旧Core解码逐值相同，输入／CPU RNG不变。独立2D GPU仅3.406秒exit0/cleanup0，验证float16同contiguous布局逐值一致、逐帧正常取消和恢复、CPU/CUDA RNG与输入不变；没有H3扩散前向，不冒称完整预训练采样资格。真实已安装2D权重加tiny原生Euler4+4的CPU关／开比较，完整AV逐值一致，三个实际LOW解码更新，原callback0..7、私有事件和无源fps标签通过；4.891秒exit0/cleanup0。其v1仅测试脚本误导入外部tests包失败、未采样，原证据保留，v2明确绝对路径加载本项目conftest；未为该失败修改生产代码。

五项原范围加2D／Avatar effects扩展618CPU通过，2143份源快照无变化、CUDAfalse、无跳过／取消选择；涵盖原五项接缝回归。JS真实源fake-DOM明确区分2D潜帧和时序24fps标签，旧定向取消／迟到事件／清理测试保持。下方旧完整H3 GPU资格仍为当时源码绑定：本次增加的是解码分派和面板说明，不重新声称当前源完整模型GPU或所有组合已通过，也不覆盖Core驻留边界。

本地证据：`artifacts/five-track-development-20260918/taeh3-2d-compat-v1/report.json`、`taeh3-2d-gpu-v1/report.json`、`taeh3-2d-native-integration-cpu-v2/report.json`和`five-track-cpu-final-v21/terminal.json`。

历史310范围CPU通过，覆盖真实已安装tiny decoder、原生Euler最终AV逐值一致、阶段标签、开关恒等、私有事件、取消恢复、JPEG显示比例，以及五项旧双采接缝回归和诊断工具入口保护。数量是范围回归，不是全仓资格。小预览latent网格的偶数取整可能改变宽高比，显示JPEG会恢复源画幅比例；这不修改采样x0或正式视频。只有精确自有只读wrapper可在内容缓存身份中投影，伪造同名槽／未知外部callable不被隐藏。

历史GPU v6已收到实际LOW解码帧、中断并恢复8次前向，但最终视频latent差异2.1114254、音频差异0，**该轮失败保留**。同缓存OFF／OFF／ON逐步诊断用于区分重复运行状态和observer影响；不能把媒体能解码写成该轮只读通过。新独立进程qualification另列如下，限定配方，不覆盖旧恢复失败或用户未知组合。最终质量及预览易用性仍分别记录，不把预览画面当成最终成片验收。未发布，不替用户重启前端。

增量诊断已完成：OFF1/OFF2最终video差异2.19877195，第一步输入／文本／参考
相同，但原生前向video输出差异0.392578125；该差异早于任何预览解码。
TAEH3构造及三次解码未改变CPU/CUDA随机状态或输入前缀。同一加载驻留状态
下立即重复三次原生前向则AV逐值相同，不能直接归咎通用kernel随机性。
跨采样驻留／恢复差异尚未定位，原失败证据不覆盖。

### 追加定位：实际权重路径变化（2026-09-18）

两个新自有GPU诊断均是OFF/OFF，每条完整LOW4/HIGH4；有效作业分别215.046秒、284.813秒，exit0/cleanup0。首步输入、完整可观察参数、文本和参考相同，仍复现首步video差0.392578125和最终video差2.198771954，audio差0。6层抽查的host/effective权重相同，不能据此推论全模型相同。

扩大到358个实际Core cast后，10处effective权重不同：`blocks.9.mlp.fc2`由BF16变INT8，`blocks.31`–`38.adaln_proj.linear`由INT8变BF16；`blocks.9.attn.out_proj`两次均BF16但内容SHA不同。另118个模块未经过此cast入口，明确不当作已查过。all-casts模式没有测全部host权重，旧收据中空字典相等不是host完整性证明。权威`taeh3-all-casts-gpu-v1/terminal.json`，6层抽查见`taeh3-residency-gpu-v3/terminal.json`。

这定位了预览关闭时仍存在的Core LoRA驻留／卸载数值路径变化，不等于完成全部因果隔离或修复同进程确定性。未修改Core、旧LoRA、sampler、sigma、VAE或接缝，不为获得相同输出强制统一权重格式或驻留策略。fresh OFF/ON资格仍成立；旧恢复失败仍保留。两个原始诊断终态沿用旧通用`three_matched...`状态名，实际`cases`均为2，不能读成3条或数值一致；后续工具状态已按真实模式区分，有CPU测试。

全新自有进程OFF/ON完整预训练模型4+4对照已完成：八步的输入／输出、
文本／参考／阶段上下文以及最终video/audio latent全部逐值相同（最大差0/0）。
ON有三次真实TAEH3解码更新，输入前缀及CPU/CUDA RNG不变；源画幅显示为2:3。
权威 `taeh3-fresh-pair-audit-v1.json` 与两份fresh进程收据；ON同潜空间已用
原生VAE完整解码73帧512×768 H264和明确原录音，完整AV独立审计通过。
这些是新鲜独立进程指定配方资格，不代表同进程任意多次恢复一致性或最终画质人审通过。

真实浏览器增量：自有CPU Core8213加载当前JS扩展、回放v6实际GPU帧，
面板绑定请求取消实际产生execution_interrupted，未输出成片／文字；
同Core下一普通来源诊断请求成功，队列为空。收据
`artifacts/five-track-development-20260918/ui-probe-v1/live-ui-audit.json`。
这是live界面、私有通信和原生定向取消资格，不是该回放的新GPU推理。
原GPU真解码／取消证据另列；自有测试Core和隐藏页已退出，没有操作用户前端。
