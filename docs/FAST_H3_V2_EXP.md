# FastH3 V2：独立八步及双MODEL入口

三个 EXP 节点与六份通过控制图随 v1.83.0 提供，见[版本说明](RELEASE_1.83.0.md)。最新验收及冷／热测量在下方前两节；后续标为历史的待审／未发布记录不代表当前状态。

此功能不迁移旧四步节点、EMA/Turbo图或已验收的双采音频／接缝缺省值。
完整V2学生checkpoint不是给原模型叠的LoRA。代码实现与CPU检查不等于实际视频、人声或提速通过。

## 2026-09-17 最新：Sol修复片B已明确验收，六份通过配方已保存

用户明确回复“B 正常，可以验收”，绑定修复片SHA3830e795479c9fc37f4581a6715a5c43dcb4ab79f1e22d109ed465b1a9d42c34。8808的A仍是原失败对照，不会随代码更新改变声音；不覆盖旧媒体，不做增益、归一化或换音轨。反馈(7)备注只指A的低音量，随后明确B正常；不能把整组评分误读为新B失败或KJ单独声音通过。

六份原生工作流保存在[加速目录说明](../examples/workflows/10-speed/FAST_H3_V2_README.md)，保留已审长度与单MODEL min_tokens=0。移除诊断观察器后沿用同一Core随机噪声、CFG1及SamplerCustomAdvanced连接，不是重新GPU生成或逐像素复现保证。资格仅限绑定样片及已评分项目，仍标EXP。

项目人为帧数上限已移除，必要对齐／上下文规则保留，见[帧数说明](FRAME_LIMITS.md)。六份控制图真实原生UI导入及完整工作流／API导出均已核验，原用户标签已恢复。下方为保留的历史机械记录，旧待审措辞已由本节覆盖。

## 2026-09-17 固定生产配方冷／热对照已完成

RTX4060Ti16GB，同提示词、seed2609032101、832×480、73帧／24fps、CFG1、Qwen和两路VAE。两个独立进程串行运行，各自新进程首跑cold及同进程cache-none完整重跑hot。旧EMA-B保留原生Euler8、shift12/3、KJ Sage；完整V2学生保留训练DMD8、shift10/3、learnedVSA、h1/c1、min_tokens=0。不是同模型／同采样算法或等画质实验。

| 配方 | cold整图／采样 | hot整图／采样 | 周期观察整卡最高cold／hot |
| --- | --- | --- | --- |
| EMA-B原生8步＋KJ Sage | 103.35s／73.03s | 99.90s／72.85s | 13,792／13,981MiB |
| V2训练DMD8＋VSA | 78.20s／49.83s | 71.92s／45.76s | 14,001／14,021MiB |

四次各完成8次真实网络执行；两路采用一致的只读线程局部CallAudit，均观察400次selector及400次选定kernel包装函数成功返回，另有2次PyTorch调用。V2原生运行报告也逐次确认400VSA。各次H.264+AAC／73帧、音视频长度及视频／音频／联合完整解码通过；冷／热媒体各自SHA相同不是缓存命中，实际8次执行和空graph_cached_nodes另有证据。

本组V2整图耗时少约24%／28%，但整卡显存观察没有下降，不能推荐为通用省显存方案。周期进程RSS最高约EMA-B57,309／57,325MiB、V246,136／46,148MiB；系统RAM最低可用约17,312／18,496与29,718／28,500MiB。全局PyTorch、测试服务reserve5／headroom2保持一致，不写入用户配置。

cold仅指新进程／模型首跑，磁盘缓存未清；hot重建graph loaders，不承诺同一MODEL对象缓存。墙时含一致观察器／调度开销，不是裸CUDA内核计时；整卡周期样本包括桌面活动，不是独占精确峰值。两进程来源／全部资产SHA首尾不变，自有服务及子进程均已清理。单机单组无统计／任意素材画质／长片／16GB安全保证，不重审已接受样片；原失败记录保留。

## Sol 音频修复：历史机械记录

用户反馈绑定实际七路媒体，全部播放到片尾。训练VSA h1/c1与h4/c2、官方模板的画面／声音／口型接受；
模板接缝未评。Dense Relay与训练VSA真实中断恢复的两条8秒视频，画面／声音／口型／接缝均接受。
Dense KJ/Sol对照的画面／口型接受，但整组声音有问题，备注明确原B/Sol声音很轻；不能外推A的声音已单独评分通过。
资格只限这些绑定样片，不重跑已接受组，不代表任意参考、LoRA、长片或低显存组合均正常。

原外部Sol直接override未传音频／文本保护范围。修复仅在`dense_compat_exp`、无Relay、已认证的Sol分支中，
通过已有H3布局适配器为实际packed非video区域保留精确Q和KV（`sink_q`／`sink_blocks`），仍执行Sol内核，
不删除Sol、不改全局attention。PyTorch/KJ、训练VSA、官方模板及已有带时间bias的Relay路由不改变。
未知override／修改的内核、callback或owner仍拒绝；T8内存节点克隆重绑定以及缓存身份均覆盖，改代码后使用新链ID。

修复同参832×480／73帧探针完成8次网络forward、1600次Sol调用，另2次PyTorch调用；
实际观察到精确保留prefix `[0,5]`，H.264+AAC完整解码，源码与资产未变，自有生成进程已清理。
同一实际工作流、模型、seed和采样参数下，原始平均音量从−36.6变为−20.7dBFS；没有增益、归一化、重编码或换音轨。
这证明真实保护执行和原始音量变化，不证明人声内容、口型或画质已修复。Sol候选仍待用户单独验收。
受影响CPU回归588通过／1条件跳过，旧255份工作流保持逐字节不变；不是全仓测试通过或已发布新版本。

独立复核另补Sol串联KJ的回退边界：在调用报告前冻结嵌套fallback类型、attention／report方法、
kernel／masked_kernel身份，拒绝实例覆盖，并仅重置该runtime独占的KJ计数，不累计前次运行。
该后续加固117项CPU专项通过，与上述回归有重合，不能相加。GPU样片使用无KJ fallback配置；
保留其原始源码receipt，不把新增CPU加固伪称为重新GPU采样或新的听感验收。

## 模型与基本连接

下载[官方Comfy INT8完整权重](https://huggingface.co/FastVideo/FastVideo-FastH3-Comfy/tree/0de92ab26fcb74ee47596332d93e55d20cddfd45/diffusion_models)，
放入ComfyUI `models/diffusion_models/`：

`fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`

固定revision `0de92ab26fcb74ee47596332d93e55d20cddfd45`；22,128,378,696字节；
SHA256 `0922785978dc9bfe1adf27d8b291b0ca763f9f165f882e6cb297c72fbb6deda8`。
复用现有MiniMax Qwen编码器、原生视频／音频VAE；不要求安装FastVideo分布式运行环境。
文件22GB不等于22GB显存。实际卸载、RAM／VRAM峰值和本机速度需实测，不能承诺16GB更快或不OOM。

单MODEL图：Load Diffusion Model → 可选普通权重LoRA／T8内存节点 → V2 Recipe。
将原生AV Conditioning输出的latent接Recipe；Recipe的MODEL/SAMPLER/SIGMAS接SamplerCustomAdvanced，
BasicGuider使用CFG1；随机噪声照常连接。使用sampler零sigma输出，而不是中间x0预览，联合AV解码。
Actual Dispatch Audit同时接Recipe MODEL和sampler输出，之后再解码；报告包含实际VSA／Dense调用。
其中VSA配方统计真实稀疏调用和Dense资格回退；Dense兼容配方不安装VSA观察器，其保留后端的真实dispatch须单独测量，空计数不能当成“零回退”或提速证据。
不要叠旧FastH3四步setup、EMA-B加速LoRA或uniform8 scheduler再称为官方V2配方。

## 三个配方，不能混称等价

| profile | sigma／积分 | attention | 资格边界 |
| --- | --- | --- | --- |
| `trained_vsa_exp` | 固定DMD八次Euler／双时钟10+3 | learned gates、64-token tiles、keep20% | 官方训练配方的本地实现；INT8／Windows内核质量仍需审片 |
| `dense_compat_exp` | 同一DMD表和专用初始化 | 保留明确Dense/Sage/Sol/Relay路由 | 非训练VSA；可生成不证明原配方提速／等画质 |
| `official_comfy_template_exp` | Core simple8／res_multistep | keep10%、起始20% | 官方Comfy模板对照，与训练表和积分方法不同 |

训练表 `[999,874,749,624,500,375,250,125] / 1000` 后追加0；
两路分别计算 `s*q/(1+(s-1)*q)`，视频s=10，音频s=3；不是uniform9点。
首窗完全生成行使用显式注入的原始FP32 Gaussian；partial restart使用实际KSAMPLER状态重建音频时钟，
不能把首窗当partial restart，或对音频重复乘AV carrier。
零／分数mask保持原生条件保护，参考图、驱动声音和续片属于EXP。
首帧参考的任务类型为`I2VA`；`FL2VA`要求同时连接首帧和尾帧。
模型卡的“参考任务未蒸馏”是训练资格说明，不是允许缺尾帧的任务输入格式。

## 内存和第三方组合

- 两个T8独立内存节点可放V2 setup前／后；真实owner、方法、token、wrapper和callback必须一致。
  ChunkFFN保留原生token-local SwiGLU。VSA head grouping使用每组learned gate／pooled cache，不能用旧generic attention替代。
- 普通KJ Sage选择器可放setup前作为Dense段后端。它不是learned VSA的替代，也不证明VSA的速度。
- Sol选择器／tau算法和Kitchen VSA底层sol_attn内核不是同一功能。
  未认证的DiT替换不通过删guard硬接；Dense兼容分支必须单独记录真实路径并实测。
- Relay的逐query时间bias当前Kitchen gate接口不能表达，因此仅显式`dense_compat_exp`保留完整Relay。
  Relay先绑定、V2 setup后接；不删除时间bias，不默默剥VSA再声称训练配方。
- `min_tokens=12288`是缺省性能资格下限，不是显存空闲门槛。小型内核探针可显式设0，报告必须记录；
  GPU／BF16／head_dim／layout／kernel条件仍要满足。Dense回退有报告，不冒称VSA实际运行。

## 双MODEL4+learned upscaler+4内循环

使用独立 `FastH3 V2 · Dual MODEL4+Upscale+4 Loop`，连接两个裸完整学生MODEL分支。
两路可分别加普通权重LoRA和T8内存节点；不要接已经绑定某单latent的Recipe输出MODEL。
前段取固定表0..4，后段取4..8，不重置sigma、不改成旧12/3网格。
LOW输出denoised_x0经过原有learned3D latent upscaler；HIGH使用自己的条件、时钟与模型。
前4步音频未完成，默认继续后4步联合AV，不把粗采音频冻结成成品。

这是未蒸馏的参考／分段／放大实验，不承诺与整幅八步相同。
新链ID开始，缓存绑定模型／LoRA／组件／代码／配方／初始化／前段身份；不把旧链stage文件搬进来。
续接策略、mask ramp和COLOR MATCH沿用显式选项，旧已验收图不迁移。
两段测试共8秒，window124/context22时边界约5.17秒；从边界前一直看到尾部并听完整音频。
尺寸按参考图比例选择32倍数，不拉伸。例如2:3图256×384→512×768。
EAV保持disabled；需要Relay时明确选Dense EXP。

## 固定来源与核验

- [训练模型合同](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/tree/3da2ddfe1954d9cda4c05b643dc0f26007a655c5)：T2AV训练，FL2VA/Ref2VA未蒸馏。
- [官方固定调度说明](https://github.com/hao-ai-lab/FastVideo/blob/01002185948f9318b8da9067896b61c96bf9a2bd/docs/inference/fasth3-distilled.md)：9点、8次forward、独立shift一次。
- [官方Comfy模板](https://github.com/Comfy-Org/workflow_templates/blob/716a8f49eab66f6f5a1eef9757a8f3aec4bba426/templates/video_fastvideo_fasth3_t2v.json)：独立实验来源，不混称同参。

`tools/run_fast_h3_v2_probe.py --cpu`只做隔离环境／注册检查，不加载模型或初始化CUDA。
GPU探针校验固定SHA和50层gates，保存实际配置、源码、调用报告、资源样本及严格媒体解码。
仅隔离测试使用额外观察节点来确认成功完成的八次网络forward；此节点不进入正式339节点注册表。
候选原生工作流由`tools/build_fast_h3_v2_workflows.py`后台构建，先保存在新的artifacts目录，不操作用户画布，不将未审片图提升为推荐。
CPU回归、机器媒体审计和用户审片分别记证据，缺任何一项不能写质量验收或推荐组合。

## 2026-09-16 首轮完整权重实测（尚未人审）

固定INT8权重的大小、SHA256及50层gates均核对通过。在本机RTX4060Ti16GB上，
同提示词、seed、832×480／73帧的两次串行训练配方探针，均完成8次实际网络forward、
400次VSA／0次Dense，并生成严格可解码的H.264+AAC MP4。显式探针min_tokens=0；生产缺省12288未改。

| T8 memory组合 | 采样耗时 | 图执行墙时 | 周期观察到的整卡最高占用 |
| --- | --- | --- | --- |
| h1/c1（无T8内存patch） | 43.97s | 79.21s | 15,262MiB |
| h4/c2 | 79.24s | 103.82s | 15,544MiB |

这两次h4/c2反而更慢、周期占用更高，不推荐把它当V2通用提速／省显存开关。
VSA的分组投影与普通Dense内存路径不同；保留可选组合但缺省仍h1/c1。
数字包括其他桌面活动，来自周期整卡采样，不是进程精确峰值或长期统计。
尚未据此确认人声、口型、画质及新双段接缝；不等同旧模型基线对照。

## Dense后端与隔离测试服务

完整INT8学生的Dense EXP同条件832×480／73帧对照已执行：
KJ Sage和Sol tau0.5各完成8次真实网络forward；h4/c2时，各观察到1600次selector及1600次selected_kernel成功返回。
两次另观察到2次PyTorch调用，保留原始计数，不写“零回退”。均交付H.264+AAC并通过视频、音频、联合三路严格解码，仍待人审。
这不是训练VSA模式，不证明等画质、普遍提速或旧EMA四步LoRA可用于V2。

测试控制器对所有后端显式保留Core全局PyTorch，仅MODEL分支选择KJ／Sol，不改变用户服务的全局attention。
本机Qwen动态加载阶段曾触发持续资源保护；单独设置自有测试服务原生`--vram-headroom 2`后完整对照通过。
这项预留影响动态卸载，不是新节点的空闲显存准入门槛，不写入用户启动配置。
可在探针显式加`--headroom-gib 2`复现。独立测试仍保留连续资源观察；失败记录没有改写为通过。

正确I2VA的Dense Relay内循环已生成两段共8秒：四个LOW/HIGH阶段各完成4次网络forward及800次Relay路由。
完整视频为512×768／192帧／24fps，保留2:3比例；严格H.264+AAC三路解码通过。接缝、人声和画质尚待最后集中审片。
完整新进程缓存返回也已观察到同一个最终视频；首次媒体联合解码进程曾异常退出，清理后的独立三路解码复核通过，原失败证据保留。
已有成品缓存返回不能代替真实中断后生成第二段的恢复测试，也不能把持久化的16次调用报作新采样。

## 真实中断／新进程续段已机械通过（人审待）

训练VSA参考图实验采用两个裸完整学生、h4/c2、256×384→512×768、window124/context22、EAV关闭、无Relay。
首段实际完成4+4并被接受后，隔离控制器向已核验自有服务发送`POST /interrupt`；HTTP200、执行历史、WebSocket和原生链的`interrupted/accepted_count=1`一致。
另一新进程使用相同配置恢复，首段视频／context／三份阶段缓存的SHA保持不变，第二段确实新执行4+4。
报告将首段持久8调用、第二段新8调用分开记录，未测量的中断中途尝试不推断为0。
最终192帧／24fps／8秒H.264+AAC通过视频、音频和联合三路严格解码；成品SHA256：
`bbce0048b154c2ca08c6c64be11116bf26cffa239b5cb3122175fa1fdb85156a`。

该本机h4/c2循环在自有服务原生headroom2时曾触发持续资源保护；测试控制器改唯一显式headroom4后完成。
这不是新节点要求4GB空闲，也没有改变用户启动配置。两学生和Qwen动态驻留还需要大量系统RAM，显卡16GB不足以证明低RAM机器可运行。
参考／续段／latent放大未蒸馏，机械通过不是人声、画质或接缝人审通过。仍须检查约5.17秒附近直到片尾。

普通权重LoRA的实际Core INT8非零apply→requant→冷热复用→unpatch身份有CPU微型测试，不能据此推荐完整V2模型任意LoRA，尤其不要套用旧EMA/Turbo加速LoRA。
所有新增节点保持EXP。开发包与生产旧336节点资格分开；三项新增后339，旧255工作流不迁移。
