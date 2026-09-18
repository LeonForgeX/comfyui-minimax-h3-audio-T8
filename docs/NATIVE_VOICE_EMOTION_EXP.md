# 原生人物音色参考／情绪对白（v1.85.0 EXP）

## 正式发布状态（2026-09-18）

最后两组反馈与`bc521fba14ab6614bf0b96ead42c6136c34be67fc88194733d40dd86f971b257`
及媒体SHA绑定，直接狂怒新声音已获用户接受；连同此前普通声音／标准4+4／独立二采／
单模型两段8秒／EAV固定音画与口型，用户要求保存正式工作流并发布。
使用[正式示例与模型配对](../examples/workflows/36-avatar-voice/README.md)。
接缝和事件NA保持，不认证100%声纹或精确词级时间；EAV+H4数学组合仍未验证。
下面旧“狂怒待人审／未发布”及失败对照是历史，不把旧失败片晋升推荐。
尾部乱说首先检查短话长时间窗：Global不放重复对白，Local每句一次，剩余时间给明确无声动作。

## 2026-09-18 第二次人审：固定视听接受，声音情绪继续补测

`five_track_human_review (1).json`已与当前v11的review_id及18条媒体SHA逐项绑定，
原导出和上一份反馈不修改。Avatar、普通音色、独立双采、单模型8秒、标准4+4及
EAV开关的画面／声音／口型均为“正常／可接受”。所有接缝项为NA，不能记成接缝
或字词时序通过；旧失败基线与ASR观察保留，但不据此推翻真人视听或判节点泄漏。

第8组明确反馈：右侧表情有怒意，但声音区别听不出来。仅将这组的表演短语改成
**直接狂怒／unrestrained FURY／explosively furious shouted voice**，不再写克制、
温柔或禁止喊叫。与原平静基线同首帧、参考录音、seed、原生Stock20及73帧512×768，
只改表演短语；平静片复用原字节，不重跑。台词仍只在Local出现一次，0–2秒完成，
之后闭口眨眼、无声呼吸，不留长时间继续说话的指令。

新候选实际20次前向，Job284.297秒exit0／cleanup0，完整73 H264帧与96个原生AAC
块独立解码。两次无提示CPU ASR均只有“你终于回来了”；开头2秒原始RMS为平静片
约1.642倍，没有削波或增益／静音／裁音／贴参考声轨。声能或波形区别不证明语气
已达到狂怒，也不认证声纹一致，仍待新视听确认。不修改采样、参考权重或接缝逻辑。
新SHA：`9e2d928754cedffec35ec65af2a7cf5f1b2c5515353856236477df4bf83a195b`。
收据：`artifacts/five-track-development-20260918/native-voice-fury-gpu-process-v2.json`、
`native-voice-fury-post-v2/report.json`及`human-feedback-20260918-v2/analysis.json`。
下面“待旧组人审”描述均为历史，不要求重跑已经接受的固定视听组合。未发布。

[仅两项补测审片](http://127.0.0.1:8841/review.html)第一组：A为原平静片，
B为新狂怒片，默认只播放B声音；通过本组声音下拉切到A作对照，勿两路同时听。
其他已接受固定音画组合不重跑，本组仍待狂怒语气真人确认。

用户再次明确：尾段额外中英文先看台词长度、说话窗口和后续动作是否匹配，
不是仅凭ASR就认定节点／参考泄漏。下文历史风险与原始识别保留，不能把它们
升级为已证实根因。新测试只在Local出现一次目标台词，Global保留声音与身份
绑定而无重复对白；给短台词短窗口，结束后明确闭口并继续具体无对白动作。
渲染补齐区间也要覆盖动作，不裁音／静音／增益／换轨掩盖多话。

## 最新：短对白时间与明确尾动作／更强情绪新组（v13）

用户指出短话配过长时间、没有后续动作会使原模型继续乱说；先改测试设计，
不据此前结果推断节点或参考泄漏故障。新两条同人物/录音/seed/native20，
73帧24fps/512×768，只比较平静与明显克制愤怒：前0–2秒说一次
“你终于回来了。”，余下闭唇眨眼/无声呼吸。各20真实前向，自己Job473.609秒
exit0/cleanup0；完整73 H264帧/各96 AAC blocks解码，源/Core/模型/正式八图不变。
auto/zh四次无提示CPU识别都只有目标句、未识别旧尾英文。这是支持用户测试设计
解释的有限观察，不以ASR替代试听、情绪/音色/口型或证明所有长片问题修好。
没有裁音/静音/增益/删参考/替换音轨。

最新集中审片 http://127.0.0.1:8838/review.html 第八组，原记录保持。旧五组真实反馈
只接受固定Avatar画面/声音/口型、Meridian最小P2画面，以及原voice普通视听口型；
轻喜悦差异不明显明确需要补测。NA接缝不升格，新第6–8组不继承旧验收。
Meridian四独立节点与空间／时间编辑及P4机械验证已补齐，具体看第9组和本地Meridian文档；
不冒称其最新全量GPU／主观验收或整项目标完成。历史P2门禁措辞不作当前未开发状态。
旧标准4+4、EAV对照及单模型长视频ASR风险仍保留，不晋升为推荐图。

## 最新接续：EAV 具体组合完整对照，保留 H4 未验证报告（v12）

按照用户纠正，先改短台词的时间与后续动作，而不是认定节点故障。新链都是前
0–2秒／2–4秒各说一次，后4秒具体无声点头挥手／放手转头微笑；Global不重复
台词，最后动作覆盖渲染padding。交付两段共8秒／192帧512×768，没有拉伸。
本组保留原双模型EAV Stock20契约，LOW20→原learned3D→HIGH4，每条真实48前向；
不替代下方独立标准4+4。原生参考音色／joint Relay／H4C2／auto已完成一采音频／
上下文22／masks／LOW成片参考／运动颜色／5ms音频桥保持，只有EAV开关不同。
参数τ4、原生视频进度15%–90%、workspace32MiB、g≤1.5，只应用两段LOW20。

OFF Job634.406秒exit0/cleanup0；ON生成完整48前向与8秒成片后，artifact审计
误要求verified状态／50测量而退出1，Job620.781秒cleanup0，原失败收据保留。
H4每活跃前向有200测量；独立CPU后审计47.25秒exit0/cleanup0重新绑定现有
成片／Core／模型／生产源码，**没有重跑GPU，也没有将原报告改成verified**。
原始两段EAV仍是executed_user_stack_unverified/composition_verified=false：
各20前向／6活跃／1200测量、gmax1.0746442／1.0773895、workspace约21.06MiB。
这只证明绑定的执行与遥测，不保证H4分头与全头FETA统计等价或组合质量通过。

两条完整192 H264帧和251 native AAC blocks全部解码。OFF/ON SHA分别
`638c586df37d35cdefb32a165f6c5b26b2e4dfea93ef388a8d119080354597a2`／
`2558b44d1949c33618e2be54a1764aca7310c4ea0f5784358e3c1e7c31cdec24`。
无提示离线CPU ASR两条均提示第二句不完整并有额外英文；不是完整对白修复或推荐，
也不是试听、音色／情绪／口型／接缝或节点泄漏根因证明。不裁音、静音、增益或换轨。
相同输出名的首尾帧放在独立off/on目录，避免审片预览互相覆盖，原审计工具未改。

[集中审片](http://127.0.0.1:8836/review.html) 第七组为本对照，原六组和旧反馈保留。
自己的隐藏loopback服务PID39124，HTTP13完整MP4 SHA／26精确206／26PNG／4私有
404通过；没有操作浏览器或冒充人审。收据见native-voice-eav-pair-terminal-handoff-v1.json
和verification-final-v12.json。70951／27170均关闭，不再轮询／重启旧GPU。
五轨原范围、Meridian原P2人审→完整P3/P4、TAE同进程恢复边界，以及Avatar／
原生音色／情绪／准确对白／口型／接缝人审保留。正式八JSON和采样／音频／接缝
算法没改，整体尚未完成；未发布、不空间采样tile、不动用户Core／前端／队列。

## 前一接续：标准4+4完整音画终态，尾段仍待试听（v10历史）

原方案标准4+4已真实执行，不以20+4替代：non-pruned Ref2VA/AdaLN2688＋两路独立
EMA B强度1，actual Core350/16节点/259对全部518键CPU形状匹配，两段8秒共16次
真实前向。LOW4/HIGH4各两次；Job636/721.547秒exit0/cleanup0，session84176已关闭。
原auto联合音频、8步表、learned3D、H4C2、上下文/masks/接缝保留，EAV关闭；
不锁未完成一采音频。不同底模/LoRA/步数，不是旧pruned20的提示词单变量对照。
Core/模型/素材/生产源绑定不变；完整192帧512×768 H264与251 AAC blocks全部解码，
MP4 SHA `9fbec89c72666f3f50b9936bee81596e7df0929cc42f4234e654d2acf3ac9581`。
无提示离线CPU ASR auto/zh在两句目标台词后约3.30秒起仍识别额外英文
`He sings, Brother Babel sings, his angry little eyes`；不能算完整对白修复或推荐。
ASR不是试听、音色/情绪/口型/接缝或参考泄漏根因证明；不裁音/静音/增益/换轨掩盖。
新[集中审片](http://127.0.0.1:8835/review.html) 第六组是本独立4+4，原五组/旧8834
和反馈保留。11完整MP4 SHA/22精确206/22PNG/4私有404通过，不冒充实际浏览器或人审。
正式八JSON不覆盖；精确receipt见verification-final-v10.json与checklist。
原生音色/情绪/台词/口型/接缝仍待人审，EAV组合未继承资格，不默认改原采样/音频。
Meridian原P2人审→P3/P4及TAE同进程Core恢复边界保留；未发布、不空间采样tile，
不动用户Core/前端/队列，ROADMAP/SKILL永久本地。下方v9是运行历史，不能重启旧任务。

## 前一接续：补原方案标准4+4资格（v9运行历史，已终态）

20+4不能替代原方案的标准4+4。新artifact non-pruned Ref2VA/AdaLN2688＋独立两路
EMA B匹配259对/518键；旧pruned输入8不匹配，不能直接套或静默丢AdaLN。
actual Core350/16节点/同GPU helper CPU通过，Job43152/35.266秒exit0/cleanup0/CUDAfalse。
GPUv1工具同名导入错误在采样前真实终态46.187秒exit1/cleanup0收据保留，只修artifact。
新GPUv2 session84176/controller16064/worker636已CIM确认活跃，预计两段8秒16前向，
观察首LOW4/HIGH1；deadline1500秒自己的Job，轮询原句柄，不因观察超时重启。
这是不同底模/LoRA/步数的独立4+4配方，不是下方pruned20仅提示词对照。
沿用最新短对白/无声动作、原auto联合音频、learned3D、AV上下文/mask/接缝，
EAV关闭，不锁未完成粗采音轨、不裁话/静音/增益/换轨，不预先判训练或质量通过。
完成后全192帧AV及无提示CPU ASR，再新审片；不覆盖旧八图，仍未发布。

## 前一接续：完整新设计8秒终态（v8，不认定程序BUG；历史）

用户指出短话分配过长说话时段、后续无具体动作会让原模型继续即兴说话，因此先改测试设计，不据此认定节点BUG。
已用新链完整跑两段共8秒：0–2秒第一句、2–4秒第二句、4–6秒无声点头挥手、
6–8秒放手转头微笑。Global不重复台词，Local各一次，frames0-47／48-95／96-143／144-208；
原193→209渲染补齐由无声动作覆盖，交付192帧／24fps／512×768，无拉伸。
实际Core350验证raw9／joint-H4C2接线12通过，CPU Job42252／8.531秒exit0/cleanup0。
新GPU实际40前向／两段完成，自己的Job38244／1174.141秒exit0/cleanup0；
原模型／seed／参考录音／每段Stock20／原生音频／AV上下文／mask／接缝设置精确保持，
没有WindowText或一次台词归属实验节点，没有采用旧提示词首段缓存。
完整192帧H264与251 AAC blocks全解码，MP4 SHA
`b0aaef97e6cc07393b02c526ebca74e0c90929c6817f3b6cc310045990dac268`。
无提示离线CPU ASR auto与zh仍识别尾段非脚本英文“He was talking to me, his angry little eyes”；
前两句已被识别，但不能判完整对白正确／问题解决。ASR不是试听、口型、音色或根因证明；
这个固定设计仍有额外话风险，不推荐晋升正式图，不默认改采样/音频代码或删参考条件。
ASRv1因本工具提供错误的本地模型目录而在推理前失败，exit1/cleanup0，收据保留；
v2读取已成功报告中的真实model_directory，27.5秒exit0/cleanup0，输入与模型字节不变。
新集中审片 http://127.0.0.1:8834/review.html 第五组A旧时段／B新完整设计，其他四组证据保留。
仅新建自己的loopback服务PID38304；HTTP10完整MP4 SHA／20精确206 Range／20PNG／4私有404通过，
没有操作浏览器，不当实际浏览器播放或人审证据，旧8833/v6服务与原反馈未覆盖。
下一步先试听新第五B与已生成的其他候选，确认尾段是否确有多话，再决定单独来源/条件诊断；
不裁话、静音、加增益或换轨掩盖失败，不重复已完成GPU测试。
522范围CPUv19／25模块／2127冻结来源只差描述性features.json，八正式JSON仍同v15；
HEAD／index／Sol／已验收Semantic Bridge图不变。五轨原范围和门禁全部保留：
Meridian原P2人审→P3/P4；TAE同进程Core权重恢复边界未解决；Avatar／音色／情绪／口型／接缝仍需人审。
本轮有实际新GPU／全媒体／ASR／新审片进展，非无进展，整个goal未完成。
不发布、不空间采样tile、不动用户前端／队列／Core／未知进程，ROADMAP/SKILL永久本地。

## 前一接续：先纠正提示词／时间设计（历史草案）

用户指出短话分配过长时段、说完无具体新动作时，原模型可能继续即兴说话。
原测试两句各约3秒，静听从6.08秒才开始；先处理这个设计问题，不做采样／音频代码修复。
新artifact native-voice-prompt-timing-v1/prompt.json明确两句各2秒，后4秒点头挥手／
放手转头微笑；Global只共同结构，台词各写对应Local一次，frames固定而非百分比。
原193→209补齐由最后无声动作覆盖，交付仍192／24fps／8秒。其他算法参数不改。
必须用新链完整两段8秒检验，不能复用旧提示词首段；尚未GPU或听感验收，不发布。
下方单后段静听负结果保留，但不证明合理完整时间设计无效或节点存在BUG。
优先验证新设计；仍复现才单独检查来源／参考条件，不将相似ASR定为程序泄漏根因。

## 后段残留说话指令：单变量诊断终态（2026-09-18，未解决）

实际builder、已执行API和真实193→209投影确认：Global本来就含
`Clear newly generated Mandarin dialogue`，不能再把“未设全局目标语言”作为
本样例的事实或优先修复。第二交付窗口为[124,192)，render90/context22；
事件2跨[73,146)，去掉成对台词后仍是 `S1 speaks gently ...`，并带不重复提醒。
同一窗口事件3却要求聆听、不再说话，形成正向说话指令残留风险。
原生Ref2VA把独立录音的完整audio latent当作参考条件；它不是仅有声纹的
独立embedding。此事实及相似ASR文句支持进一步隔离，不能证明全部根因。

新工具 `tools/qualify_native_voice_continuation_text_gpu.py` 只干预该固定投影
的事件2文本为保持喜悦表情、闭唇聆听；不是对任意用户字符串做NLP改写。
Global、事件3、绝对时间、width/sigma、joint route、参考音频、原生AV上下文、
采样和编码保持。57项/3模块CPU、2127冻结来源不变、CUDAfalse/零skip通过。
读取前段20+续20的真实终态/Job清零/实现与资产SHA，重新校验前段MP4/context；
首段只作只读输入，不冒充新链接受件，不修改原manifest或旧八正式图。

首v1工具在条件编码因直接调用未采用执行器的inference_mode而失败，46.781秒exit1/
cleanup0，未进入扩散采样；失败证据保留。工具局部推理上下文修正后57范围CPUv2通过。
独立新WindowsJob/GPU lease实际完成后段20，seed20260919、512×768、
render90→原context裁切22→交付68帧；472.187秒exit0/cleanup0，含编码/解码阶段396.703秒。
完整68帧H264/90 AAC blocks/32kHz解码通过，SHA
`578ccab4b234326121d874a986cca1b121faf24a65d12e66149179c8fa1f2acb`，
RMS0.014277910/peak0.126367047；PTS0至67/24。原首段、context、manifest、实现/Core/资产
字节前后保持；原生两尾tensor保持。完整音轨未裁音/静音/增益/替换。
无提示离线CPU ASR auto读出 `He was talking to me, the same word would like.`；
zh读出 `他在跟著我同樣的同樣`，后者平均logprob约-1.268、识别较弱。
两者都不是预定静听/无新增对白的证据；仅去掉正向说话指令不够，不判根因/试听/修复。
最后较广CPU522/25模块/2127冻结来源不变/CUDAfalse/零skip通过，非全仓资格。读取
`native-voice-continuation-text-process-v2.json`、
`native-voice-continuation-text-gpu-v2/terminal.json`和真实进程句柄后再决定下一步，
两个工具Job及媒体/ASR/CPU Job全部终态清零。它**不是完整8秒/内循环恢复/接缝/人审资格**。
下一步先只读检查/解码首段原生audio_tail，区分续接声音与独立完整参考录音条件，
再决定参考条件单变量验证。`audio_overhang=1/3`单位是40Hz音频latent帧，约8.33ms，
不是1/3秒，不能据此臆测有0.333秒未交付对白，更不能擅自截短原生上下文。
失败保留，artifact不推荐、不晋升正式图；旧v6审片及双采joint候选独立保留。
未新增生产数值路径/节点参数，未发布；Meridian原P2人审门禁和TAE Core恢复差异仍待。

## 单模型一次台词归属追加实验：未通过完整对白（2026-09-18）

独立窗口文本EXP新选项 `dialogue_start_owner_exp`：仅事件起始交付窗口保留该事件
成对 `<d>...</d>` 字面台词，后续相交窗口保留视觉／情绪和原时间／width／sigma，但
不重新发出台词。原生已知AV上下文、音频对象、mask、Stock20与接缝均未修改，
不裁音、静音、补增益或换轨。旧 `preserve_all` 恒等及 `accepted_window_text_exp`
原行为保留。此新模式拒绝Global的 `<d>` 和坏标记；未标记对白不识别。
它不是论文all-key方法，不保证逐字／一句完成／不重复。

同一193→209帧计划、三Local、seed20260918、H4C2、512×768、两段8秒：
首段20提交后受控取消，Job518.532秒exit0／cleanup0；续Job只补后段20，
632.860秒exit0／cleanup0，sampling553.438秒。共40真实前向，无废弃partial。
首段MP4 SHA与上一窗口试验相同，自己的首段/context字节恢复后保持；
完整192帧H264／251AAC blocks全解码，SHA `8989c3938c1d91ea640cf2f820c724c69dbe74a737528e34b82407b06a5a9d40`。

无目标提示CPU ASR auto／zh均读出两句中文后还有英文：
`Better/Bether talking to me ... his angry little eyes`。
参考录音的既有无提示识别为 `all the time he was talking to me his angry little eyes were following the lake`。
文句相似提示**可能参考内容泄漏**，但识别不等于真人试听或全部根因证明；
没有读出第二句重复也不代表完整对白已正确。此试验不推荐、不判修复／质量通过。
前端图只在artifacts，不覆盖八份正式EXP图或默认值。先试听片尾确认英文风险，
之后才选择经实际来源核实的单变量方案；下方Global已经设定普通话，不能当作遗漏。
本轮先机器隔离“视觉与说话指令分离”，不把用户睡觉当作真人验收；不能合并两项或
把问题藏进音轨后处理，不再盲目重复本例。原双采joint候选资格独立，未被本试验否定。

证据 `native-voice-dialogue-owner-gpu-v2/terminal.json`、
`native-voice-dialogue-owner-media-audit-v1/report.json`、
`native-voice-dialogue-owner-words-cpu-v1/report.json`；路径均在
`artifacts/five-track-development-20260918/`。范围CPU501／24模块／2125冻结来源
不变／CUDAfalse与actual Core350通过，不代替声纹、情绪、口型或接缝人审。

这部分复用H3原生Ref2VA条件，不新增伪“声音克隆模型”。人物图片接
`ref_images.ref_image_0`，参考声音接 `ref_audios.ref_audio_0`；匹配Ref2VA
底模，`audio_mode=native`，`drive_audio`／`final_audio`不连接。
SaveVideo接AVDecode的生成声音。参考录音不是直接交付音轨。

提示词三块：Global只写固定角色、`<Picture 1>`／`<Audio 1>`绑定、场景和
声音环境；Local每行一次对白，把新台词写在 `<d>[Mandarin]…</d>` 内，
喜悦／克制／停顿等表演要求写在标签外；Timeline按实际Relay时间模式
给每行配区间。一次性台词不得放Global导致重复。
连接参考视频声音后Audio编号可能改变，按media_map_json重写，不永远硬写Audio1。

单段基线Stock20、Euler、video/audio shift12/3，无遮罩冻结生成音频、不加
Turbo LoRA、不额外替换VAE。两条同首帧／参考录音／seed的普通／轻喜悦新对白
已各完成20次真实前向，完整73帧24fps512×768 H264和生成音频审计通过。
参考录音164864样本，生成声音97600样本／32000Hz，非参考tensor；源只作reference。
机器证据 `native-voice-gpu-v2/terminal.json` 与 `native-voice-media-audit-v2/report.json`。
检查工具v1误把零下标socket当公开label导致后置断言失败，修正的是工具；
生产media编号没有修改，失败样片／收据仍保留。

## 双采与长视频

### 补充有／无参考对照及台词证据

同一首帧／seed／Stock20、目标“你终于回来了。”的新受控两路已完成：有参考
20次原生音频条件数为1，无参考20次为0；只移除参考连接与Audio绑定句，不接
drive/final音轨。73帧完整AV审计在`native-voice-ablation-media-audit-v1/report.json`。
生成音频RMS分别0.013118与0.154174，整体音量明显不同；不自动归一化，不把
“有变化”当成身份克隆成功。新审片v2中A/C为本轮受控对照，B为之前的情绪片。

本机已有识别模型的离线CPU核对，没有给识别器目标句、prefix或hotwords。
四条单段auto/zh均读出目标句；原参考英文句不同。ASR仍会误识别，不能替代音色、
情绪、噪音、口型和接缝人审。证据`native-voice-words-cpu-v1/report.json`。
旧8秒双采识别到末尾重复第一句，ASR估计4.04–6.4秒，不称对白时序通过。
effects audit明确`video_only_paper`，视频时间偏置不等于音频台词的时间路由。
独立候选通过已有Query Route节点显式`joint_av_exp`，不更改旧默认或音频采样
math；该音画扩展未经Prompt Relay论文音频验证，改善与否以新完整成片和人审为准。
本候选已完成两段8秒、48真实前向，531.625秒采样／owned546.891秒exit0/cleanup0。
完整192帧512768H264／251AAC blocks已独立审计，SHA d7862932…8479bb。
第一次媒体审计误递归读到124帧段文件而触发帧数断言，保留该失败目录；精确指定
assembled完整成片后的成功证据是`native-voice-dual-joint-media-audit-v2/report.json`。
无提示CPU ASR auto/zh都只读出目标两句，不再读出第一句重复；auto误标en，
ASR时间段不精确，不用它证明静默区间或身份／情绪／口型／接缝通过。
证据`native-voice-joint-words-cpu-v1/report.json`；不沿用旧片资格。
额外显式joint EXP前端图已单独actual Core保存图转API验证，原六图字节保留。
新图在`examples/workflows/36-avatar-voice/2026-09-18_T8_voice_dual_joint_EXP.json`，
人审待，未发布、不作已验收推荐。图只增已有路由节点，旧默认和采样math不变。

现有单／双模型长视频节点已经传递图片、音色参考与原生AV尾部，不需要另造
一套情绪采样器。新增CPU测试走真实条件构建器，分别覆盖两画幅／两阶段及
下一段、原参考音频不变、时间线motion audio保留、视频声音引起的编号变化；
fake encoder资格不代表真实模型音色质量。

开发候选六图已放正式 `examples/workflows/36-avatar-voice/`，
实际Core349保留原343前缀、schema及保存图再转API验证通过，CUDAfalse、未排队。
来源资格 `avatar-voice-workflow-cpu-v14/report.json`；不是用户服务已重启加载。
其中`voice_long`每段Stock20；`voice_dual`一采完整20步→原learned3D→独立二采4步，
auto使用已完成的一采声音，不把原生Stock20验证迁移为Turbo4+4音色验证。
两段内循环8秒、192帧，window124/context22，接缝约5.17秒；Relay输入length193
按原生17n+5对齐成209帧计划，事件区间约0–3.042/3.042–6.083/6.083–8.708秒，
尾部超出交付部分裁去。需要精确8秒计划可用已对齐的192或seconds显式区间，
但换配置须另验，不把本轮GPU／人审资格迁移过去。
全局不含重复对白，Local三行对应0–35/35–70/70–100%。旧图／sigma／放大器／
音频锁定与已验收接缝逻辑不更改。组合或参数变更须新chain_id。

新的8秒双采原生音色串行GPU已完成，两段各LOW20／HIGH4，总48次实际前向，
192帧512×768 H264和完整生成音频独立审计通过；515.953秒，owned Job退出0／cleanup0。
自有VAE使用原生static patcher、原compute dtype和原内部chunks，避开当前Core动态驻留问题，
没有Core或用户全局修改。H4/C2是计算分块，不是空间采样tile。
成片SHA187fdc63918ba8b0c2de55ac213dfc967864bce7aa545839dea8cec8fe6436c2。
证据 `native-voice-dual-gpu-v2/terminal.json`、`native-voice-dual-process-v2.json`、
`native-voice-dual-media-audit-v2/report.json`。v1工具计数wrapper丢失原生forward签名，
第2段误判mask不支持；只修工具wraps，保留失败，不改能力门禁或采样math。
原单模型voice_long图仍只有CPU/Core接口资格，不把其他图成片资格迁移给它。

新增显式`voice_long_joint`，原单模型Stock20内循环、joint_av_exp／H4C2、两段8秒；
没有learned3D／HIGH二采，不接drive/final录音，不改生产采样或原生AV尾部续接。
首次自有Job在35次前向后触及900秒保护，904.547秒cleanup0，第一段124帧及上下文已机械提交；
新Job严格核原配方、生产467源、6项现有资产／Core来源及accepted文件SHA，再只补第二段20次。
641.703秒／sampling583.64秒exit0/cleanup0，第一段MP4／context原字节不变，完整192帧512×768
H264、251AAC blocks、RMS0.011867通过独立完整AV审计；SHA c295c6cc…95caac1。
跨作业实际55次前向，包括废弃15次partial；两个最终段各配置20步，不称全实验仅40次。
新图actual Core保存转API349／保留343前缀／noqueue／CUDAfalse通过，原六图字节保留。
证据`native-voice-long-gpu-v2/terminal.json`、两process收据、`native-voice-long-media-audit-v1/report.json`。

无目标提示离线CPU识别auto/zh都读出“你终于回来了这次别再走了这次别再走了”，
第二句重复的文字证据保留；不把能解码当对白时序通过，也不据ASR独自宣判全部听感或根因。
该单模型例不作质量推荐；显式joint时间偏置仍是软引导，不是逐字／精确事件控制。
现有双采joint只读两句的证据不迁移给单模型。继续以独立试听及人审定位重复是否真实。
本轮未盲目裁对白、替换声音或增加响度，旧音频／接缝／sigma保持。
原生部分4+4与EAV叠加没有本轮预训练音色GPU资格；Stock20／20+4不能替代它们。

## 限制

追加窗口文本单变量EXP：显式accepted_window_text_exp仅过滤不与交付窗口相交的局部文本，
默认preserve_all原Plan恒等，跨段事件的width／sigma／坐标和原生AV尾部不改。
同原计划／seed／Stock20／H4C2两段8秒：首段20提交后在自有Job取消，新Job只补后段20，
两作业退出0／cleanup0；没有废弃partial，首段视频／上下文字节保留。完整192帧与251AAC
blocks全解码通过（native-voice-window-media-audit-v1），但无提示CPU ASR auto／zh仍发现
第二句重复（native-voice-window-words-cpu-v1）。未修复、不推荐，实验保存图留artifacts，
不覆盖原八正式JSON；新审片v5第五组两路均为重复风险例，第三组B仍是双采joint候选。
这只排除了“省略不相交局部文本即可解决本例”假设，未证明所有原因，不裁音／增益／换轨。
详细五项进度与继续门禁见docs/FIVE_TRACK_LOCAL_HANDOFF_20260918.md。

跨语言参考、声纹相似、中文逐字正确、情绪区别、口型及接缝必须单独人审。
声音参考不是语音识别／TTS逐字保证，不承诺100%克隆或任意时长／显存通用成功。
只在审片后保存质量推荐配方；CPU/Core与媒体完整性不能替代验收。
本轮不发布，不自动排队／下载或重启用户服务。
