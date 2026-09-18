# Avatar 与原生声音正式工作流（v1.85.0 EXP）

指定样片已获用户最终验收。正式源码在本节点`h3_t8`，此目录不是独立开发树；
更新后自行重启ComfyUI，导入JSON并换成自己的合法素材，不自动改旧画布。

|文件名路线|正式用途|
|---|---|
|avatar／avatar_preview|录音驱动I2VA、lock_source、LOW4→原learned3D→HIGH4；后者加可选LOW x0预览与定向取消|
|voice_neutral|pruned Ref2VA原生Stock20参考音色，新普通对白，无Turbo|
|voice_emotion／voice_calm_comparison|已通过的直接狂怒声音及其平静对照，不推荐旧声音不足的克制怒片|
|voice_dual／voice_dual_joint|等价的显式joint_av_exp，LOW20→learned3D→独立HIGH4，短对白及无声尾动作，两段8秒|
|voice_long／voice_long_joint|等价的单模型Stock20联合路由，两段8秒，短对白＋具体无声动作|
|voice_standard4plus4|non-pruned Ref2VA配两份独立EMA B，标准LOW4→learned3D→HIGH4，两段8秒；不混用pruned权重|
|voice_eav_off／voice_eav_on|20+4独立EAV开关对照；τ4、15%–90%、g≤1.5、workspace32MiB|

Avatar的录音编码进audio latent，两阶段mask=0，交付明确用原录音；首期独立单段，无空间sampling tile。
原生声音图接ref_images／ref_audios，Ref2VA、native，不接drive_audio或final_audio，不贴参考声轨、无自动增益。
HIGH MODEL／LoRA可独立另接，未知组合保留但不继承本图资格；加入视频声音后按media_map调整Audio编号。

三块提示词：Global只写固定人物／音色／场景，不重复对白；Local每句<d>一次；Timeline与时长同步。
长视频新图0–2秒一句、2–4秒一句，4秒后明确闭口点头／挥手／转头等无声动作，避免短话长窗一直多话。
193帧计划补齐209帧，交付192帧／8秒；补齐尾由无声动作覆盖，不保证词级时序。
render124/context22的接缝约5.17秒，不是各4秒；改时长同步length与timeline，改配置用新chain_id。

示例人物2:3，512×768，LOW256×384，32对齐、不拉伸；换图同步自身比例。
单段73帧／24fps≈3.042秒；Avatar AudioWindow关闭最小上下文扩展。
H3底模放models/diffusion_models，Qwen放models/text_encoders，VAEs放models/vae，
EMA B放models/loras，learned3D放models/latent_upscale_models。Avatar无需新专用权重。
TAE下载：[t8star/Taeh3-Comfy](https://huggingface.co/t8star/Taeh3-Comfy)。时序tiny放models/vae_approx/taeh3.safetensors；2D另存taeh3_2d_kijai.safetensors并主动选择，不覆盖时序。
预览接MODEL，sampler/sigmas原接线不变；2D潜帧不是24fps最终影片；取消仅当前请求，不清全队列。

固定样片画面／正常声音／口型接受；NA接缝不当通过，不保证100%声纹或精确字词。
EAV+H4仍composition_verified=false；Core同进程LoRA驻留差异保留为已知边界；未知素材／显卡自行检查。
详见[发布说明](../../../docs/RELEASE_1.85.0.md)、[声音](../../../docs/NATIVE_VOICE_EMOTION_EXP.md)、
[Avatar](../../../docs/AVATAR_PROGRESSIVE_EXP.md)和[预览](../../../docs/TAEH3_SAMPLING_PREVIEW_EXP.md)。

## 历史开发记录（不作为当前推荐）

下方旧“待审／未发布”、旧视频路由、percent长窗及失败样片是历史。正式JSON已从实际通过配方重存，
原八图和说明备份留在本地发布artifact，不把未通过基线推荐给用户。

这里有原六份前端 JSON，以及新增的显式 `voice_dual_joint`／`voice_long_joint` EXP 候选，可拖入 ComfyUI；正式代码在本节点的 `h3_t8`，不依赖独立开发树。需要用户自行重启自己的 ComfyUI 才能加载新节点，测试不会替你重启、清队列或改旧图。

| 图名 | 用法／资格 |
|---|---|
| `avatar` | 人物首帧＋现成录音，lock_source、LOW4→原 learned3D→HIGH4；单段录音驱动，不是声音克隆 |
| `avatar_preview` | 同上，加可选 LOW x0 TAEH3 动态预览与当前请求取消；预览不是最终画质／声音 |
| `voice_neutral` | Ref2VA 原生参考音色生成新普通对白，Stock20，无 Turbo LoRA |
| `voice_emotion` | 同底模／参考／seed，改为轻喜悦新对白；声音及情绪区别仍待人审 |
| `voice_dual` | 保留视频路由基线；ASR发现第一句重复，未通过对白时序，不作推荐。Stock20→learned3D→HIGH4／8秒，H4/C2只是计算分块 |
| `voice_dual_joint` | 新候选显式 Plan→Query Route（joint_av_exp）→内循环，其他采样／接缝不改。两段8秒48前向与完整AV完成，ASR只读出目标两句；精确时间／音色／情绪／口型／接缝仍待人审，未发布 |
| `voice_long` | 现有单模型原生 Stock20 内循环长视频，图片／声音 reference-only；保存图 Core 接口通过，不继承双采图的 GPU 质量资格 |
| `voice_long_joint` | 单模型显式 joint_av_exp／H4C2，各段Stock20，两段8秒完整AV及受控续跑完成。无提示ASR发现第二句重复，未通过对白时序、不作推荐；仍需试听、音色／情绪／口型／接缝人审 |

新增joint图单独经过实际Core保存图转API验证：349节点保留343前缀，未排队、未使用GPU或浏览器。证据`avatar-voice-joint-workflow-cpu-v1/report.json`，正式拷贝仅换行符规范化，原六图字节不改。联合音频路由未经Prompt Relay论文验证，不会静默迁移旧默认；有参考路整体音量较低，没有自动补偿。

单模型joint图另有`avatar-voice-long-joint-workflow-cpu-v1/report.json`；原六份默认图重验v15 API字节与v14一致，新生成前端只变工作流UUID，正式旧图不覆盖。单模型首次900秒超时在35次前向后退出；保留原首段及上下文，新自有Job只补第二段20次，641.703秒退出0／cleanup0。跨作业实际55次，其中15次不完整段废弃，不能记成全实验40次或首次超时通过。完整192帧H264／32kHz生成音轨在`native-voice-long-media-audit-v1`；旧`voice_long`没有此显式joint／H4C2接线，资格不迁移。

六图都经过实际 Core schema 与保存图再转 API 校验，无 GPU、浏览器或排队；单段实际推理及完整 AV 已另测。8 秒双采以本轮终态收据为准，不把“能打开图”算作声音、口型、接缝或性能通过。不保证任意素材／LoRA／显卡或跨语言音色逐字准确。

人物示例为 2:3，512×768、LOW256×384，32 像素对齐。更换图按其真实比例选尺寸，不拉伸。单段 73 帧／24fps≈3.042 秒；Avatar 的 AudioWindow 关闭最小上下文扩展并明确交付原录音。生成音色的图不接 `drive_audio`／`final_audio`，交付生成声音，不 mux 参考录音。

长视频 Prompt Relay 三块分别为 Global 固定身份／音色绑定／场景，Local 每行一次台词与情绪，Timeline `percent` 的 `0-35 / 35-70 / 70-100`。本轮测试输入length193，Relay按原生17n+5对齐成209帧计划，实际事件区间约0–3.042／3.042–6.083／6.083–8.708秒；最终交付裁成192帧／8秒，尾部超出部分被裁去。不是精确0–2.8／2.8–5.6／5.6–8秒。需要精确8秒百分比分配时可用本身已对齐的length192，或seconds模式显式给区间；换配置须新验，不复用本轮质量资格。window124/context22，两段边界约5.17秒，不是各4秒。换时长同步计划长度及时间区间，换配置用新 chain_id；参考视频声音可能改变 Audio 编号，按实际 media_map 重写。

模型沿用现有 ComfyUI 目录：H3 权重放 `models/diffusion_models`、Qwen32B 放 `models/text_encoders`、video/audio VAE 放 `models/vae`、Avatar EMA B 放 `models/loras`、原 learned3D 放 `models/latent_upscale_models`、TAEH3 放 `models/vae_approx/taeh3.safetensors`。Avatar 不需新专用模型。素材文件名只是本机示例，请在 LoadImage／LoadAudio 换成自己的合法素材。

详细接线、失败边界及证据见 `docs/AVATAR_PROGRESSIVE_EXP.md`、`docs/TAEH3_SAMPLING_PREVIEW_EXP.md`、`docs/NATIVE_VOICE_EMOTION_EXP.md`、`docs/READABLE_AUDIO_P1_EXP.md`。没有空间采样 tile、不迁移旧图或已验收采样／音频／接缝，不自动下载／运行／发布。
