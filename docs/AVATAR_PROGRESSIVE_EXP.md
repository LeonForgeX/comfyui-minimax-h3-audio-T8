# 录音驱动 Avatar 渐进采样（v1.85.0 EXP）

## 正式发布状态（2026-09-18）

用户最终确认全部指定样片通过并要求发布。正式示例见
[36-avatar-voice](../examples/workflows/36-avatar-voice/README.md)，源码在正式`h3_t8`目录。
下面是原始开发／失败和修复记录；旧“未发布／待人审”不是当前状态。
首期仍限独立单段；NA接缝、未知组合和Core同进程驻留问题不被这次验收升级。

2026-09-18 补核此前漏读的真实五组反馈：精确绑定的普通8步与新73帧Avatar4+4
样片画面／音频／口型均“正常／可接受”。接缝NA保持，不扩展为长视频或所有组合通过。
见本地`human-feedback-20260918-v1/analysis.json`；这补足下方历史“仍待人审”的固定配方门禁，
不改写既有失败或TAE同进程Core驻留边界。尚未发布。

新增独立入口 `MiniMaxH3AvatarProgressiveEXPT8`。原渐进节点、旧工作流、已验收双采长视频保持不变；本功能还未发布。

人物首帧和录音接原 `MiniMaxH3AudioConditioningT8`：选择 I2VA、`audio_mode=lock_source`，录音接 `drive_audio`。同一个 DualClock Setup 的 MODEL／sampler／sigmas接新入口，默认 LOW4、原 learned3D 放大、HIGH4。可选独立 HIGH MODEL／LoRA沿用已有接口。

音频 latent 由录音编码，零 audio mask 在两阶段中锁定原生音频状态，HIGH重启噪声和clean anchor明确分离。保存时明确接原录音；不要用VAE重建音轨代替它。这不是单纯贴音轨，也不是按该音色生成新台词的声音克隆节点。

首期只支持独立单段，没有空间tile／长视频资格。无需新的 Avatar 专用权重；使用匹配原生H3、现有EMA B加速LoRA、CLIP、video/audio VAE和原 learned3D latent upscaler。用户的注意力／LoRA组合保留，但未实测组合不自动获得质量／提速认证。

## 示例与帧数

开发候选是人物2:3首帧、512×768，LOW256×384，73帧／24fps。AudioWindow设置时长73/24秒、warmup/cooldown=0，`ensure_minimum_context=false`，不把短测扩成124帧。这个选项只改新图，原节点默认不改。换图按自身比例选择32像素对齐尺寸，不直接套用2:3。

录音需覆盖成片；过短部分补零，过长通过窗口裁切。首字不要被过大的静音／淡入削掉。角色身份、口型、表现和声音仍需人审。

## 资格边界

原phase1入口现在另有专门CPU effects检查：I2VA完整录音零audio-mask条件下，EAV独立和EAV+joint Relay两组合实际LOW4/HIGH4，核对两阶段mask证据／完整时钟不重启、Relay各4次路由、report_only与关闭逐值相同、apply产生视频差异而录音锚不变，原输入和用户wrapper保持。已列入最新618范围回归；这是tiny Core+测试放大器的接口／数值锁定资格，不是新预训练模型组合GPU、人审或提速声明。生产Avatar与已验收接缝代码未改。

2026-09-18 v6真实普通8步与Avatar LOW4→原learned3D→HIGH4均完成，
取消LOW后同进程恢复4+4、收到TAEH3真解码帧后取消再恢复也完成。
独立完整媒体审计逐帧解码三片，确认73帧、512×768、H264、视频PTS递增，
音频有限且非空，媒体原SHA不变。主Avatar恢复片110.375秒，
源录音最大误差1.19e-7是数值锁定检查，不代替口型／画质验收。
审计见 `avatar-v6-media-audit-v1/report.json`。

附加预览开关实验最终video latent差异2.1114254、audio差异0，整个v6
终态因此失败，不能把三片的媒体完整性写成整轮质量／只读通过。
后续OFF/OFF对照也有差异，发生在任何预览解码前；同一已加载模型立即
重复三次原生前向则AV逐值相同。具体跨采样权重驻留／恢复原因仍在定位，
不据此更改旧采样、LoRA、注意力后端或已验收接缝。

追加OFF/OFF诊断已确认首步参数相同而10处实际Core权重读取变化，其中9处在BF16/INT8之间切换。变化发生在任何TAE解码前；详见[预览诊断边界](TAEH3_SAMPLING_PREVIEW_EXP.md)。这不是同进程恢复已修复的声明，也不使旧失败通过。没有为新入口修改旧Core驻留或量化数学。

新独立进程关／开TAEH3完整4+4，八次前向与最终AV全部逐值相同；ON三次真解码。
最终ON latent已独立完整解码73帧512×768 H264与明确原录音，完整AV审计通过，
新候选SHA ec1bb42b40038dd940436c8e9e1cedf10369c2cadf9b34709caab8c697ca726d。
见 `taeh3-fresh-pair-audit-v1.json`、`avatar-fresh-decode-gpu-v1/terminal.json` 和
`avatar-fresh-media-audit-v1/report.json`。不覆盖旧同进程恢复边界，不替代口型／画质人审。

范围CPU回归和实际Core保存图验证已通过；六份前端EXP图在正式节点 `examples/workflows/36-avatar-voice`。
真实预训练模型GPU终态及失败读取对应artifact，不是仅凭节点变绿确认。早期v5两片输出124帧，最后严格73帧断言失败，证据保留，不能算73帧验收。

节点继承原渐进采样器的资源保护。独立测试Core使用6GiB权重驻留保留预算，目的是让Core多卸载权重，未修改用户服务／全局设置；这不是所有硬件都必须设置6GiB的承诺。真实OOM或输入错误正常传出。

首次使用本地新Python需要用户自行重启自己的ComfyUI；测试不会替用户重启、清队列或改画布。机器资格不替代最终人审，不表示任意素材／显卡／模型都通过。
