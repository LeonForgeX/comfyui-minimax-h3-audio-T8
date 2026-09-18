# 独立诊断与可选音频处理（v1.85.0）

导入`2026-09-18_T8_Diagnostics_Optional_Audio_EXP.json`，更换合法录音／视频后运行。
来源诊断和音频说明只读；两个开头音频处理分支均设enabled=false，不影响原声音。

|节点|接法／作用|
|---|---|
|T8 节点来源／运行版本诊断|填写node_id；报告实际provider和可观察版本，不自动重启或删目录|
|T8 音频来源／缓存说明|连接Conditioning／采样等report_json，不是提示词，不改声音|
|T8 开头音频静音＋淡入（AUDIO）|末端AUDIO→处理→原保存分支，fps须匹配视频|
|T8 末端音频静音＋淡入（VIDEO）|原VIDEO→处理→SaveVideo，按真实PTS处理，保留时长|

有开头爆音才试N=1帧、10ms；过大会削首字。不是全片降噪、归一化、修口型或声音克隆。
视频活动trim/crop沿用Core导出，AAC有损编码可有padding，不能当作整个文件恒等。

TAEH3动态预览见[Avatar预览工作流](../36-avatar-voice/README.md)：MODEL→TAE→采样器，
sampler/sigmas保持原接线；默认temporal tiny放models/vae_approx/taeh3.safetensors。
2D tiny另存taeh3_2d_kijai.safetensors后主动选择，不能覆盖temporal模型；2D显示潜帧近似。
暂停只暂停面板，取消绑定当前请求，不清全队列；缓存命中不制造实时新帧。
详细边界见[音频工具](../../../docs/READABLE_AUDIO_P1_EXP.md)和[预览](../../../docs/TAEH3_SAMPLING_PREVIEW_EXP.md)。
