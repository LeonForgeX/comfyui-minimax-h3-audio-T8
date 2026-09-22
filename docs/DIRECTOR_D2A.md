# 曜石导演台 D2：真实生成桥（D2a–D2c）

D1 的项目、素材、版本与编译合同已经接到正式 Core 队列。当前生成桥不是演示用的假按钮：
服务端会重新编译当前镜头，按实际安装的模型文件建立原生 API 图，通过 Core 的进程内队列
校验、排队、查询、取消和历史输出。浏览器断线只轮询原 `prompt_id`，不会重复提交。

交付前置检查会查找 FFmpeg：优先使用显式 `T8_FFMPEG_PATH`，否则依次检查
PATH、ComfyUI／整合包的 `ffmpeg/bin` 与已安装的 imageio-ffmpeg，不自动下载程序。
自定义路径须为可执行文件完整路径，设置后重启 ComfyUI。该修复不改变编码参数或采样配方。

## 已接入的配方

- D2a：无图文字 T2VA；单首帧 I2VA；模型生成声音。
- D2b：首尾 FL2VA、仅尾 L2VA、Ref2VA 多图/参考视频/参考音频、首帧＋参考图的合法
  Hybrid。编辑器的稳定资产身份会先编译为 `<Picture N>`、`<Video N>`、`<Audio N>`，
  再绑定到 `LoadImage`、`LoadVideo → GetVideoComponents`、`LoadAudio` 的实际输入，不依赖
  CSS 预览冒充送模素材。
- D2c：原音驱动 `record` 使用 `LoadAudio → MiniMaxH3AudioWindowT8`，同一窗口同时接
  `drive_audio`、`final_audio` 和 H3 对齐长度；因为 `lock_source` 会把源音频作为参考，
  首帧＋录音实际编译为 `Hybrid`（不能伪装成 I2VA）。交付音频从 Conditioning 的
  `mux_audio` 输出进入 Trim，不会被解码出的生成声轨偷偷替换。参考音色 `voice` 只接
  `ref_audios`，仍使用模型生成新台词，交付音频保持生成声轨。

模型、VAE、文本编码器和可选 Turbo LoRA 均从当前 Core 的实际目录选择；找不到正式文件时在
排队前失败，不猜路径、不创建坏任务。Ref2VA 使用独立的正式 Ref2VA 权重优先级，不能用
普通文本权重伪装兼容。

## 取消、结果与边界

生成完成后，结果条目使用 Core `/view` 的真实文件信息播放；取消只删除/中断当前任务，失败
不自动重采样，也不删除项目、素材或历史输出。运行中的任务会从 Core progress registry
回显节点进度百分比（Core 版本没有 registry 时仍保留 queued/running 状态）。输出前仍由现有 `MiniMaxH3OutputTrimT8`
和 `MiniMaxH3SafeAVSaveT8Advanced` 负责时长裁切及安全编码。

当前 CPU／API 回归已经覆盖文字、首帧、首尾、Ref2VA、原音驱动和参考音色图合同；隔离
RTX 4060 Ti 又对 T2VA、I2VA、FL2VA、L2VA、Ref2VA、Hybrid、Hybrid-record、voice 各跑了
4 秒／96 帧的真实队列与封装，收据位于 `artifacts/director-d2-gpu-20260919/`。这些证明
节点类型、输入连线、媒体编号、队列、采样和封装合同，不等于 GPU 通用速度、画质、声音、
口型、接缝或新手真人验收；正式人审必须固定素材、seed、尺寸、时长和模型逐项对照。

## 后续顺序

1. 先集中播放 D2 收据，逐条记录首帧／尾帧／参考遵循、口型、响度、原音保真、接缝和取消恢复；
   不以 MP4 存在代替感知验收。
2. 人审通过后才接 D3 的长片/二采、Semantic Bridge、Relay、FastH3/Tao、EAV、LowVRAM、
   TAEH3、Topaz、Meridian；每项必须调用已有正式节点或工作流，不用无效开关代替。

`roadmap.md` 和本文件是本地交接记录，不进入 GitHub、Hugging Face 或发行包。
