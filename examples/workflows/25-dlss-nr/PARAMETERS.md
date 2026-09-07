# DLSS-NR 参数说明 / Parameter guide

## 选哪份工作流？

| 工作流 / Workflow | 用途 / Use |
| --- | --- |
| Runtime Audit | 只检查运行环境，不超分。Environment check only. |
| Image 2x Standard | 图片输入，每张独立处理。Independent still images. |
| Video Frames 2x Standard | 接 IMAGE 帧序列，可继续接其他帧处理节点；整批帧占内存，适合短片。Frames in memory, temporal processing, original AUDIO returned. |
| Video File 2x Standard | 接已有视频文件，逐帧读写，不把整片堆入 IMAGE；长片优先用它。Streaming video file, with source audio packet copy. |

图片选 Image；已有视频文件通常选 Video File；需要和 ComfyUI 中间帧处理衔接时选 Video Frames。
它们不是依次运行的四个步骤，也不是四种画质档位。

## 环境检查 / Runtime Audit

现在无需勾选接受协议。外部许可仍适用；节点不会替用户接受、下载或分发外部程序。
已有旧工作流的勾选值会被兼容忽略；旧五项位置值在前端自动迁移为四项，保留 GPU 和检测模式。

| 参数 | 含义 |
| --- | --- |
| runtime_version | 外部程序版本，默认1.3。不是节点或模型版本。Runtime version. |
| probe_mode | feature_probe_1_frame 实际做一帧小测试，通过后 READY；static_only 只检查环境，不给可运行句柄。 |
| dxgi_adapter_index | 外部 Windows 程序选哪张显卡。通常单显卡用0；31不是“自动”或显存大小。 |
| cuda_device_index | ComfyUI/PyTorch 的 CUDA 设备编号，必须和上面指向同一张物理卡；两套编号不一定相同。 |

输出 dlss_nr_runtime 接到超分节点；ready 表示可用与否；status 是简短状态；report_json 包含检查细节。
READY still requires the verified runtime files, supported driver, device mapping and real feature probe.
No checkbox is needed; applicable upstream/NVIDIA terms remain the user's responsibility.

## 超分参数 / Processing controls

**截图中的 standard 会覆盖全部 nr_* 手动值。要手动调它们，先改 quality_profile 为 custom。**
Named quality profiles replace the NR sliders. Only custom uses the values shown below them.

| 参数 | 含义 / Meaning |
| --- | --- |
| mode | sr_nr：放大+神经渲染；sr_only：只采用超分结果；nr_only：原分辨率神经渲染。 |
| scale | 宽高各放大多少倍；2x 的像素总数为4倍，不是只加一倍像素。 |
| quality_profile | standard 标准、max_detail 强细节、portrait 人像、night 夜景、light 轻度；custom 手动。不是画质排名。 |
| sr_preset | SR 网络选项；default 由运行时选择，E/F/J/K/L/M 不是从低到高的等级。 |
| nr_style | NR 风格：0默认、1自然、2电影感。 |
| nr_preset | NR 内部预设编号，通常留0；不是采样步数。 |
| nr_intensity | NR 增强强度；过高可能改变皮肤和纹理，不保证越大越好。 |
| nr_detail | 最终混入 NR 的比例；0不混入，1完整采用 NR。sr_only 强制此项为0。 |
| nr_color | NR 色彩变化的占比；偏色时可降低。 |
| nr_skin | 皮肤结构控制，-1使用运行时默认值。 |
| nr_local_structure | 局部结构、细纹理控制。 |
| nr_local_tone | 局部光影/对比控制。 |
| nr_global_tone | 整体色调控制，-1使用运行时默认值。 |
| nr_ui_correction | UI/界面元素修正，不是 ComfyUI 的界面开关。 |
| nr_auto_mask | 自动遮罩选项，可针对 UI/文字区域；不保证所有字幕都原样保留。 |
| motion_engine | auto 自动选光流；nvof 请求 NVIDIA 硬件光流；lk 请求 Lucas–Kanade。只影响视频时序处理。 |
| filename_prefix | 保存的文件名前缀和 output 下的子目录，不是模型路径。 |
| crf | 输出视频压缩质量；越小质量越高、文件越大。18是默认，不改变超分倍率。 |

NR controls in order: style, internal preset, enhancement intensity, NR blend, color contribution,
skin structure, local structure, local tone, global tone, UI correction and automatic mask.
A value of -1 for skin/global tone uses the runtime default. Scale multiplies each image dimension;
2x means four times the pixels. The optical-flow selector is for video motion, while CRF controls
output encoding rather than the enhancement model.

SR-only 的上游程序仍会执行 NR 计算，但最终不混入 NR；不能据此宣称该模式一定更省时。
v1.2 仅允许1x nr_only。v1.3 支持1.5x/2x/3x的 sr_only/sr_nr。不支持用1x SR或2x NR-only混搭。

## 常用起点

先用 v1.3、sr_nr、2.0、standard、sr_preset=default、motion_engine=auto、CRF=18。
觉得纹理过重可比较 light 或 portrait；想单独调值再选 custom。效果取舍需要看成片。

Video File 对输入有约束：完整文件、固定帧率、SDR 8-bit、兼容 Rec.709；不支持 HDR、VFR、
旋转/裁剪元数据等未处理情况。原音频保留和校验不因去掉勾选框而取消。
