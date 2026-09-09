# H3 视频扩画

这里有 12 张独立工作流。扩画不会改变旧工作流，也不需要新的专用模型或转换权重；它使用现有 H3 FL2VA、Qwen3-VL、视频 VAE 和音频 VAE。

v1.76.0 按已知限制发布这 12 张 EXP 工作流，默认联合解码。完整 32 秒已生成，但部分片段仍有上下条带、重复纹理或接缝；不要把“能生成”理解为所有素材画质都通过。先看短候选，再决定是否接续。6.6 秒严格遮罩对照仍是隔离实验，本版没有替换采样算法。

## 两种原片处理方式

- `joint_decode`：新节点默认。整幅画面一起解码，原片区域也会经过 VAE 重建，细节可能变化；跳过原片边缘 Color Match 和几何校正。
- `preserve_source`：可选。在有损编码前精确回贴原片像素，Color Match 只处理扩区并在切镜时重置；边界可能仍有断口或接缝。“编码前精确”不等于最终 H.264 解码后逐像素相同。

两种模式都保留原音轨。候选一旦生成，接续与保存沿用它的模式；没有模式记录的旧候选仍按 `preserve_source` 处理，不会因为新节点默认值改变而自动换模式。旧工作流请检查实际 `source_mode`，不要仅凭文件名判断。

## 第一次使用

1. 打开 `Geometry_Preview_EXP`，上传原片并调整四边像素、目标比例或锚点。蓝色棋盘只是待生成范围，不是效果图。
2. 推荐按 `01_Generate_Candidate` → `02_Review_Candidate` → `03_Confirm_Candidate` → `04_Continue_Selection` 运行。
3. `03` 的确认开关默认关闭。只有看过候选并主动确认，`04` 才会继续整条视频。
4. 如果整片已经采样完但最后保存失败，使用 `05_Save_Completed_Selection`；它只加载视频 VAE，不会重新采样。

`Four_Stages_EXP` 是完整的 Plan → Prepare → Sample → Compose 单图版本；`Compact_EXP` 只是把同一条四阶段链折叠成子图，算法和默认值没有变化。

## 区域提示和超分

- `06_Generate_Guided_Candidate` 允许给上下左右扩画区写不同提示词，也可以用原片坐标填写人物框进行边界风险审计。
- 区域候选仍然要经过 `02`、`03`，然后用 `07_Continue_Guided_Selection` 接续。普通路线和区域路线不要共用 `run_name`。
- `08_Save_Completed_Selection_DLSS_2x` 只处理已经完成的选择。它需要另行安装 DLSS-NR v1.3 外部运行时；新版运行时审计节点不再要求勾选许可开关，但外部软件许可仍适用。不需要新的 safetensors。
- `09_Model_Compatibility_Audit` 只检查实际 MODEL 上的 LoRA、Attention 和 wrapper，不生成视频。

## 需要的文件

正式样例默认选择：

```text
ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors
ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

还需要：

- [`ComfyUI-KJNodes`](https://github.com/kijai/ComfyUI-KJNodes)，正式生成图使用其中的 `MiniMaxLowVRAMAttention` 与 `MiniMaxChunkFeedForward`
- `ffmpeg` 可从 `PATH` 直接调用

范围预览、候选读取、确认和兼容审计本身不加载 H3 主模型。当前正式生成路线固定为 Stock20 原生噪声，不要叠加 Turbo、SPEED、SLA、OpenVDN、FastH3、Prompt Relay 或其他 MODEL/Attention 接管节点。

## 输入与输出

- 输入视频必须是未裁切的 24fps、SDR、CFR 文件；修改素材、画布、提示词、CLIP 或 VAE 后请换新的 `run_name`。
- 只换 seed、步数或采样 MODEL 时，换新的 `candidate_name`。
- `candidate_id` 和 `selection_id` 都是完整的 64 位文本，必须原样复制，不能填路径。
- Color Match 开关默认开启，但只在 `preserve_source` 模式处理新增区域；`joint_decode` 不做边缘修色或原片回贴。
- 恢复缓存会核对素材、模型和运行配置；升级 Core、KJ 或节点代码后，不能直接忽略身份检查继续运行。
- 原片音轨按包、时间线和解码 PCM 校验后复制到成片。无声源不会被静默补成音乐或对白。
- 为提高本机解码可靠性，成片使用全帧内 H.264；体积较大是预期取舍。任何机械检查都不能代替人工检查扩画边界、人物、动作、切镜和声音。

参考实现：[`TwoAbove/ComfyUI-H3VideoOutpaint`](https://github.com/TwoAbove/ComfyUI-H3VideoOutpaint)，审计固定到提交 `27df0ff0538896dd494e3541c5a374cf2fe00aab`。T8 版本没有复制其运行时，而是把几何、源画面/原声保护、串行缓存、人工候选门、区域提示、Color Match 和安全交付接进本节点现有 H3 音画链。
