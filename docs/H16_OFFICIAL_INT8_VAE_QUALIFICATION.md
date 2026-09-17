# 官方 H3 INT8 ConvRot VAE：本机编码与解码资格

状态：同 latent 解码及同 RGB 编码/往返解码机械检查完成；2026-09-17 两组四片
画面、音乐人声、口型均获用户验收。事件／接缝为不适用，不能改写为通过。
当前交付与验收范围见 [H16 源码更新](H16_SOURCE_UPDATE_20260917.md)；
下方“待集中人审”等是历史实验步骤，不覆盖此最新状态。
没有修改任何现有 VAE 默认值、TRT 路线、采样器或音频。

## 模型与使用入口

固定来源为 Comfy-Org/MiniMax-H3 revision
`7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1`。
新模型 SHA256：
`52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e`。

本机使用原生 Load VAE，选择
`h16-official/vae/minimax_h3_video_vae_int8_convrot.safetensors`。
单独目录用于避免覆盖本机已有但内容不同的同名模型。
不新增重复的“量化 VAE 加载器”，旧工作流不会自动迁移。
本次实测 Core `36da3ff763687eab86a35e1019995dd1fb369b0d`，
comfy_kitchen 0.2.34、Torch 2.10.0+cu130，RTX4060Ti 16GB。

## 单变量实测

复用真实 Bridge 原版短片采样输出，video latent shape
`[1,24,22,30,52]`。两模型读取同一内容 SHA；零次新扩散前向。
各做 cold/hot/full native-tiled、1帧、5帧、22帧及816×464边界解码。
H3 的 `handles_tiling` 接管原生 tile API，不能称为额外独立切块算法。

| 完整73帧832×480 | FP16 | 官方INT8 |
| --- | ---: | ---: |
| 冷解码秒 | 14.04 | 6.86 |
| 热解码秒 | 12.58 | 4.90 |
| PyTorch峰值已分配GiB | 5.38 | 3.08 |
| 完整解码ConvRot INT8调用 | 0 | 1728 |

同一路线 cold/hot/native-tiled 输出逐位相同。INT8所有七项累计
8208次INT8调用均完成并带ConvRot，不是静默回退普通浮点路径。
这些数字不含完整视频生成成本，不代表其他尺寸或机器的性能承诺。

两版本解码浮点画面的MAE约0.000459、PSNR约62.38dB，
最大绝对像素差约0.07175（0–1范围）。统计不等于画质验收；
必须检查细节、颜色、闪烁及尾帧。两份H264预览保留源AAC，
完整帧数、时长及音频packet身份已由工具核对。

## 可复核证据

隔离开发树内：

- `artifacts/h16/vae-fp16-decode-v1/terminal.json`
- `artifacts/h16/vae-int8-decode-v1/terminal.json`
- `artifacts/h16/same-latent-comparison-v1.json`
- 各目录 `preview.mp4` 与 `decoded-float.safetensors`

## 同 RGB 编码及往返解码补验

固定同一份73帧832×480浮点RGB，零扩散采样。1、5、22、73帧及原生
encode_tiled五项输出均有限，FP16模型与官方INT8模型的编码latent逐位相同。
73帧latent为`[1,24,22,30,52]`，两者各自普通/原生tiled输出也逐位相同。

必须区分模型内的量化范围：官方文件的144个`.comfy_quant`条目全在decoder；
encoder及quant_conv仍是浮点层。因此编码时INT8调用为0是正确路径，不是回退。
首次INT8编码探针误要求编码发生INT8调用，在五项输出完成后断言失败；原失败
记录`vae-int8-encode-v1`保留。修正探针后`vae-int8-encode-v2`完整通过，
随后73帧往返解码实际完成1728次ConvRot INT8调用。

完整73帧编码本次约15.87/15.85秒，峰值allocated约6.07/3.81GiB（FP16/INT8）；
不能称编码算子INT8加速，驻留VAE的解码器体积也影响峰值。往返浮点图像MAE
0.00037339、PSNR64.44dB、最大差0.08050；这不是画质接受结论。
两份H264预览保留原AAC包并通过严格音视频解码。

证据为`artifacts/h16/vae-fp16-encode-v1/terminal.json`、
`artifacts/h16/vae-int8-encode-v2/terminal.json`及
`artifacts/h16/same-RGB-encode-comparison-v1.json`。比较工具CPU核对模型、
RGB、Core、原探针保留副本及产物身份，未初始化CUDA。

后续集中审片，不重新采样已有VAE对照。没有自动替换默认VAE，
上述有限样例不代表所有I2VA/Ref2VA尺寸、参考数量和长视频组合均已兼容。
