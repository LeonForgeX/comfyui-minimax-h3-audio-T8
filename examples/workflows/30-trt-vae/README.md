# TRT VAE：1.78.0 可选 EXP

用途是加速 H3 视频 VAE 的计算，不是换生成模型，也不是超分。旧工作流不变，音频 VAE 不变。本次集中短片和静态文字审片获得整体认可，仍保留 EXP，不保证任意素材和环境。

**当前不保证更快。** 本机短片从 VAE 加载到最终 MP4 的中位耗时为原生 30.54 秒、TRT 31.33 秒；裸解码收益被额外加载校验开销抵消。只作可选 EXP，详细计时范围见下方说明。

| 工作流 | 做什么 |
|---|---|
| `2026-09-10_H3_TRT_VAE_Runtime_Check_EXP.json` | 只检查独立运行文件是否齐全，不占用 GPU 编译 |
| `2026-09-10_H3_TRT_VAE_Compile_EXP.json` | 单独编译一个本机引擎；不自动安装或下载 |
| `2026-09-10_H3_TRT_VAE_T2VA_Decoder_EXP.json` | 原生 8 步生成，只把视频解码交给 TRT；推荐先试这条 |
| `2026-09-10_H3_TRT_VAE_I2VA_Full_EXP.json` | 图生视频，参考图编码和视频解码都走 TRT；可选，不保证更快 |
| `2026-09-10_H3_TRT_VAE_Same_Latent_Compare_EXP.json` | 只采样一次，原生／TRT 顺序解码同一潜空间，保存两份共用同一音频的成片 |

同潜空间对照图先完成原生 AV Decode，再将它输出的 `video_latent` 交给 TRT 解码。两条保存都连接原生分支的同一条裁剪后 AUDIO，不再采一次、不再解码一次音频。输出文件名前缀分别为 `compare_native` 和 `compare_trt`。这是明确标注的对照模板，不是匿名盲测，也不要求为已有审片重新生成。

完整准备步骤、路径及限制见 [TRT VAE 说明](../../../docs/TRT_VAE_EXP.md)。首次导入后必须把引擎占位值换成你自己编译的引擎目录名；完整模式需分别选择 T1 单图、T17 视频编码器。

T1 单图图文件通过随包的 `trt_vae_prepare_t1.py` 显式准备，见安装说明；不依赖本机研究素材，不要用 T17 文件改名代替。本机已有的编码／解码测试结果，不等于其他电脑的安装和运行已经通过。

已完成本次短片验证；按用户要求不做长片验证，也不宣称长片已通过。不自动打开浏览器，不在运行生成时偷偷编译。

## English

Optional EXP workflows released with 1.78.0. The decoder-only example keeps native reference encoding and replaces only video decoding. The Full example also uses TRT encoding and requires separate T1 and T17 engines. Audio VAE and sampling settings are unchanged. Prepare the runtime and compile on your own machine, then replace the engine placeholders. The [setup guide](../../../docs/TRT_VAE_EXP.md) includes the guarded portable T1 export command. The combined short-video and static-text review was accepted overall; long-video validation is out of scope.

No total-time benefit was measured in the current short-clip output-stage test: 30.54s native versus 31.33s TRT including loading, decoding and saving, excluding H3 sampling. Keep this optional; bare-kernel speed is not end-to-end performance.

The fifth workflow is a same-latent native/TRT comparison: one sampler, native AV decoding first, then TRT decoding of that exact video latent. Both output files use the same trimmed AUDIO result; there is no second audio decode or resampling. The outputs are labeled, not blinded.
