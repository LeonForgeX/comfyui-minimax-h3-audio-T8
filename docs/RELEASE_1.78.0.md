# 1.78.0 — 可选 TRT VAE

新增 4 个独立 EXP 节点和 5 份工作流，节点总数 324，工作流总数 231。旧工作流、H3 生成模型和音频 VAE 不变。

- 安装检查：检查运行文件是否齐全，不编译、不下载。
- 本机编译：显式编译 decoder-flex、单帧/视频编码器或 W4，资源不足时停止，保留旧引擎。
- Decoder：只替换视频解码，参考编码继续原生。
- Full：同时提供 TRT 参考编码和视频解码，需要独立 T1、T17 引擎。
- 同潜空间对照工作流：只采样一次，两路顺序解码，共用原生音频。

安装步骤及模型目录见 [TRT VAE 说明](TRT_VAE_EXP.md)，工作流在 [30-trt-vae](../examples/workflows/30-trt-vae)。运行文件和模型不随节点分发。推荐先试 Decoder；Full 可能改变后续生成。

本次 8 组短片对照和静态文字对照获得用户整体认可，不推断逐项评分。覆盖普通解码、图生视频、渐进采样、VDN 二采、W4、参考图/视频编码影响，以及人脸裁剪编解码回贴；后者不是新的修脸采样。长片测试已按要求取消，不能声称长片通过。原生 VAE 重建小字也会损伤笔画，本次验收不保证任意文字精确保留。

裸热解码中位约原生 22.96 秒、TRT 10.64 秒；但包含加载、解码、传输和保存的短片测试为原生 30.54 秒、TRT 31.33 秒，**未测得总耗时收益**，不默认替换原生。测试设备为 Windows / RTX 4060 Ti 16GB / TensorRT cu13 10.13.3.9.post1，其他环境需自行核验。

另修复 Windows 子进程快速退出时 Job 计数暂未归零造成的误报；真正残留子进程仍会被清理。已有专项与实际包导入检查覆盖该行为。

## English

Adds four optional TRT VAE EXP nodes and five workflows (324 nodes / 231 workflows total). Existing workflows, H3 weights and audio VAE remain unchanged. Prepare a separate runtime and compile engines locally; no model, DLL or engine is redistributed. Start with decoder-only; Full reference encoding can affect subsequent generation.

The combined eight short-video comparisons and static-text chart received overall user acceptance, not invented per-item scores or universal quality certification. Long-video testing was explicitly cancelled. Native VAE reconstruction can also damage small text. W4 remains optional and has larger numerical differences.

No measured total output-stage speed benefit: native 30.54s versus TRT 31.33s with loading/decoding/saving, excluding H3 sampling. Bare warm decode alone was 22.96s versus 10.64s. A bounded Windows Job exit-accounting check also prevents false failures while retaining actual descendant cleanup.
