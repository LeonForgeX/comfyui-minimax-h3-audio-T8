# 视频扩画：联合解码与原片保留 / Outpainting source modes

v1.76.0 默认联合解码，保留原片模式可选。用户要求停止本轮质量试验，按已知限制发布；这不表示所有素材已经通过画质验收。

| 模式 | 原片区域 | 接缝修色 / 几何校正 | 音频 |
|---|---|---|---|
| `joint_decode`（默认） | 整幅 VAE 解码，原片也会重建，细节可能变化 | 均跳过 | 保留原音轨 |
| `preserve_source` | 有损编码前精确回贴原片 RGB | 按各自开关执行 | 保留原音轨 |

联合解码能避免硬回贴造成的轮廓断口，但不能保证扩画自然。部分片段仍可能有上下条带、重复纹理、动作不连续或接缝。问题尚未被证明全部来自模型，后续优化留待单独开展。

## 工作流与缓存

12 张工作流在 `examples/workflows/27-video-outpaint`，新图显式使用联合解码。先生成短候选，确认后再接续；每条素材仍需自行审看。

候选预览、选择、接续与保存绑定同一种模式。旧归档没有模式字段时继续按 `preserve_source` 解释，不自动切换。底层 Python helper 保留兼容默认；脚本调用应显式传 `source_mode`。

缓存核对素材、模型、Core、KJ 和采样实现身份。升级后不保证旧缓存能接续，也不能修改身份记录绕过检查。已有结果和旧缓存应保留。

## 已验证和未解决的范围

- 完整 32 秒：736×608、768 帧、24fps、22 个串行窗口；严格视频/音频解码通过，原 AAC 包、PCM 和时间线核对通过。画面仍有局部条带和接缝。
- 发布前生产代码对应的全仓 CPU 回归：3174 项通过、0 失败、0 跳过；这不是画质评分。
- 12 张新版工作流已做原生 UI 保存和 API 参数/连线往返核对；本版交付这些已核对的图。
- 6.6 秒严格二值源遮罩仅在隔离副本对照。它没有被集成，也没有完整 32 秒验证，不属于本版算法。
- 游戏、字幕、奇数尺寸、对白、硬切、区域提示和 DLSS 组合不作“全部素材最终画质通过”的承诺。

原片保留报告要求编码前 RGB 哈希一致；联合解码使用独立重建报告，明确 `source_exact_before_encoding=false`、`source_reconstructed=true`。两类报告不能互换。H.264 有损编码后，两种模式都不保证与原片逐像素相同。

## English

The default `joint_decode` mode decodes the entire canvas together, including the original region. It avoids hard pasteback discontinuities but can change original details. Optional `preserve_source` pastes exact source RGB before lossy encoding; visible border seams may remain. Both retain the existing soundtrack. Color Match and geometry correction apply only in source-preserving mode.

This is a release with known visual limitations, not a seamless-quality guarantee. The real 32-second run completed and passed strict media/audio-preservation checks, but some expanded areas still show strips, repeated textures or seams. The isolated 6.6-second strict-mask experiment is not integrated. No additional model is required beyond the documented H3 models; generation requires the supported KJ attention/FFN patches and FFmpeg. Candidate modes and cache identities remain strictly bound.
