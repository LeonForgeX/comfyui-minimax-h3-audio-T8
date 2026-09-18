# v1.85.0 — Meridian / Avatar / voice / preview

2026-09-18，用户已确认「都没问题了，保存正式工作流，推送github」。
本次发布此前开发并完成指定范围验证的五项功能，不迁移旧工作流、不改Core。

- Meridian：四个独立节点，IMAGE／VIDEO素材窗口、授权Omega几何、原生merged-DMD
  ConvRot INT8、三次前向、空间相机轨／源时间轨编辑器、H264 MP4交付。
  `slide`现在平行移动相机与look，朝向不变；已保存canonical路径不自动改写。
- TAEH3：可选LOW x0动态预览，时序tiny及独立2D tiny；绑定当前请求的取消与恢复。
  关闭时MODEL直通，不增加扩散步骤；2D预览不是连续24fps最终影片。
- 来源诊断／音频说明：四个独立节点，含末端AUDIO或VIDEO开头静音＋淡入。
  不自动重启、不做响度归一化；无爆音不必加处理节点。
- 原生音色／情绪：复用H3原生`ref_audios`，新台词由模型生成，不贴参考音轨。
  正式示例含Stock20、狂怒、标准4+4、20+4独立二采、两段8秒内循环与EAV对照。
  Global不重复对白；Local每句一次；短话短窗，余下时间明确闭口和无声动作。
- Avatar第一阶段：独立录音驱动入口，LOW4→原learned3D→HIGH4；录音latent在两阶段
  锁定，交付明确使用原录音。复用现有H3／EMA B／VAE／upscaler，无虚构专用权重。

## 正式工作流

保存位置在本节点`examples/workflows/`，不是外部独立开发目录：

- [36-avatar-voice](../examples/workflows/36-avatar-voice/README.md)
- [37-meridian](../examples/workflows/37-meridian/README.md)
- [38-diagnostics-preview](../examples/workflows/38-diagnostics-preview/README.md)

先换成自己的合法图片／录音／视频。示例人物图2:3，512×768，不拉伸；其他比例须同步尺寸。
Meridian用训练bucket及明确等比ROI裁切，不保证完整原图画布；相机warp预览的灰洞不是最终画质。
授权Omega由用户自行取得，不在GitHub重分发；资产安装见Meridian工作流说明。

## 验收与保留边界

九组18片原反馈以及最后两组四片补测按review_id和每条媒体SHA绑定。
直接狂怒新片SHA `9e2d928754cedffec35ec65af2a7cf5f1b2c5515353856236477df4bf83a195b`；
平行横移新片SHA `5a6c0add1c4f002d4f3afbd4f41423f49516ec015153aaff9dfcdf237613bb13`。
最后真人确认补足这两项门禁，不复活旧横移失败或声音不足样片。
固定路线画面、正常声音、口型接受；Meridian静音片的声音评分不冒称生成语音验证。
接缝／事件评分为NA，不升级为通过；不保证逐字、词级时间或100%声纹克隆。

已有源绑定范围回归618项、Meridian100项、实际编辑器源码fake-DOM23项通过；
发布前另验证正式保存图在当前Core的接口及往返。它们不是全仓、任意组合或所有平台认证。
新独立进程TAE关开完整4+4逐步和最终AV精确相同；已复现的Core同进程LoRA权重驻留
差异发生在预览之前，尚未解决，不冒称任意同进程恢复恒等。
EAV+H4仍报告`composition_verified=false`；人审不证明其全头数学等价。
Avatar首期无空间sampling tile或长视频GPU资格；不保证16GB更快、更省或永不OOM。

原343节点顺序保留，追加11个节点（共354）；旧模型、Sol、Topaz、Semantic Bridge与
已验收接缝配方不迁移。ROADMAP和SKILL仅本地，不上传；大型模型及本地审片记录不进GitHub。
更新后请自行重启ComfyUI载入正式源码，本次不操作用户前端或队列。
