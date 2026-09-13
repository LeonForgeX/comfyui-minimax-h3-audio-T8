# Dance 4+4：已验收示例与续段背景修复

2026-09-13，用户完整播放指定8秒样片，画面／身份／细节、运动与接缝、音频三项均为“正常／可接受”。视频 SHA256：`58a8a8fb616a741be4c76f75e5857dff098e0b649959470ac0ba9b8ff7877039`。

导入[已验收 Dance 4+4／KJ 工作流](../examples/workflows/04-long-video/2026-09-13_H3_Dance_4plus4_Accepted_Picture_KJ.json)。模型、素材需自行提供；所保留文件名用于复现，不代表仓库附带权重或人物／舞蹈素材。该示例已进入2026-09-13 GitHub主线更新；旧安装需要更新节点代码并重启ComfyUI后再导入。

## 固定配置

- Ref2VA INT8 convrot，两个独立 EMA B LoRA 强度1；一采512×256／4步，原learned 3D latent upscaler，二采1024×512／4步。
- 24fps、192帧、两段共8秒；window=124，context=22，接缝frame124，约5.17秒。不是两段各4秒。
- 每路KJ Memory Efficient Sage → Low VRAM Attention（head_chunks=4）→ ChunkFeedForward（chunks=2，seq_threshold=4096）。需要对应KJ/Sage环境，不叠加SOL或其他attention补丁。
- `low_context_source=accepted_picture_low_context_v1`，`video_context_mode=high_native_mask_exp`，`second_audio_source=auto`；Relay/EAV关闭，颜色处理保留原配置。
- 实测禁用pinned memory，reserve-vram=5 GiB、vram-headroom=2 GiB。运行约13分57秒，周期采样最低可用显存约2.44 GiB、RAM约36.14 GiB；不是最低硬件要求、精确峰值或通用速度保证。

## 问题如何定位、这次改变了什么

旧样片加强衣着提示词后，第一段人物服装改善，但第二段已生成的独立candidate由白棚跳到暗舞蹈室，片尾再回白棚。原动作源对应位置没有相同跳切；调色报告未应用变换，最终拼接之前变化已经存在。不能归因为“视频拼接造成了背景变化”，也不能说没有上下文：真实HIGH已知前缀和22帧条件均存在。

经用户授权，本次只把后段LOW参考从上一段partial low-x0换成**上一段实际成片末39帧，RGB24解码→已有resize→原视频VAE编码**。音频对象、HIGH逻辑、4+4步表、3D放大器、提示词、seed、模型和其他参数不变。39是保存尾部容量；context22仍选7个latent单元，不强制变成39上下文。

第一段low_x0、high_input、high_output三份tensor哈希与上一条完全相同；第二段确实采用直接前段的[85,124)帧，额外扩散NFE为0。实际HIGH前7个latent单元保持前段成片上下文，最大数值差约2.38e-7，mask已知区0、自由区1。边界RGB差值的邻域比由30.91降至1.356；**用户完整动态审片才是本例视觉接受依据，不是这个指标本身。**

这证明该干预解决了此固定Dance样例的续段问题，不证明所有素材或所有low-x0 continuation都有同一根因。不要追加生成后latent端点平移、视频叠化等失败处理。更新相关路径时同时遵守[通用双采接缝防回归说明](DUAL_MODEL_SEAM_FIX_20260913.md)。

## 后续更新必须保留的防回归条件

1. 后段LOW只能读取同一`chain_id`中**已经接受的直接前段**；必须核对段号、候选状态、视频路径和SHA256，不能取重试片、未接受候选或更早段。
2. 从前段实际成片解码末39帧，沿用现有resize和视频VAE；`context_frames=22`仍决定采样使用的上下文量。不要把“保存39帧”和“采样强制39帧”混为一件事。
3. HIGH原生mask、4+4 sigma／NFE、learned 3D latent upscaler、模型／LoRA、seed和音频交付策略必须保持不变。修接缝时不得顺便换采样数学或加入生成后叠化。
4. 缓存身份必须包含LOW上下文来源策略；策略、模型、素材或提示词改变时使用新的`chain_id`。恢复只允许命中同参数、同父段的已接受结果。
5. 代码更新至少运行LOW上下文、直接父段身份、时间坐标、阶段失败清理及工作流序列化回归；涉及采样／上下文数学时，再跑两段8秒短GPU候选并完整动态审片。静态边界指标不能代替人审。

## 提示词与输入规则

`<Picture 1>`表示目标人物；`<Video 1>`表示当前原动作参考。写明目标脸、发型和图中可见衣着，并明确源视频只提供动作、时序、方向和镜头，不继承其人物和服装。本例的深色卷发、浅灰开衫、浅粉V领必须随新图修改；不要把图中没有展示的下装说成已验证参考。RGB参考是软约束，不是严格骨骼控制，也不保证换任意图都能成功。

先用8–15秒恒定帧率短源。LoadVideo会解码整条源到RAM；连接GetVideoComponents的实际fps，不把30fps源硬当24fps。本次源共362帧；两段内部取0..123、102..225，输出仅前192帧。`source_start_seconds`同时移动动作和原音乐起点。

原音乐通过`aligned_original_audio`接`final_audio`，整条音轨只编码一次。251个AAC包及时间戳与上一条完全相同；AAC仍有损。本次音频接受不代表模型重唱或新生成对白口型已验证。

首次使用新的`chain_id`；`resume_existing=true`仅用于同参数恢复。更换图、视频、提示词、模型、LoRA、VAE、策略、尺寸或时长必须换chain_id，不混用旧缓存。`filename_prefix`仅为输出名前缀。

## 保留的边界

示例导出后核对了当前Core/T8/KJ实际schema、API验证及17节点20连线73显式参数的序列化一致性；不把CPU验证称为新一轮浏览器导入测试或GPU采样。直连KJ的逐kernel调用和回退未独立计数，不能声称实测零回退。

此前失败的原生Dance、旧双采Dance及两条Depth实验保留为本地失败证据，不作为推荐成片或成功模板。本次不重跑已接受Tao/LTX，不推广到任意时长、素材或后端；其他暂停项目不恢复。
