# 渐进双采长视频：公共实验节点

仅独立开发树提供，尚未部署。真实浏览器保存/重载/API往返已有证据；完整模型两段8秒已完成机械核验和零新增采样的交付恢复。其余控制变量组合、人审与最终包验证仍待完成。
这不是替换旧双MODEL长视频节点，也不把其已通过视频当作新渐进算法的质量证据。

## 接线

每路：原生H3底模 → 本路兼容H3 LoRA链 → 本路一个已适配后端。
两路送入 `MiniMaxH3ProgressiveSetupEXPT8` 的 `model`／`model_hires`。
Setup只配置完整原生Euler时间表，不加载权重、不生成视频，也不需要预先编码一遍提示词。
输入MODEL不被原地修改。不接HIGH时复用LOW源配置；结构、latent格式和音视频时钟必须匹配。

Setup输出的两路MODEL、sampler、sigmas → `MiniMaxH3ProgressiveLongVideoEXPT8`。
另接CLIP、视频VAE、音频VAE。选择现有H3 learned3D latent upscaler，不自动下载替代权重。

需要TST时，在Setup与长视频之间，两条MODEL支路各接独立TST节点；
两个TST均接Setup的同一份完整SIGMAS。不要先加TST再重新配置它所绑定的模型时间表。
EAV使用长视频内部 `eav_mode`，不能把已装独立EAV包装的MODEL直接送入本路线。

## 关键参数

| 参数 | 用法与边界 |
| --- | --- |
| Setup `steps` | 完整时间表总步数；8步配LOW4即4+4，LOW6即6+2；HIGH至少1步 |
| `width`／`height` | 最终画幅，32倍数；测试候选896×448约0.4MP |
| `low_scale` | LOW宽高比例，按32对齐；0.5是448×224，不保证4倍提速 |
| `total_duration_seconds` | 输出时长，不强制30秒；8秒对应192输出帧 |
| `render_window_frames` | 单段生成窗口；候选124帧。两段8秒不是每段4秒，接缝约5.17秒 |
| `context_frames` | 实际使用22或39帧上下文；保存39帧尾部容量不等于实际必须选39 |
| `global_prompt` | 不接Relay时的持续场景描述；单次台词不要重复放到全局 |
| `segment_prompts_json` | 原长视频分段提示输入；接Relay时以其投影后的全局/局部事件为准 |
| `guide_resize` | 默认legacy_bilinear不改旧行为；preserve_mean为未完成画面对照的可选项 |
| `context_audio` | 是否在续段使用音频上下文；不是输出音轨的选择器 |
| `audio_mode` | native生成/现有源语义；lock_source保留源约束；remix_source重混；非native须有drive_audio |
| `drive_audio`／`final_audio` | 输入完整时间轴音频，按每段含上下文窗口截取。显式final_audio优先作输出音轨，不代表其参与生成 |
| `first_frame` | 只给第0段的单首帧I2VA；本节点不宣称多参考、尾帧或身份参考等全功能 |
| `resume_existing` | true在每次执行重新核对真实模型、源、设置和磁盘回执后恢复；false在OS锁内拒绝已有数据，不删除文件 |
| `chain_id` | 恢复身份命名空间；改模型/LoRA/源音频/提示/时长/尺寸使用新ID，不搬运失败阶段到新链 |
| `filename_prefix` | 输出文件名前缀，不改变链身份或选择另一条缓存 |
| `audio_seam_policy` | cosine_bridge仅作音频交接；audio_bridge_ms默认5ms，不是视频叠化或潜空间桥 |
| `eav_mode` | disabled关闭；report_only仅测量；apply_exp实际应用。完整8或20步、CFG1；进度按1−视频sigma，HIGH不重置 |
| `reserve_vram_mib` | 节点边界余量检查，不是整机显存上限；不能保证不会OOM |

## Relay规则

全局：持续的人物、衣着、环境和声音氛围。
局部：逐行列出只发生一次的动作/台词；时间表与事件数量一致。
手填0–100百分比要选 `timing_mode=percent`，不是auto_equal；auto_equal不会按手填百分比执行。
Relay计划长度必须覆盖完整输出。候选8秒使用193帧计划覆盖192帧输出。

生成对白使用 `joint_av_exp`。锁源音频时用 `video_only_paper`，不要将已锁音频继续作为Relay待生成Query路由。
指定Sol遇到Relay时间偏置/不等长Query会明确回退；可运行不等于实际使用稀疏加速。
不要串联Sage与Sol替换器，也不要把SolAttn_triton当成已认证SolAttentionPatch。

## 本次实际验证边界

新增两个节点追加在独立TST之后，旧节点不移动/替换。
真实CPU服务完成含两路TST、内部EAV、jointAVRelay的完整公共API图验证；
候选JSON由真实object_info生成，保存在测试artifacts内，明确标UNREVIEWED，不放进已通过示例目录。
CPU检查不是浏览器加载/保存/API导出往返，也不是完整模型推理。
最终必须补真实权重0.4MP两段8秒与完整动态人审，未通过不得发布质量结论。
