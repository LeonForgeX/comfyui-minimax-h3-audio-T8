# H16 已完成部分：源码更新与使用说明

基于已发布的 v1.84.0／提交4f54cdb，本次只更新 GitHub 源码和正式节点目录，
版本保持1.84.0，不另建 tag、Release 或 Registry 版本。更新后需自行重启 ComfyUI；
磁盘同步不代表原运行实例已经加载。外部模型／运行环境不随本提交分发。

## 本次交付

| 部分 | 状态与使用入口 |
| --- | --- |
| H16-1 Prompt Relay 字节条件 | 已在基线交付，保持原 raw-byte／token ID 行为 |
| H16-2 官方 ConvRot INT8 VAE | 原生 Load VAE 选择模型，不新增重复加载器、不迁移旧默认 |
| H16-4 H3→LTX | 新标准 LATENT 转换节点；完整73帧转换／解码／公开精修案例已验收 |
| H16-5 Tao 接续 | 既有 Prepared 两节点支持 tao_stream；两请求10秒完整公开链已验收 |
| H16-3 音频边界精修 | 缺外部源码、工作流、日志及原始／精修AV；未实现、未发布 |
| Meridian | 未完成完整生成／前端；不包含转换器、模型或注册准备件 |

H16-3 所需材料已在 [issue18 请求](https://github.com/T8mars/comfyui-minimax-h3-audio-T8/issues/18#issuecomment-5717257140)，
不会用猜测实现替代复现材料。旧单模型／双模型4+learned3D+4、音频、sigma、
接缝配方、Topaz 自动 H264 交付、Sol／Sage／LoRA 与全部旧保存图保持不变。

## 官方 INT8 VAE

来源 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1)，
文件 minimax_h3_video_vae_int8_convrot.safetensors，SHA256
52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e。
本机已放到 ComfyUI/models/vae/h16-official/vae/ 下，原生 Load VAE 选择
h16-official/vae/minimax_h3_video_vae_int8_convrot.safetensors。
其他用户可放在 models/vae/ 的独立子目录，再选择对应名称。
需支持 ConvRot 的 Core／comfy_kitchen；本机资格环境和数据见
[VAE 说明](H16_OFFICIAL_INT8_VAE_QUALIFICATION.md)。

仅 decoder 量化；encoder 仍高精度，不能宣称 INT8 编码提速。同 latent 解码及同RGB回环
两组四片获画面、音乐人声、口型验收，不代表任意素材画质不退步。

## H3→LTX 标准潜空间

导入 [转换保存模板](../examples/workflows/35-h3-ltx-latent/H3_to_LTX_Standard_LATENT_EXP.json)，
选择 H3 .latent，并填写固定外部源码及模型目录：

- [NVlabs/Sana](https://github.com/NVlabs/Sana/tree/144085566a866f9784f3798d4c8d1603f3adbccf) 中
  models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/h3_ltx_adapter。
- [H3-to-LTX-Latent-Adapter](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter/tree/1792c42689a0f22de880eaf57a187c6a373a636d)
  的 config.json 与 model.safetensors 所在目录。

也可直接连接已有 H3 LATENT。第1输出是 LTX 视频 LATENT，第2输出原样保留 H3 联合AV；
不会转换音频／条件／遮罩，不经 RGB、不额外再放大。默认 CPU/F32，
CUDA/BF16 可显式选；不是整条 LTX 精修都可在16GB上跑的承诺。
填写实际 source_frames 和24fps：73帧 exact；124帧需明确 pad129／crop121，
并自行处理相应音频时长，不能静默裁切。

模板只转换并保存，不是完整一键精修工作流。下游 LTX 条件、联合AV输入、
文本缓存与三步精修仍需独立准备；详见 [边界与精修资格](H16_STANDARD_LATENT_ADAPTER_EXP.md)。
原生 LoadLatent→正式节点→SaveLatent 的 CPU 与 CUDA/BF16 逐值对照、原AV／RNG
保持及自有模块清理均已验证。完整73帧832×480三阶段案例人审通过；
不能据此宣称所有尺寸／模型／种子更快、更省或画质不退步。

## Tao 两请求接续

使用已有“读取已准备生成输入”→“串行生成与解码”两节点；
以 tools/prepare_generation_bundle.py 的 --kind tao_stream 生成新的本地 BUNDLE.json，
填写 prepared_bundle_path、noise_seed、新 chain_id、共同 serial_lease_path。
Tao 固定来源为 [TaoMate-H3](https://github.com/TaoLiveAIGC/TaoMate-H3/tree/ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad)。

必须先准备匹配的文本特征、每请求 Base10 教师3/6/9状态与音频种子及模型；
改 JSON 中提示词不产生新条件。CLI 只盘点和哈希，不下载、不合成教师、不运行GPU。
具体清单格式／命令见 [Prepared 接线说明](PREPARED_GENERATION_INTEGRATION_EXP.md)。
原 tao5s／ltx_refine schema 保留。每个原生请求5秒，两请求10秒，
保留同一个模型／KV与全局时间坐标；并非任意时长实时预览。

全部请求成功才提交生成阶段；取消／故障关闭自有 stream，不能恢复半条KV。
完整生成后解码失败可只补解码；改变代码／输入／种子则换新 chain_id，
不能改 receipt 或搬失败阶段冒充缓存。联合解码后一次交付 H264/AAC MP4。
本次修正有界哈希时间预算和 FFmpeg 单线程视频编码，不改采样、音量、CRF或AV时钟。

## 验收范围

2026-09-17 用户提交的人审记录绑定四组八片，ID、顺序、标签与全部媒体SHA核验一致；
review_id 为530eb9ea9cfee12061d17abaf2dd13c4c44e690c4c71a4d20aa137a4d025b1a6。
VAE两组和LTX三阶段画面／音乐人声／口型通过，事件接缝为NA；
Tao两请求四项全部通过，包括事件跟随与接缝。
Tao公开10秒样片SHA为e65749a5de8e68a4671ae81c22d87ccab43dd613d9d473895d2aac487afd46b6。

两组真实 Tao 取消／故障分别在4次前向后清理，再在同Core运行普通H3完整8步，
73帧AV成功、自有子进程为0。两恢复片RGB／PTS／AAC包／PCM完全相同；
没有“运行Tao前”控制片，不宣称它与未知原始基线完全等效。
冻结正式源上一轮645项范围CPU通过，其中包含本次不发布的 Meridian CPU研究测试；
不借这个数字宣称 H16-only 新提交全仓通过。新的源码／实际Core／包检查单独执行。

失败实验与原始机械、人审收据保留在本地，不分发人物图、视频、模型、私密路径、
ROADMAP、SKILL或交接文件。所有证据只适用于所列配方；其他素材与硬件须独立验证。
