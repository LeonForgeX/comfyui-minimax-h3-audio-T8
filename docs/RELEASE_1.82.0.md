# v1.82.0 — H3 契约加固与双模型 T8 低显存内循环

## 验收与发布范围

代码、CPU 合同、工作流往返和两段共 8 秒的真实 GPU 机械验证已经完成。用户确认旧硬边界
候选的人物正常、其余画音正常，但约 5.17 秒处背景突然更换，因此旧候选不通过。新的 EXP
候选保持 accepted-picture LOW 上下文，并把 HIGH 从精确锁定前缀经过三个 latent 单元按
`0.25 → 0.50 → 0.75` 渐进释放；音频和值域不变。渐释源已完成机械审计与新进程续跑检查。
后续时间稳定B和局部C复用同一源成片；用户最终接受C并明确要求发布，轻微接缝变色仍未完全消除。

用户随后复核渐释候选，确认整体“好多了”，但仍能看到轻微颜色跳变。执行报告证明旧的
Color Match V2 并未遗漏：它已经完成五帧参考、Lab 分布、8×5 局部残差和24帧渐退；残余峰值
实际位于续段第2／3帧的短促偏暗—回亮。新增的 `bounded_spatial_temporal_exp` 只在 V2 之后
对续段开头12帧的低频 RGB 均值做有界时间中值稳定，不混合相邻画面、不改人物／背景几何，
也不接触音频。旧工作流和缺省值仍为 `bounded_spatial_v2`，新模式必须使用新的 `chain_id`。

用户随后复核时间色彩 B 表示“基本可以”，但希望进一步减轻局部变色。独立
`bounded_motion_color_exp` 在旧时间稳定之后增加运动置信度／前后时刻共同支持的局部低频
修色，RGB／亮度／色度分别限幅，不挪动输出像素。当前仅复用已完成的8秒媒体生成 C；
B 原文件及音频保留。C已通过指定范围的人审；推荐工作流保存2:3首帧控制组合，详见
[局部修色说明及证据边界](MOTION_COLOR_EXP.md)。受影响干净树测试189项通过。

## 完成内容

- Core H3 VAE 合同同时识别旧的全局 token 坐标接口和当前 decoded-pixel overlap-blend 接口，
  覆盖普通／分块 decode、参考 encode、布局、dtype、设备与明确失败语义。
- 语音和时间线回归覆盖 described/reference voice、旧显式 `none`、弱辅音、前后静音、关键帧
  audio/video-audio row 和续段 PTS；参考音频不被误写成原声锁定。
- 双模型阶段缓存改为精确 schema；同名媒体换内容、同路径 LoRA 换内容、后端切换、模型 wrapper、
  accepted picture、失败／取消和跨项目状态均有失效或隔离验证。
- PDD wrapper 使用私有所有权 token，覆盖 native→PDD→native、失败恢复、双分支、重复 patch/eject，
  确保原 final forward 与 adapter 生命周期恢复。
- 前端工作流回归覆盖旧图导入、复制、保存／重载、API 往返、动态 widget、optional socket、空角色、
  重排和未知表达式；工作流总数从 254 增加到 255。
- 双模型 4+4 内循环现在可以让 LOW/HIGH 分别连接本项目 `LowVRAM Attention` 与
  `Chunk FeedForward`，并与 Prompt Relay 共存。身份合同绑定实际权重、LoRA、参数、object patch、
  runtime token、wrapper 和源码 SHA；未知或被篡改的组合继续 fail-closed。
- 新工作流：
  `examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json`。
  两路默认独立使用 `head_chunks=4`、`chunks=2`、`seq_threshold=4096`，不依赖 KJNodes。
  推荐图为LOW256×384→HIGH512×768（2:3）、首帧1024×1536不随包分发；保持比例不拉伸。
  显式选择`bounded_motion_color_exp`与新chain，保留相同Relay三事件与时间线；旧图和节点缺省不变。
  该图是原GPU渐释源加已接受离线B/C控制的组合，不是新图完整GPU重跑或逐位复现保证。
  `head_chunks=1` 仍可显式选择，但本轮用户复核发现续段接缝明显，不能作为交付默认。
- 工作流显式使用 `accepted_picture_low_context_v1 + high_native_mask_ramp_exp`。HIGH 已知前缀
  保持精确，随后的三个 latent 单元才渐进释放；旧 `reference_only` 与硬锁
  `high_native_mask_exp` 的 schema、默认值和旧 JSON 行为不变。

## 验证证据

- 修改后受影响范围的 290 项 CPU 回归通过；Ruff、Python compile、工作流 JSON 结构和
  `git diff --check` 通过。该结果包含渐释模式、真实新进程续跑探针、工作流生成和待人审发布门，不复用
  早于渐释改动的旧计数。
- 实际 ComfyUI Core：`36da3ff763687eab86a35e1019995dd1fb369b0d`。
- 唯一 GPU 验证严格串行运行两段共 8 秒；每段执行 LOW4→学习型 latent 放大→HIGH4，最终
  896×448、192 帧、24fps，H.264/AAC 音视频严格解码通过。
- h4+c2 每段 LOW/HIGH 都观察到 4 次网络 forward、800 次 Prompt Relay 路由调用，以及
  认证的 T8 `head_chunks=4`、`ffn_settings=[2,4096]` 执行回执；端到端 627.85 秒，整机
  轮询最高观察约 12.48GiB 已用、最低约 3.52GiB 空闲。
- h1+c2 机械审计同样通过且端到端 520.79 秒，但用户确认其续段接缝明显，已明确淘汰。
  h4+c2 首片接缝暂时正常；两路原音乐厅提示词均有移动光影。
- h4+c2 首条机械审计：`artifacts/t8-memory-dual-8s-gpu-20260916-v1/t8-memory-audit.json`；
  媒体 SHA-256：`85e05a8f5efbdc00f1eec00ade7610f90ffea4f7ae100236aaa06639a0e4c514`。
- 固定背景复核使用平面 2D、均匀浅灰背景和固定定位线，并明确禁止渐变、高光、阴影、反射、
  辉光、光斑、曝光／色温变化及亮度呼吸。真实 h4+c2 两段 8 秒仍生成移动彩色光圈，说明该
  现象不只是提示词要求的光照内容，更可能是模型／风格生成伪影。第 124 帧全局平均亮度跳变
  约 0.50%，角落背景约 0.71%；这些机器指标不能替代接缝人审。
- 固定背景媒体 SHA-256：`e0b1ccb4202d2d057c4b2e23c008dd1e3f7586342c3e41ee71741f9277be1cc6`；
  机械审计 SHA-256：`93ce097ce326fb91c599f7f3ea565d7b19b01964f9b8c07c4861d581e5eca9e3`；
  绑定审片 ID：`42ded616758bf001a2dbe0139b99ebd4685e285c2121ffbad86078deabb8a73f`。
- 用户随后提供的 2:3 外滩 I2VA 对照确认旧硬边界在人物正常时仍会令背景突然更换。渐释候选
  保持同一首帧、提示词、seed、4+4、h4+c2、512×768、22 帧上下文和音频，只把 LOW 上下文
  改为上一段已完成画面，并将 HIGH 释放策略改成三单元渐变。最终 192 帧、24fps、H.264/AAC
  严格解码通过；两段 LOW/HIGH 都有 4 次 forward、800 次 Relay 路由和 h4+c2 回执。
- 渐释候选媒体 SHA-256：`b8659a058204e917d8bcab808c6abcfb9a86f6e86f10f87159419ed9381b23e9`；
  机械审计 SHA-256：`708371b4a2843d41077ec19074be36e6581db2ae0ab013ee4862797a67b154bd`；
  审片入口：`http://127.0.0.1:8804/review.html?v=seam-ramp-final`。逐帧检查未再看到硬边界
  候选的瞬时斑块，背景楼体／栏杆构图连续，但这仍不是用户人审通过。
- 两版前 124 帧解码 RGB 逐像素相同。相对旧硬边界，渐释版在 123→130 后段窗口内的背景
  MAD 峰值降低 33.9%、95 分位变化降低 26.4%、大变化像素比例降低 45.8%、光流峰值降低
  13.3%、边缘突变降低 16.4%；直接段界全局 SSIM 从 0.85283 升至 0.86496。诊断回执为
  `artifacts/t8-memory-dual-8s-h4c2-bund-i2va-apc22-ramp-gpu-20260916-v2/background-seam-comparison.json`
  （SHA-256 `7a32dabac8bc09f543f3cdaf911a58142b4b0408d4674e1e47b3d6efae0e0d32`）；
  这些非预注册机器指标只用于定位旧斑块峰，不能替代感知验收。
- 同一已完成链随后由全新隔离 ComfyUI 进程以 `resume_existing=true` 重新进入；结果为
  `returned_verified_existing_final`，两段审计都被重新校验，30 个 manifest／候选／上下文／
  阶段缓存／终片文件的尺寸和 SHA-256 前后完全一致，最终媒体 SHA 保持上述值且没有重采样。
  回执为
  `artifacts/t8-memory-dual-8s-h4c2-bund-i2va-apc22-ramp-resume-20260916-v1/terminal.json`，
  SHA-256 `692dd8d9592be713bd5da017678185118b5342453009b111b43a5e6843d9dacb`。
- 在不重采样的 A/B 中，B 只改变解码后的第124、125、126、129帧，其他帧解码像素与 A
  完全相同；两边252个 AAC 包载荷 SHA-256 都是
  `780599a1c52e0340c77729bb699729a7096e5c35f54151e76ff817b3076e164f`。接缝附近相邻 RGB
  均值峰值从0.02347降至0.00344（约85.4%）；这是机器色彩连续性证据，不替代人审。
- 时间色彩 A/B：`http://127.0.0.1:8805/review.html?v=temporal-color-v1`；输出 SHA-256
  `b6362cfe6a3a69aaaf03ca698f50625b2bda2e59d3e74d5160c26c720fe12cd3`，审计见
  `artifacts/t8-memory-dual-8s-h4c2-bund-i2va-apc22-ramp-color-temporal-review-20260916-v1/`。

## 边界

- 用户接受“仍有一点”变色作为当前版本的已知限制，后续另做局部／时序颜色研究；不再追加生成。
- 验收C SHA为`0cf5413eb9715d3a9b730b8025a5f71d2f1ed8622475095927389e0895e69d11`，
  修色审计SHA为`58ba1960595ff35b431d7d276f7017992af0566224634bb9d9f1b209dc46eb1e`；
  人审记录SHA为`280131c7a8eb41801b92c81b3c49bde5465b111b069ec63f43918e478b757100`。
- 本次不新增节点 ID；旧 336 个节点顺序、旧工作流和默认值保持不变。
- 不承诺所有 16GiB 显卡安全，也不承诺通用省显存、提速、画质无损或 bit-exact。
- T8 LowVRAM/ChunkFFN 不能与占用相同 H3 forward 的 KJ 同名节点叠加。OpenVDN、EAV、TST、
  SLA、FastH3 和其他第三方 attention 组合仍须独立短片验证。
- 改变 LOW/HIGH 任一路模型、LoRA、内存参数、Prompt Relay、尺寸或时长，必须使用新的
  `chain_id`；不得复用旧阶段缓存。
