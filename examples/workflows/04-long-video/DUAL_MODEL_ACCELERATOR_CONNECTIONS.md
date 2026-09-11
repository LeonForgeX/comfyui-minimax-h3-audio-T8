# 双模型图怎么接 KJ Sage / Sol

配合本目录的 `2026-09-11_H3_Dual_Model_Long_Video_4plus4_Plain_EXP.json`
或 `2026-09-11_H3_Dual_Model_Long_Video_4plus4_Relay_EXP.json` 使用。
这是未发布候选的接线说明；不修改旧工作流，也不要求全局开启 Sage。

一采和二采各自接一条：

`底模 → 本路 LoRA → 本路加速节点 → model_pass1 或 model_pass2`

可以共用底模 Loader，但两路 LoRA 和加速节点要分开。不要让二采从一采
LoRA 后再串一个 LoRA，否则它会继承前一条补丁，并不是独立模型配置。

| 选择 | 实际节点与参数 | 本轮有执行证据的范围 |
| --- | --- | --- |
| KJ 普通 Sage | `PathchSageAttentionKJ`；`sage_attention=auto`、`allow_compile=false` | 基础图、Relay短片与24秒；EAV整模型证据为下面的省显存分块组合 |
| KJ H3 省显存 Sage | `MiniMaxH3MemoryEfficientSageAttentionPatch` | Relay短片；可接下面的已测分块组合 |
| Sol | `SolAttentionPatch`；修复候选用`tau=0.5`，其余`enabled=true`、`min_tokens=4096`、`strict=true`、`thresh_type=diag`、`int8_qk=false`、`int8_pv=false` | 旧tau1.3人审失败（闪烁/重复人物），不可推荐。增加H3条件/音频精确Q和KV保护后的tau0.5短片画面、音乐人声和口型已获认可；不是通用速度或长片承诺 |
| Core PyTorch | `ModelAttentionBackend`；`attention=pytorch attention` | 基础4+4；用于无Sage/Sol对照 |

KJ省显存的已测分块接法，每路独立复制：

`LoRA → H3省显存Sage → MiniMaxLowVRAMAttention(head_chunks=4) → MiniMaxChunkFeedForward(chunks=2, seq_threshold=4096) → 对应MODEL插口`

不要把多个不同的attention替换器串起来。上表是分别可选的路线，不是让
Sage与Sol同时修改同一次attention。未安装KJ或Sol时可先使用不接加速器的基础图。

## 必须知道的限制

- 新双模型模板默认两段共8秒，显式开启`high_native_mask_exp`二采视频重叠锁定；
  已评KJ＋Relay样片接缝可接受。旧工作流缺省仍为`reference_only`，不会自动改变。
- Sol会保留你连接节点的tau，不会暗改参数或假装调用了其他后端。H3真实布局中
  文本、参考和音频范围按64-token边界向外取整，使用内核的精确Q/KV通路；
  无法核实布局则记录明确的dense回退。只有调用次数通过不代表人物和声音正常。
- 双模型节点新增`color_match`开关，默认开启。只处理续段开头24帧内的有限颜色偏差，
  不插帧、不变更音频、潜空间或总时长，也不能修复人物/场景结构跳变。

- Relay带浮点时间偏置。已适配的KJ组合保留该偏置；不能删掉它换速度。
- 当前Sol内核不支持Relay所需的偏置／不等长query组合。这类调用需要明确回退，
  不能标成Sol稀疏提速；本说明不推荐把Sol接入Relay图。
- EAV保留Stock20要求。要测EAV，一采底模不挂Turbo、`coarse_steps=20`；
  二采可以独立挂EMA B做4步。不要把4步Turbo一采打开EAV。
- `SolAttn_triton`的Morton/shared-hooks路线不等于这里的`ComfyUI-sol-attn`，本轮未认证。
- 两个独立大底模比共用底模更占系统RAM。本机FL2VA／Ref2VA换入换出短片在
  隔离Core关闭锁页缓存后通过；默认缓存曾触发RAM保护。不要承诺任意16GB电脑都够。
- 后端是否生效看运行报告里的实际调用和回退原因，不能只看节点连线。
  当前短片、长片的机器检查不替代最终画面／声音／接缝人审。

固定验证环境：Core `488e8f8ab84592670bcc2ff6a1aa20fabafd5160`，
KJ `b3ec064dde7d122b333660918e1200e928e67ff1`，
ComfyUI-sol-attn `930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf`。
不同版本或未知补丁应重新核验，不通过修改保护检查强行放行。
