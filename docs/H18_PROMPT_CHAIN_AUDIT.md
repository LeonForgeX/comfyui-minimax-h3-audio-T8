# H18-1：提示词、角色、Bridge 与 Relay 组合核验

本轮选用既有正式双模型图：
`examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json`。
**没有改这份工作流，没有运行 GPU、生成新成片或发布。**

## 发现的问题与最小修复

预算节点保留原文，但后接 Relay Plan 时，Global 和每行 Local 的 `.strip()`
会移除首尾空格、Tab 或 NBSP。因此“预算输出等于原文”不能证明组合后的原文仍一致。
该反例先在旧实现上失败，再修改 `h3_t8/prompt_relay_advanced.py`：

- Global 保留原字符串，只用 `.strip()` 判断是否全空。
- Local 保留事件行的原字符串，只用 `.strip()` 跳过全空行。

事件的换行／管道分隔、时间解析、时间偏置公式、包装器和注意力数学均未改。
普通正式示例的 Plan 全部输出及最终提示词，与修改前备份完全相同。
没有自动润色、截断、归一化、重写节点，也没有修改音频、步数、sigma、接缝或放大器。

Local 是逐行事件格式：行分隔符是结构，不承诺把一段任意多行散文变成单个事件。
Global 内部的 CRLF 与 Unicode 原文，以及 Local 每行的首尾空白，均由组合回归校验。

## 实际覆盖的组合链

CPU 回归在上述正式图的合同下，显式接入既有预算／角色绑定函数，核对：

`原文 → Budget／角色追加 → Relay Plan → 分段投影 → 原生 tokenizer → Bridge → LOW／HIGH Conditioning → 实装 Relay wrapper／attention router`。

正式图本身没有预算节点；测试增加这一步是组合取证，不是改写已验收图。
输入含中文、韩文、CRLF、Tab、emoji、NBSP、扩展汉字和 `<Picture 1>`；
全片不重复对白，局部 0–3 秒只有一次短台词，其后明确闭嘴微笑／举手和环境声。

| 核验对象 | 实际证据 |
| --- | --- |
| 原文与角色 | 保存原始 UTF-8 字节；角色追加一次，最终 Global／Local 空白保留 |
| 最终文字与 token IDs | 使用本机 Core 的真实 MiniMaxH3Tokenizer；最终文字尾 token 与 binding 哈希相符 |
| 素材与时间 | 图片槽、事件全局时间、context／分段投影，以及真实 packed layout 对应 |
| 两段、两阶段 | 每段独立 LOW／HIGH MODEL–Conditioning 绑定；不同尺寸及 Bridge 收据不可混用 |
| Bridge | 合成权重回归另测试 LOW 0.10／HIGH 0.20；本机真实转换权重补验正式 0.10／0.10 |
| Relay 实际生效 | `apply_exp` 实际安装，四组各完成 4 次 wrapper 和 4 次 PyTorch attention 路由；调用后 runtime 清理 |
| 旧合同 | 旧 354 节点顺序／输入输出不变；普通正式提示词输出精确相同；双模型必需五组 CPU 回归通过 |

真实转换权重为 `MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors`，
SHA256：`896829a178e89e23f35799d403971b19b10bbacb2da00454c576770b2f58a9b1`。
测试读取本机现有权重，不下载、不重新转换、不覆盖模型。

## 时间窗与资格边界

既有默认 **preserve_all** 在每个分段保留所有事件文字键，并用时间偏置降低窗外影响；
不是把窗外台词从文本中硬删除。测试证明事件时间／槽位／binding 对应、偏置方向正确、
实际路由调用成立，**不证明模型只在目标窗口发音、逐字对口型或尾段绝不乱说**。
不能把“所有段仍有相同 token IDs”误判为串接丢失事件，也不能把该事实写成硬隔离保证。
如要比较可选窗口文本策略，需要独立同条件成片与试听，不自动改变已验收默认配方。

Qwen embeddings、VAE、H3 projection／forward 使用明确的 CPU 边界替身；
真实 tokenizer、Bridge 计算、Plan、layout、绑定、Relay wrapper 和 CPU SDPA 不使用替身。
这不是完整 H3 diffusion、所有媒体 prefix／第三方组合、GPU 画质或新接缝验收。
没有把文本预算当作全部视觉 embedding 预算，没有新增统一 token／生成帧数上限。

## 重现与回归

从节点项目根目录运行（`python` 使用本机 ComfyUI 环境）：

```powershell
python tools/check_director_d1_cpu.py -q tests/test_h18_prompt_chain.py tests/test_director_d1_integration.py tests/test_prompt_budget_advanced.py tests/test_prompt_relay_token_bytes.py
python tools/check_h18_live_chain_cpu.py --model <本机已转换Bridge.safetensors> --output <新的本地证据目录>
```

后者使用正式 0.10／0.10，生成 `h18-chain-trace.json`，包括正式图／权重 SHA、
原文十六进制、最终文本／token IDs、媒体映射、投影事件、四组绑定／Bridge 收据和调用数。
输出目录须是新目录，失败证据不覆盖。两个入口都强制 CPU 并检查 CUDA 未初始化。

涉及双模型 Relay 的后续变更仍遵守 [接缝回归门禁](DUAL_MODEL_SEAM_FIX_20260913.md)：
只在确认新反例时最小修复；没有新完整 GPU／真人证据时，不扩大既有成片资格。
