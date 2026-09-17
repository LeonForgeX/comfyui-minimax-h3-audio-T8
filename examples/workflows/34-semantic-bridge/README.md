# Semantic Bridge / BUNNY 正式发布工作流（v1.84.0）

历史五份图已通过原生前端保存、刷新重开与 API 导出核对，并完成相应短片或两段8秒的机器验证。
2026-09-17接续版按最新Core schema重建，并使用已发布的T8无损封装路径；最新Core/API结果见Semantic Bridge完成验证，不把历史媒体/UI记录冒称新源码GPU验收。
**用户已验收修复后B：两段8秒独立双采640×320→896×448，4+放大+4，两阶段Bridge0.10均开启。**
对应成片SHA256：`6da418003510e040d2b3ccc7d9961dcb8d6b652395c0b50425f326159348ce9c`。
验收仅绑定这个配方；其他四份仍为EXP用法示例，未评项不写通过。旧工作流没有迁移。
旧448×224失败片与关Bridge对照不进入推荐；H16／Meridian不在本版。

| 文件 | 入口 |
| --- | --- |
| 2026-09-17_H3_SemanticBridge_Internal_EXP.json | 原版模型，直接接普通条件节点的可选插口 |
| 2026-09-17_H3_BUNNY_External_EXP.json | BUNNY，原生条件→Apply→Guider |
| 2026-09-17_H3_SemanticBridge_Relay_EXP.json | 原版模型，Relay内置插口；base MODEL→Relay→LoRA |
| 2026-09-17_H3_SemanticBridge_SingleLoop_8s_EXP.json | 单模型两段共8秒；每段重新编码后应用一次 |
| [2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json](2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json) | 正式推荐：640×320→896×448，两模型4+4；一采原版、二采BUNNY，各自独立条件 |

模型下载：[t8star/Semantic-Bridge-Comfy](https://huggingface.co/t8star/Semantic-Bridge-Comfy)。
本版示例使用 `ComfyUI/models/semantic_bridge/t8_compat/` 中两份 `*_T8_Compat.safetensors`；
也可以在节点模型列表中重新选择自己的路径。两份原始权重直接可用，不必转换才能运行。
无损 T8 Compat 封装工具在节点目录 `tools/convert_semantic_bridge.py`；它不训练或降低模型精度。

默认强度0.10、per_token、all_tokens。先对照再调整；不要把两个桥串在同一条件上。
关闭对应配置或设强度0即旁路；双采空槽继承公共设置，单独关闭一采/二采需连接禁用配置。
修改模型、强度、提示词或采样条件后换新chain_id。长片先检查两段边界到片尾的画面与声音。
参考声音、歌唱和Hybrid仍属实验范围；原音保留不等于生成声音无退化。

来源与完整参数、接线及限制见 [节点说明](../../../docs/SEMANTIC_BRIDGE_EXP.md)。
工作流附指定样片的内容身份及验收范围，不附私有参考图、样片、模型或本地路径。
Python更新后需重启ComfyUI并刷新；磁盘同步不等于旧运行进程已加载。
