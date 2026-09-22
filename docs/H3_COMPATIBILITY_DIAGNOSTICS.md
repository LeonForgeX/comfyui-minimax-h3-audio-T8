# H3 权重诊断边界

通用 LoRA 加载器继续委托 Core 的适配器解析，不因陌生名字、哈希、用户组合而禁止运行。
新增报告把不同层次分开：

- `input_tensor_count`、`format_hint`：读到的原始键与格式线索，不是支持证明。
- `structural_conversion`：实际布局转换；QKV 合并使原始/转换后数量不能一对一比较。
- `mapping_diagnostics.converted_unmapped_keys`：Core 解析器没有报告消费的转换后键。
- `applied_patch_count`：ModelPatcher 注册数量，不等于 GPU 真正执行，更不代表图像/声音质量。
- `payload_diagnostics.special_runtime_requirements`：HyperFlow 双时间与固定网格、PDD 动态音画头、VSA gate attention 等额外运行条件。只加载普通骨干权重不能证明这些算法完整运行。
- 强度零也会注册 patch，但报告明确没有有效权重变化证明。

完整报告随节点 JSON 输出，并附在 MODEL 的 `t8_h3_lora_compat_report`，供后续诊断读取。
本诊断功能本身不执行 HyperFlow；另有独立的 HyperFlow 专用加载/采样实验路线。
仅看见 metadata 仍不等于已通过该路线的真实推理与成片验收。

`inspect_h3_weight_file` 仅读最大 16 MiB safetensors header，不读 tensor、不下载、不全文件哈希。
缺文件、GGUF 或无法解析的头返回 unknown，由实际加载器决定格式支持。header 通过也不证明
数据完整性、量化内核、设备支持或 LoRA 的实际数值效果。

只有 `merged_loras`、`merged_adapters`、`t8_merged_adapters`、`merge_recipe` 明确声明加速合并时，
才报告 `merged_acceleration.declared`。这仍是文件作者声明（`verified=false`），不是可信来源认证。
文件名含 turbo、量化名称或标题，不足以判定已合并。额外 LoRA 的潜在重复叠加只能给可解释风险提示，
不可静默删除用户显式选择，也不能恢复未知组合硬禁令。

ClipProj 的 quantization 字段只描述 CLIP 编码器；不能据此宣布 DiT 支持 NVFP4、RotNVFP4 或任何底模格式。
