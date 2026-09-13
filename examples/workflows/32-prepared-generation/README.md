# Tao / LTX 已准备输入生成（实验候选）

这里的两个工作流连接真正的生成与解码入口，不是固定旧片播放器。
当前已完成节点注册、Core API、序列化、两条原生 CPU 输入检查，以及真实 Core
执行器的旧样片迁移→状态保存→原文件预览返回。该贯通没有重新生成或解码，
不等于新封装 worker 的 GPU 实测。原生浏览器 Export(API) 已核对执行值/连线；
实际解包导入与三项准备 CLI 检查也有通过证据。指定LTX与Tao新对白恢复片已通过人审，不重复生成或审片；
Tao原生成收尾触发RAM保护仍是失败任务，不能据媒体接受宣称完整生成可靠性已通过。
当前收尾调整仅有CPU作用域/循环引用释放、原采样代码AST一致及异常门禁证据，尚无新完整模型GPU复测。
不要当作已验收正式示例；当前测试环境是 Windows/Python3.12。

- `2026-09-13_H3_Prepared_tao5s_EXP.json`：匹配 Base10 教师及文本，Tao 原生单请求5秒。
- `2026-09-13_H3_Prepared_ltx_refine_EXP.json`：已经转换好的 LTX AV 潜空间与文本缓存，原生3次精修。

## 准备步骤

使用同一 Comfy Python，运行项目的 `tools/prepare_generation_bundle.py --help`。
提供明确的生成输入 request.json 与解码输入 request.json，不是 Comfy 工作流 JSON。
工具兼容本项目既有已通过原型的这两份 request，会去掉旧 GPU ID、旧脚本身份和
旧 latent 路径，重新构建模型/输入/运行目录的实际文件身份。它不会生成教师、
编码新提示词、做 H3→LTX 转换、下载、安装或运行模型。

Tao 命令格式：

```text
python tools/prepare_generation_bundle.py --kind tao5s --generation-request <生成输入.json> --decode-request <解码输入.json> --audio-seed <匹配教师的种子> --output <新清单.json>
```

LTX 命令格式：

```text
python tools/prepare_generation_bundle.py --kind ltx_refine --generation-request <生成输入.json> --decode-request <解码输入.json> --prompt "与已编码缓存一致的提示词" --frames 73 --width 2048 --height 1024 --output <新清单.json>
```

底模分片和文件会完整读取计算 SHA256，大模型这一步较慢，但没有 GPU 采样。
同一路径替换文件、同大小修改、增加/删除目录成员、改变源代码修订或运行目录
都会影响身份。工具不覆盖现有清单，请提供新文件名。

再使用 `tools/qualify_prepared_inputs_cpu.py --bundle <新清单.json> --output <新报告目录>`
做原生 CPU 输入核对。LTX 检查 AV shape/dtype/finite、时间几何及实际文本缓存；
Tao 检查文本提示词、教师种子、3/6/9三份里程碑逐值一致。不加载 DiT/VAE，
不代表新模型推理成功。输入源代码按明确 Git 修订检查，要求跟踪源码未修改；
不会删除或重置用户目录中无关的未跟踪项目。

## 导入工作流后

1. 第一个节点填刚准备的 JSON **绝对路径**。模板故意不写开发机的私有路径。
2. 第二个节点的 `serial_lease_path` 填与其他研究入口相同的 GPU 锁文件绝对路径。
   此入口当前仅 Windows 单 GPU；不得换锁文件规避已有任务。
3. `noise_seed` 是视频噪声种子。Tao 的教师音频种子来自准备清单，不能只改它
   就冒充另一份教师。LTX 使用相同种子驱动原生视频→音频加噪顺序。
4. 改输入、提示词、seed、模型或后端代码后换 `chain_id`。仅使用字母数字、下划线、
   连字符。`resume_existing=true` 只复用严格匹配的已完成阶段；生成成功而解码失败
   时只补解码。`false` 不是覆盖旧结果，仍须换新链 ID。
5. 生成进程与全部子进程退出后才加载 VAE。缺少内存/显存余量会停止，不降低保护线。
6. 输出节点已经保存 MP4，并直接预览该文件。无需再接 SaveVideo 重编码原音轨。
   通过 `saved_path` 找成片，`report_json` 查看新生成/恢复/迁移及失败记录。

## 输入与质量边界

Tao 目前是864×480原生单请求5秒，复用已准备 Base10 音频，不是多请求长视频；
124原生渲染帧按上游时序交付120帧。要换提示词，必须重新准备匹配的教师与文本。

LTX 输入必须是 normalized LTX AV、没有额外 reference prefix；不能把普通 H3 latent
改名后直接接入。CFR24、8n+1帧、最多8秒且受 token 数限制；这里只对既有73帧
2048×1024样片有 GPU 证据，其他合法几何没有因此被认证。固定3次原生 Euler、
LoRA0.8，执行为反量化 INT8 底模→BF16，不是原生BF16权重质量/速度对照。
最终保留原AAC包，LTX新生成音频不用于交付。

已有片子的复用必须经过原始 request/report/文件身份核对，不能手工填写一个
fingerprint 就当作已证明来源。目前模板没有嵌入旧片缓存。显式迁移工具用法：

```text
python tools/migrate_prepared_checkpoint.py --bundle <准备清单.json> --generation-directory <原生成回执目录> --decode-directory <原解码回执目录> --seed 8301 --output <新迁移清单.json>
```

LTX 还必须提供 `--media-directory <CPU媒体恢复回执目录>`，不接初次损坏的MP4。
这个迁移仅适用于已有完整证据的原测试参数，不是任意旧视频导入器。模型、输入、
源代码、Python/依赖身份发生改变时，旧指纹失效，必须重新核对和显式迁移。
不要手改 fingerprint、绕过校验，或将已通过样片重新采样冒充迁移。
完整候选边界见 `docs/PREPARED_GENERATION_INTEGRATION_EXP.md`。
外部源码固定修订、测试依赖和不随包分发的内容见
`docs/PREPARED_SOURCES_AND_ENVIRONMENT.md`；不是自动安装器或任意环境兼容保证。
