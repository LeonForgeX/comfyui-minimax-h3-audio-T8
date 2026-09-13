# Prepared Tao / LTX：来源、环境与分发边界

这是未发布候选的复现说明，不是许可授权或任意设备兼容承诺。新入口不会自动
下载安装模型、源码或Python依赖，也不会修改用户主Comfy环境。

## 本地实测来源

| 外部组件 | 本次固定源码/模型修订 | 用途 |
| --- | --- | --- |
| TaoLiveAIGC/TaoMate-H3（GitHub） | ccc1a70adbf7f552a84a0cd7eeac0a6f3d461cad | 原生teacher/clean KV/streaming算法 |
| Lightricks/LTX-2（GitHub） | d151147788a9284cca791edc6ce898007e727fe6 | 原生AV类型、packing、3次Euler及解码 |
| NVlabs/Sana（GitHub） | 144085566a866f9784f3798d4c8d1603f3adbccf | models/minimax_h3/Sol-H3-Spark下转换实现 |
| MiniMaxAI/MiniMax-H3（Hugging Face） | 42ed227ee7df40d41602854ae760620d6eb651fe | 官方底模 |
| TaoLiveAIGC/TaoMate-H3（Hugging Face） | cfa5150cc7b41974a6b657ec6479adc5cbabc611 | Tao适配器 |
| Efficient-Large-Model/H3-to-LTX-Latent-Adapter（Hugging Face） | 1792c42689a0f22de880eaf57a187c6a373a636d | 194M潜空间适配器 |

上述Git源码由用户本机的显式目录提供，不随节点包复制整个仓库。节点包中的
`prepared_backend`为本项目的单卡调度、身份校验、权重搬运和媒体封装代码；
它调用指定上游原生实现，不用普通LoRA＋少量步数冒充完整Tao算法。
旧GPU原型中的8个Tao与5个LTX数学/输入helper已在迁移时逐文件哈希比较。
四个新封装worker作了导入、参数、输入预检等改动，未重新跑完整GPU，不能据
helper相同就宣称新封装在所有配置下已经实测。

当前本地Tao源码的LICENSE标题为“MiniMax H3 COMMUNITY LICENSE AGREEMENT”；
LTX源码的LICENSE.md标题为“LTX-2.x Community License Agreement”。上游代码、
权重和所用模型条款分别保留，不因本节点仓库的许可证而被重新授权。
当前H3→LTX适配器模型卡没有单独明确新的权重许可；不要自行补写为GPL或镜像
上传权重。Sana使用稀疏源码目录，本说明不从缺失的根LICENSE推断无条件授权。
分发前应核对相应固定版本的完整原始条款；本候选不分发任何上述外部权重、
隔离Python运行目录或外部完整源码。

## 实测环境和依赖

本次为Windows、单张RTX4060Ti16GB、128GB RAM、Python3.12、PyTorch2.10.0+cu130。
这不是最低系统配置或其他显卡/操作系统认证。Tao全底模CPU加载会使用大量内存；
生成和VAE分进程串行，继续保留资源保护线，不以降低保护换取“跑通”。

共同使用现有Comfy Core、Torch/torchaudio、safetensors、numpy、psutil、
nvidia-ml-py及FFmpeg/ffprobe；具体上游导入另依赖其原环境。LTX本次在隔离路径
使用Transformers5.14.1、OpenImageIO3.1.17.0等已准备运行文件，未升级主环境。
环境指纹包含Python/可执行文件、系统，以及相关已安装分发包的版本/位置；
LTX隔离运行目录还单独按全文件清单核对。这不是每个主环境依赖文件都已做
对抗性完整性证明，源码Git pin也只覆盖跟踪源码，不包含无关未跟踪项目。

项目旧入口的`requires-python>=3.10`不能解释成新Prepared流程已在3.10实测。
Prepared实际资格仅为上述Python3.12组合；缺少3.11 `add_note` 的异常兼容测试
只是模拟该API缺失，不构成完整3.10执行认证。

## 包内准备工具

仅以下三个准备CLI随本候选交付，不包含测试控制器、研究服务器、私有请求、
本地交接或样片：

- `tools/prepare_generation_bundle.py`：构造全身份输入清单。
- `tools/qualify_prepared_inputs_cpu.py`：原生CPU输入/提示词/教师一致性预检。
- `tools/migrate_prepared_checkpoint.py`：从原始回执显式迁移已证明来源的样片缓存。

三者的`--help`不加载模型。完整命令和工作流接线见
`examples/workflows/32-prepared-generation/README.md`。准备清单中的路径必须属于
本机实际输入；复制节点目录、改Python/依赖/代码后旧指纹会失效，重新构造/
迁移清单，不手改fingerprint。输入准备、缓存复用和新GPU生成在报告中分别记录。
