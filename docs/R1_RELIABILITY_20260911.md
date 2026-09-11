# R1 可靠性：当前证据与剩余边界

## Fun Control

固定W4A8头复现了旧metadata未转换导致误选dense的缺口；两个现有加载器已局部修复。
见`R1_FUN_QUANTIZATION_REPRODUCTION_20260911.md`。损坏、冲突、未知格式、缺转换接口
不静默回退。另有真实小型safetensors：旧/新编码分别经当前Core量化Linear加载与CPU前向，
与原生W4A8函数结果逐位相同（`artifacts/fun-quant-native-cpu-v1.xml`，2通过）。
这是小张量运算边界，不是完整Fun模型推理、质量或速度验证；尚未下载完整1.45GB文件。

## Creator

规划器只比较调用者填写的模型/LoRA/参考声明，不读取模型或媒体，不执行缓存复用。
相同文件名换内容、声明没有更新时，它无法证明内容一致。因此匹配行改为
`declaration_match_unverified`；`hit_count`为兼容保留，但含义明确为声明匹配数。
新报告同时给出`declaration_match_count`、`verified_reuse_count=0`和
`execution_reuse_authorized=false`。`source_packet_hash`不是媒体文件hash。

已有镜头范围失效规则保留：模型/LoRA强度及顺序改变影响相应契约；参考顺序只影响对应镜头。
排队后实际文件变化必须由未来的执行消费者再次验证，本轮不悄悄新增消费者。
accepted历史产物不删除、不移动。复现前5失败，修复后与Fun合计36聚焦测试通过，
见`artifacts/r1-fun-creator-after-v2.xml`。旧输出端口和输入字段不改。

## Core 增量

对真实源码树和import来源逐项核对，不覆盖当前ComfyUI：

| Core SHA | 针对性 CPU 结果 | 限制 |
| --- | --- | --- |
| 488e8f8ab84592670bcc2ff6a1aa20fabafd5160 | 99通过 | 当前依赖环境 |
| fbed745c8d7d62573b099cd61fe51cb64b9b807e | 97通过、2跳过 | 归档缺两个历史多关键帧用例所需git对象 |
| 563b98eefbe643a4cd510ee7f0b43e79880d5a3f | 44通过、55跳过 | 老接口能力不具备，不能把跳过算通过 |

覆盖FinalLayer部分schedule拒绝、attention所有权、Relay/Core与BlockCache边界；不吞TypeError，
不新增Core全局patch。六个源码文件前后hash与声明SHA核对，CUDA未初始化。
这是接口增量回归，不是完整GPU组合、旧依赖栈或所有硬件验证。
证据在`artifacts/core-incremental-{current,prior-fbed,old563}-v1/`。

## 渐进工作流 UI/API

两份渐进图已在隔离前端实际打开、将low_evaluations从6改7、保存、重载并导出API。
文件复核确认T2VA16节点25边、I2VA17节点26边，除测试参数外执行值和连线保持一致；
原生界面的中文task标签在API中仍为规范代码。两份实际导出又通过当前Core CPU验证，
327项目节点、未初始化CUDA、没有排队生成。独立验证器的未用builtin nodes_replacements
因无PromptServer实例发出导入警告，不据此声称完整builtin启动测试。

原生“粘贴并连接”也已在两份图完成：新增节点保留六路输入、全部参数，原节点和边未改变。
自动化工具直接CtrlShiftV报空虚拟剪贴板，因此在隔离设置中为同一原生命令临时绑定
AltShiftP，测试后恢复；不是注入剪贴板数据，也不代表物理键盘快捷键已测试。
测试只修改临时副本，交付图仍默认6+2，用户前端未动，两个自有CPU服务已清理。

证据：`artifacts/r1-ui-save-export-audit-v3/`、`artifacts/r1-ui-connected-copy-audit-v1/`；
复制审计器补测发现原有输出持有的旧连线和输出元数据尚未全查；新增两项负例先失败，
补齐检查后14项通过。用更严格审计器再次核对原来两份真实保存文件也通过，见
`artifacts/r1-ui-connected-copy-audit-v2/report.json`，没有重做前端或生成。
此前Chrome拦截未绕过，本次使用另开的隔离内置标签完成。

清空值也已单独验证：两份图原提示词经实际输入框清空、CtrlS保存、重载，
再用原生菜单导出API；两条请求均保留空字符串，未带回原文。
所有其他执行参数和连线逐项核对不变，实际导出通过Core CPU校验，未生成。
证据在`artifacts/dual-r1-ui-native-audit-v1/`；这里同时包含两份双模型图改参检查，
不是用它们代替渐进图。8项审计器正负例通过；隔离服务已清理。
