# 两套独立底模：内存边界与补验结果

共用底模的两条LoRA分支与加载两套完整底模，不是同一内存需求。
不改变底模，不用小模型替代测试，不降低保护线。

| 记录 | 启动可用RAM | 最低可用RAM | 结果 |
| --- | ---: | ---: | --- |
| dual-distinct-base-unpinned-gpu-v1 | 105.51GiB | 14.75GiB | 旧音频策略下完成 |
| dual-audio-repaired-distinct-base-headroom2-gpu-v1 | 90.58GiB | 3.82GiB | 触发持续低于4GiB保护 |
| dual-repaired-distinct-base-v2 | 101.65GiB | 10.79GiB | 修复后4＋4及独立媒体检查通过 |

来源是各自artifacts目录的resources.jsonl、terminal.json和server-command.json。
三次Core均为488e8f8ab84592670bcc2ff6a1aa20fabafd5160，二采文件SHA均为
9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9。
三次均禁用锁页、reserve-vram=5；两次修复后测试均额外headroom=2。
与最早成功片不是严格单变量对照，不能单凭这张表确定硬件最低内存要求。
这些是整机周期采样，不是精确峰值或单进程分配统计。

失败日志到两套UNET和Sage配置加载，没有开始采样；最低空闲显存仍约11.8GiB。
ConnectionReset是控制器保护终止后的连接错误，自有子进程全部清理。
初始RAM少约14.9GiB是明确差异，但不能据此断言每一模块占用或已证实内存泄漏。
不能把这次失败写成Sage故障，也不能用旧音频策略成功片替代修复后验收。

源码核查使用原生UNETLoader→comfy.sd.load_diffusion_model；当前Core的aimdo
经load_safetensors映射及动态patcher的assign语义加载。发生在上游加载阶段，
内部采样节点的阶段释放尚未执行；仅在内循环结尾追加gc不能解决这个发生时点。
没有修改Core加载器、全局策略、用户启动参数或其他应用。

24秒修复链完成后，观察到可用系统内存回升，才再次补验；没有关闭用户软件、
清系统缓存或降低内存/显存保护。新测试最低空闲显存约4.78GiB，保护范围内完成，
自有测试进程全部清理。独立审计确认两套UNET和各自LoRA绑定、实际不同模型内容身份、
4＋4步、Sage200＋200调用、音频二采续完及72帧完整音视频解码。
见`artifacts/dual-repaired-distinct-base-v2/independent-audit-v1.json`。
这里只验证这一对FL2VA／Ref2VA，不证明任意模型家族、参考组合或其他机器都能运行；
画质、人声和口型仍待集中人审。过去的保护停止保留为失败记录。
