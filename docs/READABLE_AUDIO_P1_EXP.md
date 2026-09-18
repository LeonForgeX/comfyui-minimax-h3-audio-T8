# 来源诊断、音频说明、开头静音淡入（v1.85.0）

四个新增独立节点，旧节点／图无需改接线。
[正式示例](../examples/workflows/38-diagnostics-preview/README.md)将可选音频处理关闭，
不是删除独立处理节点自身的enabled默认；无爆音不必开启。下方本地开发记录为历史。

`MiniMaxH3NodeSourceDiagnosticT8` 输出实际加载模块路径、磁盘源码SHA、当前可观察的来源／重复provider。磁盘版本号和文件SHA不证明内存字节码相同；未知项明确写未知。不删目录、不自动重启、不阻止采样。

`MiniMaxH3AudioSourceExplanationT8` 接已有Conditioning文本报告或JSON，说明native生成、lock_source录音驱动、原音轨mux、双采策略／缓存证据的区别。报告不能独立证明交付音轨来源或音色克隆成功；不改声音、提示词、模型或缓存。两个节点提供可滚动换行的只读面板，不是可编辑参数框。

`MiniMaxH3AudioOpeningMuteFadeT8` 只在输出AUDIO端使用。前N帧对应声音置零，随后半余弦淡入；FPS必须与视频一致，支持30000/1001。N=0可只淡入；N=0且fade=0恒等；disabled返回原对象。保留真实采样率、通道、样本数和时长，不做整片降噪、响度归一化或重采样。

`MiniMaxH3VideoOpeningMuteFadeT8` 接末端VIDEO、再接原SaveVideo。tensor VIDEO帧数据直通；文件VIDEO按实际显示帧PTS确定静音边界，不用平均FPS处理VFR。视频包字节／时间戳保持不变，仅重编码音频。原有trim/crop/BytesIO/非MP4视图由Core流式导出，再处理该活动视图，不取全片RGB batch；Core视图导出本身可能转码，不能声称保持原完整文件字节。

AAC是有损编码，有解码块padding；校验有效音频起止／时长／PTS和解码后内容对齐，不把容器SHA或padding样本数当作恒等。未知字幕／数据轨不静默删除，真实错误正常传出。过大的N／fade会削掉首字，建议先N=1、10ms听取效果，无爆音时无需开启。

## 已测与限制

59范围CPU通过：开关／tensor恒等、CFR/VFR、正负音频offset、真实活动视图、非周期chirp内容没有隐藏位移、视频包不变、原文件不覆盖、Windows Job取消及下一请求恢复。合并范围回归另覆盖旧采样／接缝路径，均CUDAfalse；最新数量与冻结源读五项checklist对应terminal，不冒称全仓通过。

Windows codec子进程执行前纳入Job，取消后不遗留本任务子进程；没有系统进程扫描／杀其他任务。Linux fallback的stderr改文件尾读，避免无人读取PIPE导致死锁，但尚未做Linux实机资格。

JS只读面板的回调链、纯文本显示和监听器清理通过actual-source fake DOM测试。
自有隔离CPU Core8213／隐藏页实际展示当前来源面板，TAE定向取消后下一普通来源诊断
原生请求成功且队列为空，见ui-probe-v1/live-ui-audit.json。回放不是新GPU采样；
自有Core／页已退出，不动用户前端。仅本地新源码，用户自行重启载入，不自动重启服务。
