# Fun Control 旧量化元数据：局部兼容修复及小型加载验证

当前覆盖下方复现阶段：两个现有加载器已经保留并规范化旧量化 metadata，
使用 Core 原生转换后再检测，不新增 loader、不修改 Core。损坏/未知/冲突声明、
缺转换接口、转换丢标记均明确拒绝；原权重对象保持不变。
与 Creator 身份边界合计36项聚焦 CPU 测试通过，证据
`artifacts/r1-fun-creator-after-v2.xml`。另有真实小型safetensors经当前Core旧/新
W4A8编码加载与CPU前向的2项通过，结果与原生量化运算逐位一致，见
`artifacts/fun-quant-native-cpu-v1.xml`。这不是完整1.45GB模型的推理验证；
没有因此下载完整模型，也不宣传画质、速度、显存结论。以下失败是修复前证据。

固定来源：berryber09/MiniMax-H3-Fun-Controlnet-Union-w4a8，revision
`4709bcbb03fa9dc34e342bf46d19d9039463244f`，文件
`minimax_h3_fun_controlnet_union_pruned_w4a8.safetensors`。

`tools/probe_fun_control_remote_header.py`以两次严格HTTP206范围读取取得真实文件头：
全文件1,453,695,488字节，只读取28,416字节，没有下载张量数据。
证据`artifacts/fun-control-w4a8-header-v1.json`；头SHA256
`0cabc25d5bc40881e3337edf989e34452efe67cd3c20d7295120f16ddeeede3e`。
124张量，`_quantization_metadata`中20层`asym_w4a8_int8`，group_size16、
convrot=true、convrot_groupsize256；没有原生`.comfy_quant`张量。
QKV I8形状21504×2688（输入维打包减半），输入投影F32形状5376×196。

当前Core488e8f8ab84592670bcc2ff6a1aa20fabafd5160真实CPU函数复现：
未转换时`detect_layer_quantization`返回None；先调用原生`convert_old_quants`
后返回mixed_ops=true，20层元数据逐条一致，原张量对象不变。
`artifacts/fun-control-w4a8-selection-gap-v1.json`记录函数源码SHA；
张量仅用meta设备重建形状，CUDA未初始化，不是W4A8实际前向或画质验证。
首次工具因未列F8_E4M3 dtype失败；补齐真实头中该dtype后成功，不隐藏失败。

现有两个加载器的直接回归：`tests/test_fun_control_quant_selection.py`。
`artifacts/fun-quant-selection-before-v1.xml`：4失败、4通过。
两条旧元数据路径错误选择dense；两条损坏元数据路径未在设备/运算选择前拒绝。
原生`.comfy_quant`和普通精度四条通过。测试拦在运算选择处，不分配整模型。

上述计划已经实施到两个现有加载器：保留metadata，在检测前做局部原生转换，
校验损坏/冲突声明并保持新格式/普通精度语义。无转换接口、同层冲突、未知格式
和原对象不变等负例均有测试；不新增节点。完整候选发布仍需主线最后的回归和打包，
不把文件头检查或小型算子前向等同整模型画质、提速或省显存验收。
