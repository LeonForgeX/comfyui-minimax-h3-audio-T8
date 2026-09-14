# TST 来源核对与实现边界（开发记录）

这不是已交付的 TST 功能说明。本文记录实现前发现的公式／注入位置差异，防止把三个不同实现统称为同一复现。

## 已核对的第一手来源

- [论文 v1，第2、3节](https://arxiv.org/html/2609.08505v1)：用行熵与谱熵之差作为有正负号的失衡指标；指数因子乘在 Query 上，随后重新计算注意力。层／步余弦调度使深层、早期干预较强。论文的主体评估模型为 Wan2.2，不是 H3。
- [作者代码 core.py，固定 ed1ed527](https://github.com/lytang63/temporal-state-transport/blob/ed1ed527935ab071323fe77b1c20e8f9ed0d98a4/tst/core.py)：实际返回 `1 + tau_effective * temporal_alpha * H_vn * (1-H_vn)`；虽计算行熵，但没有将其用于返回值。正 tau 下的这个因子不小于1。
- [同一固定版本的 Wan 注入](https://github.com/lytang63/temporal-state-transport/blob/ed1ed527935ab071323fe77b1c20e8f9ed0d98a4/tst/models/wan.py)：实际在 attention、合并图像支路以及输出投影之后乘上上述因子，不是论文中的 Query 温度调节。每次 forward 的 pre-hook 重置全局调用计数，但 core 又用该计数推断扩散步；不能直接照搬为双阶段全局时钟。
- [SelfLift H3 适配，固定19ec5405](https://github.com/facok/comfyui-SelfLift/blob/19ec540505dcbc261aaecf450d83c7536be2826d/h3_tst.py)：以有符号失衡调节视频 Query，并占用 attention override 和 diffusion wrapper。其 H3 实现不能直接当作作者代码的逐位复现，也不能直接覆盖我们已有的 Relay／EAV／后端 owner。

## 对本项目实施的约束

1. 公开参数／报告必须指明具体公式和 H3 时间算子的构造方式，不使用含混的“上游一致”标签。
2. 不能把作者当前代码的输出增益直接换成指数 Query 修正，再声称只是兼容性改动；也不能把前者写成论文第5、6式的复现。
3. LOW/HIGH 和断点恢复使用真实采样时间表与层号，不使用模块级全局调用次数猜测扩散进度。HIGH 不重新从第0步计时。
4. 通过真实 packed layout 选目标视频，不把末尾所有 token 都当视频；文本、音频、参考和上下文范围单独核验。视频 Query 改动仍可能通过联合网络影响声音，不能承诺音轨锁定。
5. 与 Relay/EAV 组合必须明确操作顺序、浮点偏置和已有后端的保留，禁止覆盖已安装的 attention override；Sol 回退仍应如实报告。
6. 禁用状态须与原生输出逐位一致；启用状态须有独立数学 oracle、真实调用计数、取消／恢复和资源测试。张量分数改善不是视频质量验收。
7. 在选定并测试 H3 算子之前，不复制未声明许可证的 SelfLift 源码、不开放节点或宣传1–2%开销。后续必须完成真实 H3 控制变量成片及动态人审。

已完成来源差异核验、独立数学核、独立MODEL节点及渐进组合的小型CPU/CUDA验证；正式交付和完整模型质量验收仍未完成。
原已验收长视频路径没有因本次研究被修改。

## 本地接入点与组合约定

当前Core主DiT循环在每个block调用前写入`transformer_options['block_index']`，
原生attention传递同一options；不要以token-refiner调用数推测层号。
空间／时间范围来自实际`minimax_h3_layout`的目标video段与当前阶段几何，不能仅凭末尾长度猜测。

选定独立实验profile：`signed_query_mean_frame_v1`。
每个head对目标视频的post-RoPE Q/K逐帧做空间均值，再建立F×F softmax算子，
按论文的归一化行熵减谱熵计算有符号T，使用`exp(tau_eff*T)`修正目标视频Query。
这是显式H3帧均值代理：不是完整空间attention聚合后的精确时间算子，
也不是作者仓库当前的输出增益代码。对照与报告须一直保留这个区别。
逐帧均值不混时间，原K/V、非目标Query不改；不能承诺联合网络音频不受间接影响。

组合顺序固定为：实际完整Q/K与目标布局 → TST Query修正 → 原Relay时间偏置／分块
→ 原Sage／KJ／Sol或明确回退 → EAV对目标输出施加增益。
现有`route_prompt_relay_attention`在分块后才调用delegate，因而不能把需要完整视频范围的
TST塞到delegate里。现有`route_eav_prompt_relay_attention`则在Relay返回后计算FETA，
实现时需明确其统计使用修正后的Q，不能把这称为与无TST的EAV数值等价。
TST应使用显式内部组合入口，而不是覆盖已安装owner或放宽未知补丁保护。

调度由真实完整sigma位置和Core层号确定：HIGH使用完整表中的位置，恢复从实际HIGH位置继续。
具体余弦端点约定需要写成可测试函数，不能用阶段局部`sample_sigmas`重新归零。
禁用须不安装额外入口且不分配修正张量；report_only返回原Query对象。
测试需包含恒等／均匀／非对称算子、不同batch/head、dtype／NaN／尺寸拒绝、
非目标Query及K/V不变、LOW/HIGH／恢复调度、实际后端与Relay偏置保留、取消清理。
代理F×F、输入转换和Query副本的工作量需显式预算；预算检查不是整卡显存保证。

上述接口已接入渐进运行路径及开发版独立MODEL节点；节点配置/实际执行/完整模型质量三种状态分开记录。

## 独立数学核阶段结果

为保持当时正在执行的整链回归源码不变，数值核最初放在本地研究目录，未注册节点。
43项CPU测试通过，CUDA未初始化；第一次数学测试调用曾因pytest越界收集父包而缺少Core导入，
修正独立rootdir/confcutdir后运行通过，没有为通过测试修改生产依赖或数学预期。

- 归一化行熵、谱熵及有符号差值用独立NumPy/SVD对照；恒等、均匀、置换和非对称算子通过。
- 多batch/head与非连续Q/K布局逐head对照；只改指定目标视频范围，原Q/K和其他Query逐位不变。
- 正负失衡分别使温度因子大于／小于1；未保存每层F×F算子到诊断结果，避免长期累积大张量。
- 余弦端点明确：layer_fraction=(index+1)/count，step_fraction=index/(count-1)；
  最深层／首步强度最大，末步为0。8步的HIGH使用全表index4..7，不重新从0开始。
- disabled不访问输入tensor；report_only、tau0、末步返回原Query对象。
- 半精度溢出、非有限输入、错误布局／时间索引和超工作预算会拒绝，失败不污染输入。

最初43项仅验证公式和边界。该整链回归随后171项通过，数学核迁至内部`tst_math.py`，
加入`tst_runtime.py`及以下实际组合测试；各批测试有重合，不应简单相加成总覆盖数。

## 内部渐进运行接入（CPU通过，未公开交付）

`sample_progressive_h3`／`sample_progressive_continuation`及`NativeProgressiveJob`接受内部实验配置：
`tst_mode=disabled|report_only|apply_exp`、`tst_tau`、`tst_max_workspace_mib`。
默认disabled。下述独立MODEL节点另行追加；没有把现有用户JSON静默改为启用TST。

- 在现有APPLY_MODEL观察入口内，以实际原生视频sigma开启当前阶段作用域；CFG必须为1。
- 每层取Core实际block_index，按实际packed layout定位唯一目标video段。
  token-refiner不计为主层；每次前向必须完整处理每个主层一次，缺层、重复层或布局漂移拒绝。
- Relay/EAV组合保留原owner和后端。TST先处理完整Q，Relay分块后不重复处理；
  EAV对同一修正后的Q计算统计。独立dense SDPA＋Relay偏置oracle验证了这个顺序。
- 原Q/K/V、非目标Query、输入MODEL选项不原地修改；关闭与report_only在小型原生H3中逐位一致。
- LOW/HIGH的配置、完整表和缓存身份绑定；LOW缓存恢复报告明确为历史执行，不虚构本次LOW调用。
  HIGH恢复仍使用完整表index4..7。取消／异常清理当前作用域，失败runtime不能重新使用。
- 诊断只保存每个前向的标量统计和完整配置，不保存每层F×F算子或Q/K/V。
  工作预算是显式tensor估计，可能在较大画幅下拒绝；不是显存峰值或1–2%开销保证。

初次迁移的数学＋owner测试74项通过；加入实际路由oracle与原生组合后，8个测试文件合计181项通过。
覆盖真实小型H3/Euler的两块网络、两阶段时钟、Relay/EAV、已安装KJ省显存定义和Sol选择器，
以及续段中每路两个真实Core LoRAAdapter的实际权重核对。LOW缓存后HIGH重跑与未中断结果逐位一致。
编码器／学习型放大器仍为明确替身，Sage/Sol的CPU回退不算CUDA内核或加速证明。

随后15项TST/缓存/任务身份测试与90项旧双采接缝、上下文、恢复及工作流检查合计105项通过。
这覆盖旧路径的CPU防回归，不替代新TST数值路径的GPU成片验收。

仍需：正式UI/API工作流、完整模型/GPU和资源检查、控制变量成片、
0.4MP两段8秒动态人审、UI/API及发行包验收。不能把上述CPU结果写成画面更稳定或全部支持。

## 独立MODEL节点（开发版，未部署）

`MiniMax H3 TST Temporal Query / 时序Query修正 (EXP/T8)`追加在已有节点之后。
它不是另一个采样器：输入MODEL和完整SIGMAS，输出配置后的MODEL及`configuration_json`。
配置报告只写`configured_not_executed`；普通采样的真实完成/中断诊断在控制台`[T8 TST]`中记录，
渐进采样则在自己的report_json中记录每阶段实际执行。

接线规则：

- 普通原生Euler：模型/原生采样设置 → 本路LoRA链与一个后端 → 可选Relay/EAV → TST → 原生采样器。
  保留Relay配对positive，不把它换回未绑定的普通提示词。CFG必须为1。
- 渐进双采：LOW/HIGH可各接一个TST节点，再进入渐进采样器的model/model_hires。
  两个TST节点都接同一张完整SIGMAS；HIGH不会因只执行后4步而重新从第0步计时。
  未另接model_hires时沿用LOW的TST；另接独立HIGH时只启用明确配置的那一路。
- 渐进EAV使用渐进节点的内置eav_mode。不能把外部EAV包好的MODEL直接传入渐进节点，
  也不能同时用TST MODEL节点和内部tst_mode关键字设置两套TST。
- `mode=disabled`原样返回输入MODEL，连额外wrapper也不安装；`report_only`测量但不修改Query。
  `tau=0`仍计算诊断，不等同disabled。较大画幅可能超过默认256MiB的显式tensor预算，会明确拒绝。
- 普通采样只接受原生Euler和完整表内的连续区间；Heun等多次求值算法需要另外的时钟契约，
  当前明确拒绝。未知wrapper、STG跳层、重复TST和配置后更换后端不被当作兼容。
- 普通采样每次建立新作用域，取消释放；后一次执行不会沿用前一次的计数或Q/K/V。
  渐进检查点保存实际阶段证据，改变TST配置会改变任务/缓存身份。

独立节点、原生普通采样、分阶段配置、Relay/EAV、LoRA、缓存以及旧路径合计293项CPU检查通过。
其中普通采样器＋TST已经实际执行小型H3；并非只检查节点字段存在。

随后在独占GPU租约下完成两批真实CUDA检查：内部接口12项、MODEL节点进入渐进采样12项。
两批均覆盖Core Sage、KJ省显存/FFN、Sol × Relay开/关 × EAV开/关，
每路两项真正Core LoRAAdapter运算、4+4、两块网络、原生视频/音频遮罩与TST完整时钟。
Core/KJ的实际CUDA调用检查通过；Sol无Relay时执行稀疏内核，有Relay时因偏置/Query形状明确回退。
这些不是所有后端组合均有加速，也不是完整权重或新长视频成片质量验收。
GPU模型是随机BF16小型H3，学习型放大器仍为明确插值替身，没有实际VAE/视频输出。
