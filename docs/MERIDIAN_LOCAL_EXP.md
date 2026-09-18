# Meridian 独立运镜入口（v1.85.0 EXP）

## 正式发布状态（2026-09-18）

新平行横移成片SHA `5a6c0add1c4f002d4f3afbd4f41423f49516ec015153aaff9dfcdf237613bb13`
已与最终反馈精确绑定，用户确认正常并要求发布；旧无明显横移片不推荐。
此前P2和P4冻结／源运动门禁已完成指定范围确认，四节点／空间时间编辑器保存为
[正式工作流](../examples/workflows/37-meridian/README.md)。
新图采用实测横移0.12；节点默认0.08和已保存canonical plan不迁移。
下方未发布／待审描述均为历史。仍有训练ROI、补洞、新路径／硬件和原声口型边界；
静音片的音频评分不是生成声音资格，接缝NA不升级。

## 2026-09-18 第二次人审与横移预设修正

v11九组18片的真实反馈已按review_id和每条媒体SHA绑定：P4的B冻结源帧绕拍及
C原运动／原声正常，A“没有横移”。只补A，不重复推理已接受的B、C，不扩大成
任意相机路径或新视野质量认证；反馈中的NA接缝仍保持NA。

原A用strength0.03，同时相机移动、look固定在枢轴，使镜头回转补偿保持人物居中。
真实旧warp中央特征首末中位位移约2.57个预览像素，旧成片约0.44像素，确实不明显。
现在`slide`是**平行横移**：相机和look点同向同幅平移、朝向恒定，不锁定人物。
仅空camera_plan选择slide时应用；手写／保存的canonical plan不改写，orbit、
freeze_orbit、source_camera、源时间轨／焦距／roll及旧H3采样接缝逻辑不变。
旧计划要使用新预设，应明确清空路径再重选slide，不用后台迁移改用户文件。

默认strength仍0.08。本次补测为更容易观察的0.12，同一横图及seed，1408×704，
73帧／24fps，保持训练ROI裁剪比例、不拉伸。预设语义和强度同时变化，不称单变量。
100项Meridian CPU及23项实际源fake-DOM检查通过；真实Omega几何的CPUwarp实际
相机朝向逐帧不变，22184个中央点在最终画布的横向位移中位数约-310.85像素，
纵向0，12个小时间步LK均有效。直接首末LK丢对应的v2失败保留，v3改小步跟踪并
独立核对真实点投影，仅修审计方法，不为失败更改生产或缩小移动幅度。
几何warp不等于扩散成片质量。新完整原生INT8三前向补测以
`meridian-slide-gpu-process-v2.json`、`meridian-slide-landscape-gpu-v2/terminal.json`
及`meridian-slide-post-v2/report.json`终态为准；新A仍须人审，不能继承B、C验收。
前端二维示意已同步为“朝向保持不变，不锁定目标”。正式代码在`h3_t8`／`web`，
不是仅研究目录；公开版本1.84.0及旧节点默认／接线不变，未发布。

补测实际终态：授权Omega真实单图几何、原生INT8三前向callback012、完整73
H2641408×704／24fps静音，Job760.547秒exit0cleanup0。独立完整解码和12个
小步LK有效，横移中位累计约-318.49最终画布像素，仍非相机真值或画质认证。
新片SHA `5a6c0add1c4f002d4f3afbd4f41423f49516ec015153aaff9dfcdf237613bb13`。
[两组补测](http://127.0.0.1:8841/review.html)第二组B为此新片，A是旧片；请确认
平滑横移、人物及背景。下面“尚无人审”的旧P4描述以本节A失败／B,C正常为准，
新候选仍待人审，不继承旧B,C验收。

最新回归：96项CPU通过，实际VIDEO8次选窗RGB相同。Meridian自身解码器固定单线程，
避免本机默认FFmpeg/PyAV解码出现不同像素身份而无谓重算；不修改Core或全局codec。
横图和真实VIDEO冻结两条73 H26424fps成片已完整检查通过，但为该解码改动前的源码绑定，
不冒充最新全量GPU／人审。新的真实73帧联合Omega几何已通过；整体Core移动VIDEO空间／时间
实际编辑、保存重载、原生API导出和两次CPUwarp均通过，5节点4边18输入独立核对。
其后仅增加可选只读当前进程allocated/reserved：CUDA尚未初始化时不初始化，
不reset/sync/分配/卸载或设空闲门禁；周期最大观测非独占或精确瞬态峰值，旧实跑不倒填。
自有4MiB实卡计数／RNG／累计peak不变通过，不是重新生成主模型。成片和UI资格均注明
原实跑源绑定。[最新集中审片](http://127.0.0.1:8838/review.html)第9组，尚无人审或发布。

代码位于正式 `h3_t8` 和 `web` 目录，追加四个节点；旧350节点 API schema、
默认值和顺序已按实际 Core 核对不变。公开版本仍为1.84.0／343节点，不代表新版发布。

## 简单接线

`资源配置 → 素材窗口 → 运镜／时间编辑器 → 三前向生成 → 原有Save Video／Topaz／插帧`。
IMAGE或VIDEO只接一个。尚无几何时，可先打开「二维预设规划（不运行GPU）」选择
运镜、强度和帧数；这里是明确标注的方向示意，不是真实深度或生成预览。应用后写入
实际控件并明确清空旧规范路径；关闭不应用不改变设置。无几何时「大号运镜／时间
编辑器」也会进入这个二维面板，不要求先做GPU几何才能设置预设。
第一次选资源，随后选窗口、预设、运行运镜节点看真实灰洞warp，
满意后再生成。只运行运镜节点不加载34GB主模型、不开始扩散采样。

预览模板末端接Core `Preview Image`，且完全不含生成节点；按整图运行也仅做几何与warp。
生成模板末端才接Meridian生成节点。不要在含生成节点的整图按运行却期待自动停在预览。
五份实际API验证的候选模板目前保留在本地artifacts，不混入已接受的正式工作流目录。

默认资源位置如下，任意安装盘均可；高级路径和环境变量可覆盖，不依赖研究盘位置。

|资源|默认位置／环境变量|
|---|---|
|转换主模型|`ComfyUI/models/meridian/meridian_dmd_int8_convrot_comfy.safetensors`／`MERIDIAN_MODEL`|
|原生视频VAE|`ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors`／`MERIDIAN_VIDEO_VAE`|
|上游源码与assets|`ComfyUI/models/meridian/source`／`MERIDIAN_SOURCE_DIR`|
|授权Omega源码|`ComfyUI/models/meridian/vggt-omega`／`VGGT_OMEGA_DIR`|
|授权Omega1B512|源码下`checkpoints/vggt_omega_1b_512.pt`／`VGGT_OMEGA_CKPT`|

上游[Meridian](https://huggingface.co/Viggle/Meridian)固定代码revision
`2083d059d8544ff7eaaf86966b83e4964a904737`的`recam/path.py`、`recam/geometry.py`
由薄适配层按SHA核验；不引入整套Diffusers服务。几何是自己的授权
VGGT-Omega1B512，不替换普通VGGT，不从本节点重分发门禁权重或代码。
主模型要求已合并DMD的原生ConvRot INT8，保留高精度岛；不会先把完整teacher BF16搬上GPU。

在此机器上模型现存研究盘，可用高级路径指向现有文件，无需再复制34GB。
首次真正消费大权重的阶段验证完整SHA，预览不提前读取主模型或VAE全部payload。
配置中的`content_verification=deferred_to_consuming_stage_not_cache_identity`只是文件描述，
不能当成内容已认证；正式生成缓存以消费阶段的真实SHA为身份。

## 空间、时间和比例

预设包含平移、推进、拉远、升降、绕拍、冻结绕拍及源相机。
强度`.08`为固定深度单位的位移；orbit对应8度。大动作容易补洞，不保证画质不退步。
坐标固定为窗口首源相机：x右、y下、z前；单位为首帧ROI中央有效深度，roll=0。
`source_camera`使用每帧实际估计相机，而不是仅两个端点。

大编辑器的空间轨是独立`camera_keys[{t,pos,look,focal,ease}]`，
源时间轨是独立`time_keys[{t,src}]`。两轨各自覆盖输出0..N-1，源索引绝对且不逆向。
后端沿用上游空间Hermite／可选缓动，源时间线性插值后NumPy ties-even取整数；
添加时间关键帧不会改空间切线。焦距是倍率，FOV标注水平角度，不是毫米。
点选只改变该空间关键帧目标，不重新定义深度单位或暗中自动跟踪角色。

有效修改直接写入原`camera_plan`控件，因此工作流保存／重开和API用同一规范。
素材／窗口身份改变后旧计划拒绝复用，需显式重置或重新确认。
输出帧数改变也要同步两个轨；没有人为的帧数上限，但真实VAE长度17k+5与配对assets
必须成立。缺少某长度assets时明确报缺失，不拿另一长度pad/slice冒充。

VIDEO先由Core导出活动trim/crop视图，再按真实PTS归一24fps并仅留所选窗口。
残留SAR／旋转元数据由窗口解码处理，原画面比例不拉伸。1280方形letterbox后按
768训练桶的最近log-aspect选中心ROI；这是比例保真裁剪，不承诺原图完整边缘全保留。

几何预览最多8张缩略图、每张1024抽样点，不是生成成片，也不用于最终warp替代。
最终warp仍按原1280深度、可信度／边缘过滤和逐帧3×3点云splat处理，灰128为洞。
没有新的mask通道、额外RGB通道、空间采样tiles或自动OOM缩小重跑。

## 生成、音轨和外部补丁

固定原生两路480级video token参考、768级目标、raw5120任务embedding；参考VAE使用
posterior seed42抽样，先FP16 round再归一。CFG1／euler／shift3-3，sigmas
`[1,.857142806,.599999964,0]`，实际三次diffusion边界及回调核对；内部为joint AV。
自有文件VAE使用限定的static FP32全驻留，解决原生fused MLP直接读取CPU参数的边界。
不改变外部传入VAE，也不编辑Core或全局unload。

默认交付无声H264 MP4。`source_1to1`仅在源索引逐帧+1的VIDEO路径交付原声，
保留采样率／声道和PTS对应的N/24样本窗；冻结／变速必须显式选silent。
这是Meridian媒体原声政策，不用于原生音色任务掩盖乱说，也不认证新生成口型。
AAC有损及编码padding不冒称bit-exact。完整视频和存在的音频全解码核验后才交付。

标准MODEL／VAE插口为高级组合保留原补丁／委托并提示未经验证，不因LoRA／Sage／Sol
主动禁用或清空。外部可变对象不推断文件内容身份，不使用其不可靠的encode/sample磁盘缓存。
真实原生结构、尺寸、权重格式、有限值和自有缓存完整性仍检查。

## 中断与缓存

所有GPU仅由Core普通执行或本任务独立Job验证启动；前端没有后台GPU线程。
检查点遵循普通Core任务取消，不影响其他实例或全局队列。
分阶段geometry／warp／encode／sample内容身份包含选定素材、窗口、算法／Core源、
完整消费权重SHA、配对assets和规范路径。完整payload按SHA核对；半成品不回读。
锁恢复后旧running收据另存interrupted，自有UUIDpartial才被清理。
关闭缓存或传入不可移植对象时仍保存每次执行的独立UUID收据，失败／取消不会冒充成功。
生成job收据在完整权重核验前建立，保留资源／VAE／采样／解码／交付的具体失败阶段；
VAE加载阶段取消也仅释放该入口自己新建的patcher，不清空其他模型。
磁盘GiB预算不等于显存准入，0关闭；缓存满或锁竞争则完成但不缓存，不偷偷缩小素材。
只淘汰自有stage-SHA、匹配完整收据的可重算缓存，用户源视频和交付文件从不作为目标。

收据新增只读资源观测：默认每秒和阶段前后记录按UUID整卡占用／利用率、当前进程
RSS／private、系统可用内存以及自有Omega／VAE的offload时间。不设置显存门槛，
不初始化／reset CUDA，不控制其他模型，不自动重试；缺少psutil／NVML只写观测错误。
最大／最小值是周期内观察值，包含其他模型或进程，并非节点独占或精确瞬时峰值。
warm cache只标操作已跳过，不冒充又执行／测量该阶段。

## 验证边界

最新91项CPU通过，权威meridian-p4-tests-cpu-v4/report.json，CUDAfalse／exit0／cleanup0。
异常清理现在用Python3.10已有的sys.exc_info；缺少可选add_note或外部note失效时
仍保留原始错误，warning说明清理错误。缺3.11 API是本机模拟，不冒称实际3.10
平台运行资格。只动错误处理，不动采样数学／dtype／媒体policy／旧350schema。

修补前v18源码真实VIDEO73/24fps/512×768一次全帧联合Omega、source_camera1:1
warp、native INT8实际3前向和完整73 H264/24fps/864×1184成功，自有Job601.547秒
exit0cleanup0。原声32kHz双声道N/24窗97334样本，完整AAC解码98304含padding；
独立无推理CPU全解码＋PCM相关.9993855212/RMS比.9994216112通过，exit0cleanup0。
SHA70186c8ab7578922a6d6d79f54afca8e8f6b7d83669709fa7b70eb83646f7c61。
不是bit-exact、新生成口型或人审；输入／权重FP32而保留上游aggregator autocast。
同一完整联合几何的真实冻结／变速两组CPUwarp也通过、源stage SHA保持，非1:1
时间轨明确silent且拒绝原声1:1；不是冻结扩散或整体浏览器移动VIDEO轨验收。
这些GPU/warp源绑定历史v18，不冒称后来只错误处理修补的新源码又跑GPU。
证据meridian-p4-video-gpu-v1/terminal、video-post-audit-v1/report、video-time-cpu-v1/report。

以下80CPU与普通取消恢复为此前v17源码历史证据，不重复推理。
实际Core354/旧350 schema-prefix已验证；此前80项CPU数学、缓存、媒体／资源和边界
测试通过，历史meridian-p4-tests-cpu-v2/report.json，CUDAfalse／exit0／cleanup0。
普通Core取消GPU v1已真实触发第一前向后异常、无MP4和自有VAE清理；测试误判Core
消费后flag应仍为true而失败，失败收据保留，新增CPU真实flag/throw测试复核。
v2现已真实终态：自有Job626.5秒exit0/cleanup0，第一真实native INT8前向后普通
Core中断，随后同进程新请求完成3真实前向和完整73 H264/24fps/864×1184无声成片。
当前资源源码与80CPU绑定相同，外部MODEL wrapper保持，geometry/warp为自有缓存
复用，不又计冷GPU；这是独立普通Core机制验证，不等于浏览器HTTP取消或人审。
成片SHA `da405c20c217680c41e864454c7ed3c77c6db54a1be72aebf65c577c62585a98`。
三份交付链候选API／序列化和真实native file VIDEO插口已通过。当前成片实际由
正式Iris3高清2x交付73/24fps/1728×2368/H264/5.12MB，再ApolloFast2x交付145/48fps/
同尺寸/H264/9.05MB；正式端点约定下少一个末尾插值帧，不硬补146。
两个自有滤镜子Job均exit0/cleanup0，完整成片独立全解码、源与9生产SHA复核通过，
未重跑推理。原父probe最后断言失败exit1/cleanup0仍保留：单独导入诊断复现
xformers在正式packageimport查询GPU能力而初始化CUDA，即使Core参数为--cpu。
不能说此父进程无CUDAcontext，也不将其判作Topaz滤镜失败；没有修改Core或依赖。
权威`meridian-p4-topaz-post-audit-v1/report.json`。DLSS仅API资格，未继承Topaz
实际推理资格；模板仍需自己的授权资源／正式路径，整浏览器交付图或人审尚待验证。
真实转换assets＋已验证授权几何通过新生产warp／点云payload／warm复用；没有把它说成
本次重新运行Omega或完整GPU。前端当前22项fake DOM通过；此前17项与旧JS的
隔离部件编辑／点选／关闭重开／重置证据保持历史绑定，不冒称当前整体几何编辑宿主。
最新独立CPU Core8839实际装正式节点，五候选在原生浏览器打开并保存；二维预设
orbit/.12/90保存reload重开保持，非法89拒绝且未写入。实际保存25执行节点、20边、
93显式输入重建API后Core验证通过，seed1234/fixed不偏移、预览无Generate。
这些历史二维证据不是原生API导出、整宿主真实几何3D或GPU质量证明。两次启动/reload
Core graph-before-init控制台错误与检查工具v1/v2误判的失败收据原样保留，未改Core。
自己的CPU服务和隐藏标签已关闭，不动用户前端。权威meridian-p4-browser-roundtrip-v3。
随后独立CPU Core8839真实几何warp预览成功，在实际大号点云编辑器修改独立位置
.06/焦距1.05、点选look、导出规范JSON以及ComfyUI原生API，保存reload保持。
新authoredwarp普通CPU预览51.31秒成功，真实导出与保存5节点/4边/18显式输入
逐项相同并通过Core API验证。IMAGE的时间轨只有src0冻结，不认证移动VIDEO时轨。
双击第二输入可命中新开的点云改变look，本次经「恢复本次后端路径」恢复并保存；
保留真实交互说明。自有host790.64秒exit0/cleanup0、server自有controller终止，
无children，自己的隐藏tab4关闭；独立审计9.157秒exit0/cleanup0。
权威`meridian-p4-geometry-browser-audit-v1/report.json`，不是新Omega/扩散质量或人审。
四节点portrait73帧24fps、864×1184、显式无声候选实际完成冷授权几何、真实warp、参考
VAE、nativeINT8三前向与最终H264完整解码；自有Job752.375秒exit0／cleanup0。
成片SHA为`d3053ae372e695e3fda1f4de75faedcd220ab1c4b919005de7657e16df43e27c`。
这是`meridian-p3-gpu-v1/terminal.json`绑定的源码版本，不是人审通过或性能承诺。
终态后又修补未知SAR0、缓存源身份和未缓存／生成早期失败收据；这些新源码已独立通过
62项CPU回归，GPU成片不被冒充为修补后的相同源码验证。
P4横竖IMAGE/VIDEO、源声1:1与冻结、错误取消恢复、Topaz/FI完整链及人审仍需逐项补齐。
固定最小P2人审接受不覆盖这些新组合，也不承诺16GB更快更省或全路径质量一致。
