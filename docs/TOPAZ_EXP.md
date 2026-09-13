# 正式 Topaz 高清后处理（开发候选）

> 2026-09-14 当前行为：普通高清默认由正式Topaz直接输出高质量H.264 MP4，
> 不再把每帧保存成16-bit PNG/FFV1，也不要再接ComfyUI `SaveVideo`。
> 本机同一Iris2x源片截取60帧实测约11.5秒、2.9MB；此前错误的PNG48路线对完整15秒
> 产出12.9GB并导致F盘写满，现已改为高级`lossless_master`才会启用。
> 新增独立`Official Topaz · 视频插帧`节点；高清与插帧不会混用模型或滤镜。
> v1.79.5进一步按模型定义过滤手动滑块，运行前选择音频安全的MP4/MKV封装，
> 拒绝静默10bit→8bit，并把推理帧与后续审计阶段同步到ComfyUI进度条。

使用节点时**不需要保持Topaz软件开启**。必须已经安装正版Topaz、登录有效授权，并先在
Topaz软件里下载准备所选模型；节点随后直接调用正式安装中的`ffmpeg/tvai_up/tvai_fi`。
若首次运行提示授权失败，再打开Topaz确认登录和模型能在软件内运行。

> v1.79.0发布常规增强节点；星光继续暂停。下面的历史“待人审”以本页最新集中人审结果为准，不影响尚未进行的官方GUI同参数对照。

最新集中人审：用户已将人物2x、自定义1.5x、游戏2x和24秒2x的画面与声音评为
正常／可接受。24秒组只记录增强侧播完，不据此认定左右完整比较或正式GUI同参数等价。
本次反馈绑定review_id98b05c3cec09bc24ce29579caee9ddc2e35042a6ab36c5a28d7228a67e42e4bc，
不再要求重复审核这些已经接受的增强结果。星光仍按用户要求暂停，GUI对照仍未执行。

**2026-09-11覆盖以下历史状态：** Iris2x已通过公共节点完整3秒测试：1024×512 →
2048×1024，72帧/24fps，96个原音频包、AAC填充、解码PCM和音画起点全部保留。
来源是用户已接受声音的修复短片，画质尚待人审。1.6.1正式星光引擎和SLP2.5权重已按
用户授权补装；实际Neuroserver探测返回15／License checkout failed，尚未有效推理。
用户随后明确“星光晚点再说，先把其他的弄好即可”，因此星光暂停，不继续探测或处理授权。
以后恢复时需先确认正式Topaz中星光能正常使用；不绕过授权、不使用star2.6。
此处安装是本机明确授权的开发准备；公共节点本身仍不自动安装或下载。
当前未发布，不能把2x机器检查当成星光或所有素材验收。

常规路线已补齐游戏3秒和连续24秒的2倍文件测试。24秒结果为2048×1024、576帧、
24fps，原752个音频包、AAC填充、解码PCM及音画起点均保持一致；游戏片保留原96个音频包。
证据分别在`artifacts/topaz-relay24-repaired-2x-v1/receipt.json`和
`artifacts/topaz-game-repaired-2x-v1/receipt.json`。这证明该输入的文件交付完整，
不证明增强更好；正式GUI同参数导出对照和集中画质人审仍待。24秒源片后段原本有不需要的文字，
审片会明示，不能把普通超分当作删除文字或修复剧情的工具。

**最新验证：** 已修复 MP4 AAC 填充信息在 MKV 转换中丢失的风险。含 AAC 的视频使用
PNG48 无损 MOV，其他已支持音轨使用 FFV1 MKV；不转码音频。新增检查验证 Skip Samples、
discard padding 和实际解码 PCM，而不只看压缩包。实际公共节点已完成一条1024×512、
72帧/24fps MP4 → Iris1x → MOV：95音频包、PCM及音画相对时间全部保留，自动尺寸读取通过。
这不代表2x/4x、星光、任意音频编码或人审通过。

CPU证据已进一步确认：统一给编码器同一rgb48be输入后，PNG-MOV与FFV1-MKV解码像素逐位相同；
PNG-MOV的解码PCM与原MP4完全一致，而MKV多出1024个采样。
见`artifacts/topaz-container-priming-cpu-v3/report.json`。整合后实测见
`artifacts/topaz-public-aac-1x-gpu-v1/receipt.json`，70项聚焦CPU通过。

这条路线处理已生成的视频，不改变 H3 采样，也不替换 DLSS。需要自己安装并正常使用正式
Topaz Video；本项目不包含程序、商业模型或授权文件，不安装、不自动下载、不代替你接受协议。
没有安装 Topaz 时，其余 T8 节点照常导入。

环境节点填三处路径：程序目录、模型 JSON 定义目录、模型数据目录。它们可能不同。
本次机器的正式程序为 `G:/Program Files/Topaz Labs LLC/Topaz Video`，定义在
`C:/ProgramData/Topaz Labs LLC/Topaz Video/models`，数据在
`G:/ProgramData/Topaz Labs LLC/Topaz Video/models`。其他电脑要填自己的正式路径。
`G:/star2.6`不是运行来源。环境检查不运行增强，也不能证明所有模型都可用。

环境输出的 `report_json → model_catalog` 可查模型 ID、名称、对应倍率权重和参数。
普通增强、需要 Neuroserver 的模型、插帧/辅助模型分别列出；后两类不能填入普通增强节点。
每个倍率会写明“缺候选权重”或“有候选文件但运行未验证”，不会把安装成功写成推理通过。
模型参数只列该定义与当前滤镜共同支持的项目，附范围及定义中的默认值。
清单只读取目录与小型定义文件，不遍历授权数据、不运行增强，也不扫描数GB权重的完整哈希；
真正执行时才核对选中权重身份。此信息会随用户选择的正式安装而变化。

本机实际清单检查识别89个模型定义：52个普通增强定义、6个Neuroserver定义、31个
插帧/辅助或未分类定义，另有6份非模型JSON。不是89个定义都属于本节点下拉框。
当前已通过正式授权引擎为常用1024×512路线准备下拉框内14个普通增强模型的官方声明倍率，
以及5个插帧模型；模型数据目录约3.45GiB。每组下载都实际加载至少一帧，40项候选检查无缺失，
但这不等于所有模型、素材和画质均已人审。商业权重不随本项目发布，公共节点仍保持
`download=0`，其他电脑仍须在正式Topaz内自行准备。

权重预检依据正式定义中各后端、各倍率声明的网络模板，不再假定文件名一定含目标倍率。
因此Rhea 1x/2x复用定义中的4x网络、Nyx XL无`-1x-`文件名、Theia共享fnet都能正确识别并绑定，
不会再出现官方引擎能跑但节点误报“未下载”的情况。

执行子进程会清除继承的外部Topaz引擎、模型和许可证环境变量覆盖，只设置已选正式目录；
不读取授权文件，也不改用户或父进程环境。正式安装自己的正常授权仍由官方程序处理。
该隔离改动已通过公共节点Iris1x复测，母版字节与前次一致；见
`artifacts/topaz-public-official-env-gpu-v2/receipt.json`。不据此宣称2x或星光已完成。

视频节点接环境输出与原生 Load Video。先把裁切/剪辑/帧批次保存为文件，不能静默忽略编辑。
`model_id`下拉框列出常用正式增强模型定义ID；未收录的新模型可在高级`custom_model_id`
填写定义文件名去掉`.json`，并覆盖下拉选择。`scale`为1x原尺寸增强、2x或4x放大，
需要先在正式Topaz中准备对应权重；不是调帧率。`vram_fraction`是外部引擎显存预算比例，不是总显存大小。
`parameter_mode`提供三种清楚的用法：`model_defaults`沿用模型默认；`auto_estimate`让Topaz
分析指定数量的帧；`manual`启用独立中文参数，包括抗锯齿/去模糊、降噪、恢复细节、去光晕、
锐化、去压缩伪影、输入预加噪、输出颗粒/尺寸、模型色彩校正和原片混合。
手动模式只把所选模型定义中明确支持的六项恢复参数交给引擎；Artemis、Gaia、NXL等
固定预设不会再因这些滑块报错，被忽略的项目会写入`parameter_audit`。
高级`parameters_json`只是未来参数或精确覆盖接口，普通用户保持`{}`即可；例如
`{"noise":0.3,"details":0.5}`。只接受当前滤镜与所选模型支持的参数。不要把星光型号填进常规节点。

常规Topaz节点不再设置“启动时必须空闲12GiB显存”的固定门槛；不同模型、倍率和素材的
实际需求不能用一个数准确表示。`vram_fraction`交给正式Topaz引擎管理显存预算，控制器仍会
记录启动快照，并在遥测无效、显存进入极端危险区或运行中持续资源不足时停止自己的任务。
系统内存仍需至少16GiB可用，防止外部增强与逐帧审计压垮宿主机。

高级`output_directory`留空时写入`ComfyUI/output/MiniMaxH3-Topaz`；也可填写另一个本地
绝对目录。任务不会覆盖已有输出。默认H.264路线记录保守工作空间估计并在目标盘低于256MiB
时停止自己的进程。高级`lossless_master`仍按RGB48上界审计并保留1GiB停止线；该模式只供
需要逐像素归档的开发审计，15秒视频就可能达到十几GB，不适合普通保存。

高级 `size_mode` 默认 `scale`，旧工作流仍按原倍率运行。切换为
`target_dimensions` 后填写 `target_width/target_height`，此时忽略上方倍率。
宽高须为32到8192的偶数、保持原视频比例，且为原尺寸的1至4倍；
例如1024×512可填1536×768（1.5倍）。不自动裁剪、拉伸、补边或取整。
自定义尺寸的候选权重清单不代表引擎一定选了某个倍率权重，缺失项目仍会列出；
执行失败不会自动下载或改用普通缩放。自定义模式先由Topaz增强，再用Lanczos调整
增强结果到指定尺寸；正式滤镜的宽高参数仅估算模型倍率，不是精确输出尺寸。
实际1.5倍已完成72帧1536×768，原96音频包、填充和解码PCM全部保留，画质仍待人审。
119项Topaz聚焦CPU测试通过，包括实际交付JSON及真实worker控制流的模拟外部程序负例。
旧API缺省和新自定义参数序列化均通过；固定倍率路径不增加重采样。
CPU模拟不替代正式引擎，实机证据仍以上述实际输出为准。
此限制仅适用于普通增强，不套用到星光。

English: `size_mode=scale` preserves existing workflows. Select
`target_dimensions` to supply both even dimensions (32–8192), with the original
aspect ratio and a 1–4x size ratio. The fixed scale selector is then ignored.
No automatic cropping, padding, stretching, rounding, downloads or resize fallback.
Candidate weight hashes do not prove which variant the engine selected.
Custom sizing explicitly applies Lanczos after Topaz enhancement; the engine's
native output dimensions are recorded separately. The 1.5x short-clip test passed
frame/timing and original-audio checks; visual quality remains pending.

默认输出为高质量H.264，NVIDIA NVENC preset p5、HQ tune、CQ16；AAC等MP4兼容音轨或
无音频使用MP4，其他没有预延迟元数据的音轨使用MKV。无法证明安全映射的非AAC预延迟/
填充会在GPU推理前拒绝，并提示先转换AAC。原音频包直接复制并复核解码PCM。
节点本身已经保存完成，`saved_path`就是最终文件，
不要再连接原生SaveVideo重复转码。高级`lossless_master`才输出PNG48 MOV（AAC）或FFV1 MKV。
当前接入只接受渐进8bit SDR、方形像素、固定尺寸且时间轴
可证明为 CFR 的文件。HDR、VFR、旋转/裁切、掉帧等会明确提示先处理，不自动修改素材。

进度条0～90%对应准备和正式Topaz逐帧推理，92～100%对应输出探测、音频PCM复核与严格
完整解码。后半段GPU占用下降是审计阶段，不代表任务卡死。

每次任务独立目录，执行前后核对源片、程序、定义和候选权重 SHA；编码保留输入时间基，
执行后核对尺寸、逐帧时间轴、音频包/填充信息、解码PCM与音画相对起点；父进程再次核对源片及母版身份。失败只留下诊断/未发布候选，
不覆盖源片。不声称任意帧恢复。取消/超时只清理本任务的 Windows Job 进程树，不关闭用户
Topaz。一个 ComfyUI 队列按顺序运行；不要同时在其他程序开启 GPU 增强/生成。

当前证据：正式1.6.1程序三份签名有效；Iris1x的512×256、72帧/24fps机械测试通过，
95个音频包字节和相对时间不变。首轮发现编码时间基舍入造成10ms错位，已修正并复测。
以上为早期1x历史证据，当前2x和星光状态以本文顶部为准。

## 独立插帧节点

`Official Topaz · 视频插帧 (T8 EXP)`调用`tvai_fi`，只做2x/4x帧率转换：24→48、
30→60或24→96，保持原时长、尺寸和音轨，不做慢动作或高清放大。Apollo/Chronos质量版
偏质量，Fast版偏速度；实际可用项由本机正式定义和已下载权重决定。`duplicate_threshold`
默认0.01，0或负值关闭重复帧检测，过高可能误判正常静止帧。输出同样是直接保存的H.264 MP4，
无需SaveVideo。正式Apollo `apo-8` 已通过授权引擎下载，并用生产worker完成一条1秒
24→48fps、带AAC音轨的真实机械验证；命令保持`download=0`，时长、音频包、解码PCM与
严格解码全部通过。该结果尚未完成人眼画质验收，也不覆盖Chronos、Aion等其他模型；
首次使用未准备的模型前仍须在正式Topaz中下载。

## English

Optional pixel-domain postprocessing through a separately installed, licensed official Topaz Video.
No proprietary binaries, weights, automatic downloads or license modifications are included.
Set the installation, model-definition and model-data directories separately. Use an untrimmed,
uncropped native file-backed VIDEO. The regular route uses `tvai_up`; Starlight is a separate
Neuroserver route. Official components are now installed, but the actual probe
failed its official license checkout; no valid Starlight inference is claimed.
The user subsequently deferred Starlight. Regular Topaz work continues; Starlight
will not be retried or presented as delivered until the user resumes that scope.

Choose the installed model definition ID and1x/2x/4x.1x is enhancement at original size, not upscale.
The default output is a directly saved high-quality H.264 NVENC MP4; do not add SaveVideo.
Lossless PNG48 MOV/FFV1 MKV remains an explicit, potentially huge audit-only profile. Original audio
packets, priming/padding and decoded PCM are verified. Frame count, rational timing, audio bytes and AV offset are verified before
publication. Unknown/HDR/VFR/edited inputs are not silently converted. Tasks are isolated and never
overwrite the source. Iris2x public-node execution now passed72-frame timing and96 original
audio-packet/bitexactPCM checks. Starlight, GUI parity and visual acceptance remain pending.
Do not interpret component installation as successful model inference.
