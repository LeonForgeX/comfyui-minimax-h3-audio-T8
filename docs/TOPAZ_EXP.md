# 正式 Topaz 高清后处理（开发候选）

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
插帧/辅助或未分类定义，另有6份非模型JSON。不是89个模型都已安装或可运行。
Iris3的1x/2x有候选文件、4x缺少候选；SLP2.5明确标为独立路线待验证。
证据 `artifacts/topaz-installed-catalog-v2/report.json`，未导入Torch或启动推理。

执行子进程会清除继承的外部Topaz引擎、模型和许可证环境变量覆盖，只设置已选正式目录；
不读取授权文件，也不改用户或父进程环境。正式安装自己的正常授权仍由官方程序处理。
该隔离改动已通过公共节点Iris1x复测，母版字节与前次一致；见
`artifacts/topaz-public-official-env-gpu-v2/receipt.json`。不据此宣称2x或星光已完成。

视频节点接环境输出与原生 Load Video。先把裁切/剪辑/帧批次保存为文件，不能静默忽略编辑。
`model_id`是定义文件名去掉`.json`，例如`iris-3`。`scale`为1x原尺寸增强、2x或4x放大，
需要准备对应权重；不是调帧率。`vram_fraction`是外部引擎显存预算比例，不是总显存大小。
高级`parameters_json`默认`{}`沿用程序默认；例如`{"noise":0.3,"details":0.5}`，
只接受选中模型和当前版本真正支持的参数。不要把星光型号填进常规节点。

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

输出为无损 PNG48 MOV（含AAC）或 FFV1 MKV，保留原音频包并复核解码PCM。浏览器不一定支持这种母版，后续可接原生 Save Video
转 H.264 预览；预览不替代母版验证。当前接入只接受渐进 SDR、方形像素、固定尺寸且时间轴
可证明为 CFR 的文件。HDR、VFR、旋转/裁切、掉帧等会明确提示先处理，不自动修改素材。

每次任务独立目录，执行前后核对源片、程序、定义和候选权重 SHA；编码保留输入时间基，
执行后核对尺寸、逐帧时间轴、音频包/填充信息、解码PCM与音画相对起点；父进程再次核对源片及母版身份。失败只留下诊断/未发布候选，
不覆盖源片。不声称任意帧恢复。取消/超时只清理本任务的 Windows Job 进程树，不关闭用户
Topaz。一个 ComfyUI 队列按顺序运行；不要同时在其他程序开启 GPU 增强/生成。

当前证据：正式1.6.1程序三份签名有效；Iris1x的512×256、72帧/24fps机械测试通过，
95个音频包字节和相对时间不变。首轮发现编码时间基舍入造成10ms错位，已修正并复测。
以上为早期1x历史证据，当前2x和星光状态以本文顶部为准；插帧不在本轮范围。

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
The output is lossless PNG48 MOV for AAC, or FFV1 MKV for other supported audio. Original audio
packets, priming/padding and decoded PCM are verified. Browser preview may require a
separate H.264 export. Frame count, rational timing, audio bytes and AV offset are verified before
publication. Unknown/HDR/VFR/edited inputs are not silently converted. Tasks are isolated and never
overwrite the source. Iris2x public-node execution now passed72-frame timing and96 original
audio-packet/bitexactPCM checks. Starlight, GUI parity and visual acceptance remain pending.
Do not interpret component installation as successful model inference.
