# H3 TRT VAE：安装与使用（1.78.0 可选 EXP）

这是原版 H3 视频 VAE 的 TensorRT 执行方式。原生视频 VAE 权重仍然需要；音频 VAE、H3 生成底模和 LoRA 不变。不是新训练的 VAE，也不是把整个视频生成过程换成 TensorRT。

参考实现：[ComfyUI-H3VAE_TRT](https://github.com/lihaoyun6/ComfyUI-H3VAE_TRT)。ONNX 来源为 [MiniMax-H3-VAE-ONNX](https://huggingface.co/lihaoyun6/MiniMax-H3-VAE-ONNX)，当前代码锁定 revision `deacbbd48dd3bba13461fa29c2ea8ffb1042c5da` 的已核对文件。flex 解码图只调整输入输出形状声明，不改变算子和权重；T1 图是原版单帧编码路径的单独导出，不是新训练模型。

## 当前边界

- 开发环境实测为 Windows、单张 RTX 4060 Ti 16GB、Python 3.12、PyTorch 2.10/cu130、TensorRT cu13 `10.13.3.9.post1`。不是对所有 RTX、Linux、Python 和 CUDA 组合的承诺。
- 不安装本功能依赖也能继续使用旧节点。安装检查仅检查文件，不等于 GPU 功能测试成功。
- 想试用时优先选择 decoder-only，参考编码仍走原生；它并不保证总耗时更少。Full 编码差异可能影响后续画面，本机单图编码也没有测出收益，不默认开启。
- 真实短片各路线和部分特殊接口已测，已审短片和静态文字获得用户整体认可。长片测试已取消，不构成长片已通过的结论。
- W4A16 已完成本机真实编译和同潜空间 0.5MP／73 帧解码；相对原生的 PSNR 为 45.32dB，FP16 对照约 61dB，量化差异明显更大，本次集中审片整体可接受。引擎文件较小不等于已测得显存节省；不宣称它更快或任意素材都能保持画质。

## 文件放哪里

相对 ComfyUI 根目录：

```text
models/vae/
  minimax_h3_video_vae_fp16.safetensors
  h3_trt/
    minimax_h3_vae_decoder.onnx
    minimax_h3_vae_decoder.onnx.data
    minimax_h3_vae_decoder_flex.onnx       # 从上面两个文件生成
    minimax_h3_vae_encoder.onnx            # Full 的 T17 编码器
    minimax_h3_vae_encoder_t1.onnx         # Full 的单帧编码器，单独准备
    minimax_h3_vae_decoder_w4a16_awq.onnx   # 可选，短片已执行，已审样片整体可接受
    runtime/site-packages/                # 独立 TensorRT，不覆盖全局环境
    engines/<本机编译目录>/
      model.engine
      manifest.json
```

编译记录在 `output/MiniMaxH3/TRT-VAE/builds/<本次编号>`。不要把 engine 放到 `diffusion_models`，也不要只复制 `model.engine` 丢掉 `manifest.json`。

## 依赖与准备

使用你实际运行 ComfyUI 的 Python。下面的 `<ComfyUI-Python>` 和 `<ComfyUI>` 是需要替换的路径，不要直接照抄尖括号。

```text
<ComfyUI-Python> -m pip install --target <ComfyUI>/models/vae/h3_trt/runtime/site-packages tensorrt-cu13==10.13.3.9.post1
```

本节点还使用 `psutil`、`nvidia-ml-py`（导入名 `pynvml`）、已有的 PyTorch 和 safetensors。准备 flex ONNX 时需要 `onnx`。缺什么再补什么，不要为此升级或替换 ComfyUI 的 torch、CUDA、现有 TensorRT。文件安装完成后先运行安装检查，错误中会列出缺失项。

下载同一固定 revision 的 decoder ONNX 和 `.onnx.data` 后，在节点包目录运行随包提供的 CPU 图准备工具：

```text
<ComfyUI-Python> trt_vae_prepare_flex.py --source <ComfyUI>/models/vae/h3_trt/minimax_h3_vae_decoder.onnx --output <ComfyUI>/models/vae/h3_trt/minimax_h3_vae_decoder_flex.onnx --report <ComfyUI>/models/vae/h3_trt/flex-preparation.json
```

它要求新输出路径，不覆盖源文件；校验原始文件和数据哈希，只修改形状声明。它不编译，也不算实际推理验收。

Full 模式需要另外准备 T1 图。在节点目录显式运行以下命令（会短暂使用 GPU，与其他任务串行；日志目录取尚未使用的名字）：

```text
<ComfyUI-Python> trt_vae_prepare_t1.py --native-vae <ComfyUI>/models/vae/minimax_h3_video_vae_fp16.safetensors --core-directory <ComfyUI> --output <ComfyUI>/models/vae/h3_trt/minimax_h3_vae_encoder_t1.onnx --run-dir <ComfyUI>/output/MiniMaxH3/TRT-VAE/prepare-t1 --execute
```

它从原版权重导出真正的单帧路径，不需要本项目研究素材。新图必须与已经验证的 T1 图 SHA 完全一致才会保存到目标路径；原生与专用路径也会用有限值测试输入核对。已有相同图直接复用；已有不同文件不覆盖。其他 Core／导出器版本可能无法产生相同文件，届时明确停止，不改名蒙混或自动放宽校验。然后在编译工作流中分别选择 `encoder-t1` 和 `encoder`，得到两套引擎。只用 decoder-only 不需要这一步。

## 怎么连

1. 先单独运行安装检查，再单独运行编译工作流，选择 `decoder-flex`。编译一次只处理一张图，不和其他 GPU 工作并发。
2. 编译成功后刷新模型列表，将生成工作流里的占位值换成刚编译出的目录名。不要选择别人机器的引擎编号。
3. 解码节点的 `video_vae` 接原来视频 VAE 的位置。音频 VAE 保持原连接。Full 节点额外选 T1 和 T17 编码引擎。
4. `runtime_directory` 留空表示上面的默认目录；已有独立安装可以填绝对路径。引擎绑定编译时的运行目录和文件身份，移动后需重新编译。
5. `cpu_output_budget_mib` 是 CPU 像素缓冲上限，不是显存上限；默认 1024 MiB。超出会明确报错，不静默裁帧。

## 编译、显存和速度

编译要求至少 **12000 MiB 空闲显存、24 GiB 空闲内存、20 GiB 空闲磁盘**；运行中持续守护资源，不够会停止本次子进程，保留旧引擎。不会强制关闭其他程序，也不会自动降低安全门槛。编译有独立超时和取消清理。

每次调用按需加载、校验和释放引擎，目前不常驻缓存。更换显卡、驱动、运行库或移动绑定路径后，应显式重新编译；节点不会在点生成时偷偷重建。

本机同潜空间、0.5MP 短片的裸热解码中位耗时约原生 22.96 秒、TRT 10.64 秒，**只代表那一组视频解码内核测试**，不包括首次编译、引擎加载、采样、音频和保存，不是完整视频生成快两倍。

随后用当前公共 decoder-only 节点做了同一 73 帧短片的串行对照：每次新建 VAE，包含加载、完整解码、像素张量保存、相同 H.264 编码和原音轨复制。排除第一组观察后，后两次的中位耗时为：

| 测量阶段 | 原生 | TRT |
| --- | ---: | ---: |
| VAE 节点加载 | 3.32 秒 | 7.69 秒 |
| 解码及数据传输（TRT 包含本次引擎校验、加载、释放） | 24.63 秒 | 21.17 秒 |
| 从加载到最终 MP4 | **30.54 秒** | **31.33 秒** |

**这组短片没有测出总耗时收益**，因此目前只保留为可选 EXP，不建议为了这组结果专门安装。上表不包含 H3 采样、编译、进程启动及导出后的证据哈希检查；也不是清空系统文件缓存的冷启动测试。第一组有 CPU 审计活动重叠，未用于上述后两次中位数。已缓存 VAE 的重复工作流、其他显卡或尺寸可能不同，未测结果不作保证。

公共 Full／Decoder 节点已经实际运行过短片人脸裁剪→编码→解码→回贴接口：原音频潜空间及遮罩外原片像素保持，未偷偷退回原生解码。这只是接口联调，不是重新采样修脸或改善画质的证明。5 张工作流通过独立后台的实际 Core 参数／连线校验；没有操作用户前端。第五张是同潜空间原生／TRT顺序对照，只有一个采样器，音频只解码一次，两份成片共用同一 AUDIO 输出。本次集中审片获得整体认可，不代表任意素材或长片均通过。

## English setup summary

This is an optional, unreleased TensorRT execution backend for the original H3 video VAE, not a newly trained model. Decoder-only keeps native encoding; Full requires separately compiled T1 image and T17 video encoders. Audio and sampling are unchanged.

The tested environment is Windows, one RTX 4060 Ti 16GB, Python 3.12, PyTorch 2.10/cu130 and TensorRT cu13 `10.13.3.9.post1`. Install TRT into the isolated `models/vae/h3_trt/runtime/site-packages` directory using ComfyUI's Python; do not replace the existing torch/CUDA stack. Optional dependencies are psutil, nvidia-ml-py and ONNX for graph preparation.

Place the pinned upstream ONNX files under `models/vae/h3_trt`; keep decoder `.onnx` and `.onnx.data` together. The bundled `trt_vae_prepare_flex.py` derives a separate shape-annotation-only flex graph. For Full mode, use the explicit guarded `trt_vae_prepare_t1.py` command above with the native VAE and Core paths. It needs no research media, preserves existing files and only publishes the exact previously validated graph hash. Other exporter/Core versions may require separate qualification; a mismatch stops rather than silently substituting a graph. Run the separate compile workflow for each graph, then select the local engine directories. Each bundle must retain both `model.engine` and `manifest.json`.

Compilation needs 12000 MiB free VRAM, 24 GiB free RAM and 20 GiB free disk, is serial and guarded, and never overwrites an existing bundle. Inference never silently compiles or installs packages. Engines are bound to the tested runtime, device and driver identities. W4 compilation and one complete short decode now pass execution checks, but its native-relative PSNR is 45.32dB versus about 61dB for FP16; the combined reviewed samples were accepted overall, without a universal quality guarantee or a measured VRAM/speed benefit. Long-video validation is explicitly excluded; the combined short-video and static-text review received overall user acceptance.

The actual public decoder-only output-stage test did **not** improve total time: native 30.54s versus TRT 31.33s for the same 73-frame, 0.5MP clip. These are medians of two later observations with fresh VAE objects, including loader checks, full decode, tensor save, identical CPU H.264 encoding and original-audio copy. They exclude H3 sampling, compilation, process startup and post-export evidence hashing. The first pair overlapped a CPU audit and is not used for this comparison. Natural OS/library caches remain; this is neither an OS-cold benchmark nor a whole-generation speedup. The earlier bare warmed-kernel 2.157x result must not be presented as end-to-end performance. Full/Decoder short FaceRefine interfaces and five headless workflow validation checks pass, without claiming new face-resampling quality or browser export testing. The fifth workflow samples once, decodes native AV first, then decodes that same video latent with TRT; both output files share one decoded and trimmed AUDIO output.
