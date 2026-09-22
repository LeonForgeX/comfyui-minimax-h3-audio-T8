# 可复现 H3 阶段性能基线

本项交付测量工具，不开启 pinned offload，也不保证某个优化可提速。
旧 `Measurement` 无 CUDA 副作用；新 `SynchronizedCudaMeasurement` 只供自有隔离 benchmark worker。
不可接到用户共享 Core：reset peak 会影响同一进程的其它测量。

## 采集

1. 固定模型/conditioning/视频与音频 VAE/recipe 的完整 SHA256、Core revision、画布、帧数、帧率、seed、NFE、CFG。
2. `runtime_identity` 记录 `gpu_uuid`、`gpu_name`、`torch_version`、`cuda_version`、`driver_version`、`attention_backend`；额外精度/内核配置也可记录。不能填另一台机器数据。
3. 独占进程显式传入 `exclusive_process=True` 和 CUDA device index。测量器在开始时同步并 reset 一次 allocator peak，每个 stage 边界同步。
4. 用 stage 包住真实的 `model_loading`、`sampling`、`video_vae`、`audio_vae`、`delivery`。无法分拆时诚实命名合并阶段，不伪造细粒度时间。
5. 图输出缓存必须调用 `record_cache_hit()`。有缓存命中、异常、条件或软件/硬件不一致的报告不进入速度对照。
6. cold/warm 各至少 3 次，分别汇总。cold 指本次开始前尚未加载目标模型，warm 指声明的模型驻留/热身状态；工具不自动卸载/热身，运行脚本须记录具体准备过程。

```python
measurement = SynchronizedCudaMeasurement(
    workload, cache_state="cold", runtime_identity=runtime_identity,
    exclusive_process=True, device=0,
)
try:
    with measurement.stage("sampling"):
        result = run_actual_sampling()
finally:
    report = measurement.finish()
```

所有常规加载与采样仍由现有代码完成。测量器不下载模型、不改变 sigma、pin、offload、tile 或注意力。
`resource_scope` 仅指**本进程 torch allocator**的 allocated/reserved 峰值，不是整卡 VRAM 峰值；
也没有声称采得 host RAM 峰值。外部 NVML/RSS 采样若另接，需要注明采样周期与可能漏峰。
异常后最终同步失败会保留测量错误，不能输出成功基线。

## 汇总与验收

`tools/h3_cold_warm_baseline.py` 提供显式隔离执行器。无参数只打印固定配方，不加载模型。
确认独占 GPU 测试窗口后才运行 `python tools/h3_cold_warm_baseline.py --run`；默认模型路径是本机
ComfyUI，其他安装可用 `--core`。每次创建唯一 evidence 目录和源码快照，不覆盖旧证据或操作用户 Core。
恰好三个新 Python 进程，各 cold → warm；warm 只复用 loader 对象，重新 conditioning、固定同一 seed
重新采样、解码和交付，不复用 latent 或图输出。计时不包括 Python/Core 启动及完整权重身份核验；
cold 是模型进程冷启动，不是 OS 磁盘缓存清空。暖态不承诺模型始终全部驻留 GPU，Core 仍正常 offload。
这是直接公共节点调用，不是 GUI/HTTP 排队延迟基准；报告会记录真实 AIMDO/编译器/pin/reserve 状态。
若独立 worker 未启用 AIMDO，不能把该数据冒称启用了 AIMDO 的用户 Core 性能。

固定普通 full INT8 + EMA-B Turbo4，PyTorch attention，832×480、107 模型帧，原有 Trim 输出4秒96帧；
5 GiB reserve、关闭锁页，不改产品默认。真实 forward 数必须为4；视频/音频 VAE `.decode` 分开计时，
数据搬运归属于实际发生的调用阶段，不伪称独立 PCIe 计时。交付复用 SafeAVSave 完整 AV 解码核验。
执行器生成 `pair-N/cold.json`、`warm.json`、日志/出片和总表；运行失败保留证据，不汇总为成功。

```text
python tools/summarize_h3_measurements.py cold-1.json cold-2.json cold-3.json
python tools/summarize_h3_measurements.py warm-1.json warm-2.json warm-3.json
```

工具只读取 JSON，在 stdout 输出总时长与阶段时长的中位数、范围、运行次数。不同 cold/warm、
workload、runtime、stage 顺序，缺少/损坏的阶段、非有限数值均拒绝混算。一次运行只能作 smoke，不能代替重复基线。

只有同机基线表明 VAE 搬运占比值得优化，才单开 EXP 研究冻结 CPU mirror 或 pinned offload；
仍需取消/连续请求/资源释放、INT8与BF16、完整画面/声音核查。任何性能数字不代替画质与音画验收。

## 2026-09-22 本机三对 cold/warm 实测

新证据目录位于开发树 `G:/CodexHome/.codex/worktrees/t8-h3-final-20260920/artifacts/development/h3-cold-warm-a2b078ab17/`，不在正式安装目录。
`frozen-request.json` 绑定源码及五份完整权重 SHA；`pair-1` 至 `pair-3` 各保留
`cold.json`、`warm.json`、worker 日志、exit 收据和两条 MP4；汇总为 `summary.json`，
CPU 全片复核为 `media-audit.json`。此前用户切换任务时中断的旧目录不计入此结果。
实际执行工具另存 `controller-source.py` 并与原清单SHA一致；实测后仅把后续源码身份改为
逻辑相对路径，避免随机证据目录影响跨次对照。`source-identity-note.json` 记录此规范化，原六份计时不改写。

RTX 4060 Ti 16 GiB、Torch 2.10.0+cu130、CUDA 13.0、驱动616.56；
PyTorch attention、AIMDO未启用、pin关闭、reserve 5 GiB。参数为上文固定配方，
每次实测4次 DiT 前向；三进程各生成 cold 后重新采样 warm，没有复用图输出。

| 同类3次统计 | cold | warm |
| --- | ---: | ---: |
| 总时间中位数 | 111.845秒 | 104.329秒 |
| 总时间范围 | 106.656–112.355秒 | 104.229–104.395秒 |
| loader文件读取/对象复用中位数 | 2.050秒 | 0.00043秒 |
| conditioning中位数 | 7.231秒 | 4.513秒 |
| sampling中位数 | 91.121秒 | 89.766秒 |
| video VAE中位数 | 9.549秒 | 8.268秒 |
| audio VAE中位数 | 0.441秒 | 0.313秒 |
| Trim与交付中位数 | 1.465秒 | 1.446秒 |
| 本进程allocator峰值 allocated | 9.031 GiB | 8.928 GiB |
| 本进程allocator峰值 reserved | 9.656 GiB | 9.342 GiB |

阶段各自取中位数，不要求其和等于总时间中位数；模型设备搬运仍归实际调用阶段。
这些 cold/warm 差异不归因为某项加速收益，不能推算 AIMDO GUI 服务性能，也不是整卡/RAM峰值。

六片均完整解码为832×480、96帧、24fps、4秒；均有4秒32kHz双声道音轨，
解码PCM为128000 samples、无非有限值，文件 SHA 与保存收据一致。
三条cold彼此RGB8/PCM hash一致，三条warm彼此一致；cold与warm**不位精确**：
全片RGB8 MAE 3.4414/255、PSNR33.0921dB，PCM相关0.998398。仅记录差异，未归因、未人审。

三个自有worker均exit0，控制器exit0；复核无残留baseline/audit进程，GPU已释放。
验证结论仅为本配方阶段测量和完整AV/PCM机械验收；不授予画质、口型、跨模型兼容、
普遍16GiB安全或新优化收益资格。生产默认pin/offload和用户Core均未修改。
