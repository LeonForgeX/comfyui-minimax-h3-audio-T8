# 更新日志 / Changelog

[首页](README.md) · [English](README_EN.md)

## 当前版本：1.85.0

### 2026-09-21 导演台交互修正

- 修复当前镜头生成误检整片、被其他首尾／录音草稿阻断的问题；当前镜头编译、D3 预检与导出使用相同范围。全部生成提交前检查全片，报错标注具体镜头。新建项目不再预置三个未完成示例镜头。

- 全局 D3 模式下可直接编辑右侧功能开关，共用镜头与新镜头同步；取消全局保留当前选择并改为本镜独立设置，修复旧本镜值覆盖显式全局的问题。
- 模型、多 LoRA 与分辨率移到顶部“模型设置”窗口，不再占据剧本区。
- 输入预览采用紧凑高度，宽屏下与素材卡并排，增加收起／展开预览；原图比例和生成尺寸不变。

Meridian 四节点与运镜／源时间编辑器、Avatar 渐进音频驱动、原生音色／情绪、一采动态预览及诊断功能已保存正式工作流。指定样片人审通过，原有采样和接缝配方不迁移。

### 2026-09-20 GitHub 源码更新

- 曜石导演台入口移入 ComfyUI 官方左侧栏，移除会遮挡画布工具栏的右下角悬浮按钮；节点内入口继续保留。
- 新增曜石导演台正式入口与干净启动工作流：统一图片／视频／音频素材、全片共享参考、首尾帧、Ref2VA、录音驱动、参考音色、新手／高级提示词和逐镜生成；高级面板可编译已验证的 Bridge、Relay、FastH3 V2 与低显存组合。
- 导演台新增全片 D3 默认配置与逐镜继承开关、按顺序生成全部镜头；模型、文本编码器、视频／音频 VAE 可从本机 H3 文件选择，LoRA 支持逐条叠加、独立强度、启用／移除与手动排序。
- 导演台新增 0.4／0.5／0.6／0.8 MP 总像素档位；后端按画幅计算并对齐 H3 要求的 32 像素网格，预检报告回显实际宽高与像素量。
- 新增 H16-3 `DeciiaChunkedPass2Sampler` 与正式 I2VA 4+4 模板；安全默认保留一采音频，`refined_exp` 才启用绝对时间轴、重叠交叉淡化和安静尾段保护。
- 修复新版 Topaz 官方 TensorRT 定义中 `[C]`／`[R]` 占位符解析，并保留旧版兼容；本机 `iris-3` 2×短片实际运行及原声保持已验收。
- 本次为同版本 GitHub 源码更新，不重复发布 Registry 1.85.0；安装 GitHub 最新源码后需完全重启 ComfyUI。

本次模型／文档整理：

- [TAEH3 模型](https://huggingface.co/t8star/Taeh3-Comfy)：时序及独立 2D tiny decoder，分别保留 MIT／Apache-2.0 许可。
- [Meridian 模型](https://huggingface.co/t8star/Meridian-Comfy)：合并 DMD 的原生 ConvRot INT8，附正确 Omega 1B512 原始 PT，目录／许可分别注明。
- 节点说明及模型参数提示新增下载入口；旧输入、默认值、采样逻辑和最终音频不变。
- [模型位置与源码准备](docs/MODEL_DOWNLOADS_1.85.md)，避免把权重文件夹误当完整程序或clone到非空目录。
- 中英文首页只保留安装、入口、模型与注意事项；历史内容独立保存，新增首页检查及 CI 防止再次堆积。

[完整 1.85.0 发布说明](docs/RELEASE_1.85.0.md)

## 历史版本

|版本|说明|
|---|---|
|1.84.0|[Semantic Bridge 与 H16 已完成部分](docs/RELEASE_1.84.0.md)|
|1.83.0|[FastH3 V2 与帧数合同](docs/RELEASE_1.83.0.md)|
|1.82.0|[版本说明](docs/RELEASE_1.82.0.md)|
|1.81.0|[版本说明](docs/RELEASE_1.81.0.md)|
|1.80.0|[版本说明](docs/RELEASE_1.80.0.md)|
|1.79.6|[版本说明](docs/RELEASE_1.79.6.md)|
|1.79.5|[版本说明](docs/RELEASE_1.79.5.md)|
|1.79.4|[版本说明](docs/RELEASE_1.79.4.md)|
|1.79.3|[版本说明](docs/RELEASE_1.79.3.md)|
|1.79.1|[版本说明](docs/RELEASE_1.79.1.md)|
|1.79.0|[版本说明](docs/RELEASE_1.79.0.md)|
|1.78.0|[版本说明](docs/RELEASE_1.78.0.md)|
|1.77.0|[版本说明](docs/RELEASE_1.77.0.md)|

旧首页逐条记录：[中文归档](docs/CHANGELOG_ARCHIVE_ZH.md) · [English archive](docs/CHANGELOG_ARCHIVE_EN.md)。
归档中的旧“未发布／待验收”状态仅供溯源，不覆盖当前版本说明。

详细功能、参数和安装合同：[中文](docs/README_DETAILS_ZH.md) · [English](docs/README_DETAILS_EN.md)。
