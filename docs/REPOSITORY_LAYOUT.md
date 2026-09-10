# Repository layout / 目录结构

这次只整理文件位置，不调整生成算法、模型、采样参数或节点接口。

1.78.1 将根目录 Python 文件从 318 个减少到 4 个。324 个完整节点接口和
231 份工作流文件均与整理前一致。完整 CPU 测试名单分段续跑核对后为
4,018 项通过、136 项跳过，没有失败或漏项；实际安装包导入也已检查。
本次没有重新进行 GPU 生成测试，不扩大已有功能的质量或性能承诺。

| 位置 | 内容 |
| --- | --- |
| 根目录 `__init__.py` | ComfyUI 加载入口，保留原位置 |
| `h3_t8/` | 节点实现、后台工作进程和随包第三方代码 |
| `examples/workflows/` | 可直接使用的工作流，目录和内容不变 |
| `web/`、`subgraphs/` | 前端扩展和子图，位置不变 |
| `docs/`、`tests/`、`tools/` | 文档、回归测试和开发工具 |

普通用户更新后重启 ComfyUI 即可，不需要重新下载模型或修改旧工作流。
不要在旧目录里手工复制另一份代码；正常 Git/Manager 更新会处理文件移动。
根目录保留 `trt_vae_compile.py`、`trt_vae_prepare_flex.py`、
`trt_vae_prepare_t1.py` 三个兼容命令入口，已有文档中的调用方式继续有效。

## Developers

The root entry adds `h3_t8/` **before** the root to its package search path. Existing
qualified imports such as `<plugin>.nodes`, `<plugin>.sampling` and
`<plugin>.dlss_fi_backend.process` retain their names. Stale root files therefore
cannot take precedence. Avoid importing the same implementation through a second
`<plugin>.h3_t8.*` namespace. The implementation directory is not a new public API.

Code reading implementation files by filesystem path must now use
`<plugin-root>/h3_t8/<filename>`. This includes private research tools that bypass
the package entry. Worker and bundled-vendor paths have been updated together;
licenses remain with their original vendored sources. The Registry archive must
include this directory recursively, without caches, tests, research artifacts,
models, DLLs or engines. Node IDs, widget order, inputs/outputs, model directories,
`WEB_DIRECTORY` and all existing workflow files are unchanged.
