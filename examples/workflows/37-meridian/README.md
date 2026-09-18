# Meridian 正式工作流（v1.85.0 EXP）

四个业务节点：模型与资源 → 素材与几何 → 运镜／时间编辑器 → 原生3前向生成。
图片／视频使用Core原生加载节点。此目录在正式T8节点中，不在独立开发目录。

- `2026-09-18_T8_Meridian_image_parallel_slide_EXP.json`：IMAGE平行横移，73帧／24fps、strength0.12、silent；对应最后通过的横移配方。
- `2026-09-18_T8_Meridian_video_freeze_orbit_EXP.json`：VIDEO取73帧联合几何、冻结源时间绕拍、strength0.03、silent。
- `2026-09-18_T8_Meridian_video_source_camera_EXP.json`：VIDEO保持每帧源相机与一对一源时间、source_1to1交付原声。

换成自己的合法素材后运行。IMAGE只重建一次，不复制73张再重建；VIDEO按选定窗口联合几何。
输出使用训练bucket及等比ROI裁切：不拉伸，但可能裁掉原图边缘，并非完整原画布。
这三类指定样片已获用户确认，不保证任意素材、远距离新视野或新相机路径质量。

## 模型和源码

工作流路径留空，自动寻找下列标准位置；已有其他盘的资产可在「模型与资源」填实际路径，
不用再复制34GB。保存工作流时会保留你填写的路径。

|资源|标准位置／可选环境变量|
|---|---|
|转换后的merged-DMD native ConvRot INT8|`ComfyUI/models/meridian/meridian_dmd_int8_convrot_comfy.safetensors`／`MERIDIAN_MODEL`|
|H3视频VAE|`ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors`／`MERIDIAN_VIDEO_VAE`|
|Meridian源码及assets|`ComfyUI/models/meridian/source/`／`MERIDIAN_SOURCE_DIR`|
|授权VGGT-Omega源码|`ComfyUI/models/meridian/vggt-omega/`／`VGGT_OMEGA_DIR`|
|授权Omega1B512权重|`ComfyUI/models/meridian/vggt-omega/checkpoints/vggt_omega_1b_512.pt`／`VGGT_OMEGA_CKPT`|

上游[Meridian](https://huggingface.co/Viggle/Meridian)固定源码revision
`2083d059d8544ff7eaaf86966b83e4964a904737`；保留`recam/`和配对长度的`assets/`。
使用[转换工具](../../../tools/convert_meridian_convrot.py)合并配对DMD并转换为Core native INT8，
详见[转换说明](../../../docs/MERIDIAN_CONVERSION_EXP.md)。已转换主模型34,038,894,278字节，
SHA256 `2c31fe6cd336b3d67cc23cdcb87965a925b6be6005062a228c0e2a8e0a8f0e67`。
不是改后缀、LoRA或原完整BF16测试模型；ComfyUI需支持native ConvRot INT8加载。

[VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)须用自己的获授权账号取得正确1B512，
不是普通VGGT。授权源码／权重不在本GitHub重分发；按其要求安装依赖。
Omega和Meridian的上游用途／许可边界仍适用。节点懒加载，不自动代填申请或绕过门禁。

## 前端编辑与音频

先跑预设获得真实几何预览，再在相机节点打开大编辑器。空间轨控制pos／look／focal，
源时间轨独立控制src；修改直接存入camera_plan，保存重开和API读同一规范。
slide相机和look一起移动、朝向恒定，不自动锁定人物；手写或已保存路径不被改写。
换素材或输出帧数后显式重置／重新编辑两条轨。warp中的灰洞不是最终生成质量。

冻结／变速用silent；source_1to1只在完整一对一源时间时交付原声，不保证生成口型。
配对长度assets和17k+5对齐仍必须满足；真实OOM和坏输入正常报出，不设人工显存准入。
指定实验在本机完成，不保证16GB更快、更省或任意长窗口能跑。
