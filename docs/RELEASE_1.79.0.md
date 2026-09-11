# v1.79.0

## 新增与修复

- 双模型内循环：一采、二采可分别接底模和LoRA，4步后使用已有学习式潜空间放大器，再做4步细化。旧单模型工作流保留。
- 新Plain/Relay工作流位于`examples/workflows/04-long-video/`，默认两段8秒并开启二采视频重叠锁定。旧节点缺省模式不变。
- 修复4+4和续段音频处理；适配指定KJ Sage及普通Sol路线。Sol0.5短片已获对应画音口型认可，不推广为所有参数或后端组合。
- 新Topaz环境检查与视频增强节点，工作流在`examples/workflows/31-topaz/`。需自行安装正式Topaz并准备有效授权和模型；不附带模型或自动下载安装。常规人物2x/1.5x、游戏2x及既有24秒2x样片获对应评分认可。
- OpenVDN不再因`optimized_attention_override`存在就拒绝上游Sol。保留上游对象，不改变其参数；VDN块仍走自己的分组SDPA和线性分支，不通过该override。报告明确标注，不将正常连接等同Sol已生效或叠加提速。实际块替换、冲突的attention hooks、重复VDN及权重补丁检查仍保留。
- R1：Fun旧量化metadata兼容、Creator缓存声明边界、Core兼容性和工作流往返可靠性检查。

## 验证与边界

已评两段8秒新样片四项接受。旧24秒接缝失败保留记录，不代表所有素材都无接缝。
末次177项针对性CPU检查与62项VDN/Core检查通过；前者覆盖新增双模型交付，后者覆盖override限制调整。override调整没有重新跑GPU视频。
更新前候选包实际解包327节点、235工作流通过；发布包另行核验。
星光暂停，正式Topaz GUI同参数对照未执行；不发布星光、Topaz插帧或通用画质/速度承诺。

## English summary

Adds optional dual-MODEL4+learned-latent-upscale+4 loops and official regular Topaz enhancement.
New loop templates default to8s with a reviewed high-resolution context-prefix lock; legacy defaults remain unchanged.
VDN permits a retained upstream attention override while keeping its own block computation. This is graph coexistence, not evidence of Sol acceleration inside VDN.
Starlight remains paused; no official-GUI parity, all-material quality or universal speed claim.
