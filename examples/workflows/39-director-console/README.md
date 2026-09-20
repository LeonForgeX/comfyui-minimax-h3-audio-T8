# 曜石导演台

将 `2026-09-20_H3_Obsidian_Director_Starter.json` 拖入 ComfyUI，然后在节点上点击
**打开曜石导演台**。也可以直接使用页面右下角的 **T8 导演台** 按钮。

这是一个干净的入口工作流，不带测试素材或本地文件路径。素材统一从导演台左侧上传；新手模式可整段粘贴
镜头提示词，高级模式可拆分整镜描述与时间事件。保存项目后再执行准备检查或生成当前镜头。

导演台会按所选模式编译 T2VA、I2VA、FL2VA、L2VA、Ref2VA、原音驱动或参考音色图；高级选项还包含
Semantic Bridge、Prompt Relay、FastH3 V2、LowVRAM 与 ChunkFFN。长片、H16-3、TAEH3、Topaz、
Meridian 等仍通过导演台的原生路线包进入各自正式工作流，不伪装成同一条万能生成图。

详细操作和边界见 [导演台说明](../../../docs/DIRECTOR_D1.md) 与
[真实生成桥说明](../../../docs/DIRECTOR_D2A.md)。
