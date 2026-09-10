"""Frontend candidates for already sampled progressive pilot recipes; no inference."""
from __future__ import annotations

from copy import deepcopy
import uuid

try:
    from .api_to_frontend_workflow import convert
    from .frontend_workflow_compat import normalize_native_widget_inputs
except ImportError:
    from api_to_frontend_workflow import convert
    from frontend_workflow_compat import normalize_native_widget_inputs


def build_workflow(prompt, object_info):
    graph = deepcopy(prompt)
    sampler = graph["10"]
    if sampler["class_type"] != "MiniMaxH3ProgressiveSamplerEXPT8":
        raise ValueError("Only progressive first-sampler recipes are exported here")
    task = sampler["inputs"]["task"]
    if task not in {"t2va", "i2va"} or graph["6"]["inputs"]["task_type"].lower() != task:
        raise ValueError("Conditioning and sampler tasks must agree")
    if (graph["7"]["inputs"]["sampler_name"] != "euler" or
            graph["7"]["inputs"]["scheduler"] != "native_flow"):
        raise ValueError("Frontend candidate must use the qualified native Euler setup")
    title = f"H3 Progressive {task.upper()} / 渐进首采 EXP"
    workflow = convert(graph, object_info, title)
    normalize_native_widget_inputs(workflow)
    labels = {
        "1": "视频 VAE", "2": "音频 VAE", "3": "H3 文本编码器",
        "4": "原始 H3 底模（不接 VDN / FAST / SPEED）", "5": "新版 Turbo EMA B",
        "6": "目标画幅与音视频条件", "7": "原生 Euler · 完整 8 步时间表",
        "10": "渐进首采：小画幅 6 步 → 学习放大 → 大画幅 2 步",
        "11": "原生视频 + 音频解码", "12": "按时长裁剪（画音同步）",
        "13": "空负面条件（本图 CFG=1）", "18": "保存原生音视频",
        "19": "保存检查报告", "20": "首帧参考图（请换成你的图片）",
        "21": "采样次数 / 尺寸 / 放大模型报告", "90": "提示词", "91": "时长（生成与保存共用）",
    }
    for source_id, node in zip(graph, workflow["nodes"]):
        node["title"] = labels.get(source_id, node["title"])
        if source_id == "90":
            node.update(pos=[0, -820], size=[640, 350])
        elif source_id == "91":
            node.update(pos=[680, -820], size=[390, 350])
    note_id = workflow["last_node_id"] + 1
    note = (
        f"# H3 渐进首采 · {task.upper()} · 1.77.0 EXP\n\n"
        "用途：先以小画幅完成6步，再用学习型潜空间放大器放大，完成剩下2步。总计仍8步。"
        "不是成片超分，不是VDN的8+4二采。\n\n"
        "日常改动：提示词、时长、目标宽高、seed。首期只验证本图1024×512约3秒，"
        "更长或不同画幅仍需验证；不要把单组约35%耗时下降当普遍承诺。\n\n"
        "必须保留：Setup使用euler/native_flow；sampler、sigmas与MODEL来自同一Setup。"
        "渐进节点low_evaluations小于总步数，task与Conditioning一致。"
        "不要接已采样latent或遮罩。声音继续联合生成，不是锁原音轨。\n\n"
        "模型：models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors；"
        "复用已有模型，无需新转换，也不需要安装SelfLift插件。另需原H3底模、EMA B、文本模型和双VAE。"
        "没有额外自动下载或依赖安装。\n\n"
        "只限当前Core原生H3路径；本图不叠加VDN/FAST/SPEED、区域/尾帧/多参考，"
        "不启用分块或像素锚。失败请回到独立原生8步工作流，不要改动旧图默认设置。\n\n"
        + ("I2VA：更换首帧后保持task=i2va；当前测试无对白，不据此验口型。"
           if task == "i2va" else "T2VA：task=t2va；本图包含音乐和一句对白，请检查声音与口型。")
    )
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "title": "怎么用 / 使用范围",
        "pos": [1120, -820], "size": [740, 700], "flags": {}, "order": len(graph), "mode": 0,
        "inputs": [], "outputs": [], "properties": {}, "widgets_values": [note]})
    workflow["last_node_id"] = note_id
    workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8:progressive-first-sampling:{task}:candidate"))
    workflow["extra"]["progressive_delivery_status"] = "exp_release_1_77_0_short_clips_only_audio_limits_documented"
    return workflow
