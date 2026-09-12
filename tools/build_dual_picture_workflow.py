"""Promote only the accepted 8s recipe; no diagnostic runtime dependency."""

from copy import deepcopy
import argparse
import json
from pathlib import Path
import uuid

from tools.api_to_frontend_workflow import convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs
from tools.audit_progressive_workflows import audit_candidate

ROOT = Path(__file__).resolve().parents[1]
NODE = "MiniMaxH3DualModelLongVideoEXPT8"
DEST = (
    ROOT
    / "examples/workflows/04-long-video/2026-09-13_H3_Dual_4plus4_Accepted_Picture_KJ.json"
)


def recipe():
    graph = json.loads(
        (ROOT / "tests/fixtures/dual_picture_accepted_api.json").read_text(
            encoding="utf-8"
        )
    )
    graph["8"]["inputs"].update(
        low_context_source="accepted_picture_low_context_v1",
        chain_id="h3_accepted_picture_8s_20260913",
        filename_prefix="H3_Accepted_Picture_4plus4",
    )
    # Preserve the reviewed zero-strength external slots, including filenames:
    # the compatibility loader still reads files/metadata even at strength 0.
    for key in ("30", "31"):
        assert graph[key]["inputs"]["strength_model"] == 0.0
    return graph


def build(info):
    info = deepcopy(info)
    # Stored live object_info predates this append-only optional input.
    info[NODE]["input"].setdefault("optional", {})["low_context_source"] = [
        ["independent_low_x0", "accepted_picture_low_context_v1"],
        {"default": "independent_low_x0"},
    ]
    order = info[NODE].get("input_order", {}).get("optional")
    if order is not None and "low_context_source" not in order:
        order.append("low_context_source")
    graph = recipe()
    workflow = convert(graph, info, "H3 4+4 · 已验收画面上下文 · KJ · 0.4MP / 8秒")
    normalize_native_widget_inputs(workflow)
    positions = {
        "1": [0, 0],
        "2": [480, 0],
        "3": [480, 380],
        "30": [960, 0],
        "31": [960, 380],
        "21": [1440, 0],
        "22": [1440, 380],
        "32": [960, 190],
        "33": [960, 570],
        "4": [0, 300],
        "5": [0, 570],
        "6": [0, 790],
        "7": [480, 850],
        "8": [1920, 0],
    }
    titles = {
        "1": "H3 原生底模（共享）",
        "2": "一采 EMA B · 4步",
        "3": "二采 EMA B · 4步",
        "30": "一采额外 LoRA · 默认0关闭",
        "31": "二采额外 LoRA · 默认0关闭",
        "21": "一采 KJ Memory Sage",
        "22": "二采 KJ Memory Sage",
        "32": "一采额外 LoRA 报告",
        "33": "二采额外 LoRA 报告",
        "4": "H3 Qwen CLIP",
        "5": "原生视频 VAE",
        "6": "原生音频 VAE",
        "7": "全片 Relay：全局场景 + 局部单次对白",
        "8": "通过示例 · 4 + 3D latent upscale + 4",
    }
    for key, node in zip(graph, workflow["nodes"]):
        node.update(pos=positions[key], title=titles[key])
        if key == "7":
            node["size"] = [1250, 900]
        elif key == "8":
            node["size"] = [850, 1870]
        elif key in ("32", "33"):
            node["size"] = [390, 150]
    note = (
        "# 2026-09-13 用户完整审片通过\n\n"
        "本例：448×224 → 896×448（约0.4MP）；8秒192帧24fps；window124/context22；每段4+4；"
        "接缝约5.17秒。并非两段各4秒。指定样片无跳切/虚影/声音异常，不保证任意新素材。\n\n"
        "关键：low_context_source=accepted_picture_low_context_v1。下一段一采使用上段实际成片末39帧，"
        "缩小后视频VAE编码；context22仍只选所需7个latent单元，不是强制39上下文。"
        "仅换LOW视频参考，HIGH条件、音频和4+4步数不改。每个续段多一次短VAE编码，不增加扩散步。"
        "video_context_mode=high_native_mask_exp保留二采已知区域锁定。\n\n"
        "两路 EMA B 强度1；额外LoRA槽默认0，不增加权重效果，但加载器仍读取所选文件。"
        "保留已验收的 minimax_h3_turbo_4步加速_comfyui.safetensors 文件名；缺文件需自行选择兼容文件。"
        "改额外LoRA属于新配置，需重新审片。"
        "需安装KJNodes及Sage；不要同时串多个互相覆盖的attention补丁。"
        "upscaler模型放 models/latent_upscale_models/；不是RGB超分。\n\n"
        "4+4时EAV关闭；second_audio_source=auto继续完成联合音频，不锁住半成品声音。"
        "audio_seam_policy=cosine_bridge、bridge_ms=5仅作用于音频，不是视频叠化。\n\n"
        "首次保持resume_existing=false并使用未占用的chain_id。中断后参数不变时改true续跑。"
        "改模型/LoRA/VAE/策略/提示词/尺寸/时长后换chain_id，不搬旧缓存。filename_prefix只是文件前缀。\n\n"
        "提示词：全局只写贯穿场景/人物/衣着/音乐。一次性台词只写在局部对应事件，每行一个。"
        "本例percent时间0-100；不要把秒填进percent。计划193帧覆盖输出192帧，改时长同时改length。\n\n"
        "禁止用生成后latent端点平移、RGB叠化或只看边界差值来掩盖结构问题。"
        "最终策略12秒/39上下文未GPU验收；不要把8秒结果当任意长片保证。"
        "后续升级先读 docs/DUAL_MODEL_SEAM_FIX_20260913.md，并跑其中回归门禁。"
    )
    nid = workflow["last_node_id"] + 1
    workflow["nodes"].append(
        {
            "id": nid,
            "type": "MarkdownNote",
            "title": "正确用法／验收范围／防回归",
            "pos": [2860, 0],
            "size": [850, 1250],
            "flags": {},
            "order": len(graph),
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
            "widgets_values": [note],
        }
    )
    workflow["last_node_id"] = nid
    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "t8:accepted-picture-seam:20260913")
    )
    workflow["extra"]["accepted_video_sha256"] = (
        "3ff583bc817dd845fa288aaf0b16c2d2be24a6ce282f50c1b37a823edc91ad73"
    )
    workflow["extra"]["acceptance_scope"] = (
        "user-reviewed 8s candidate; explicit production integration; not universal quality"
    )
    return workflow, audit_candidate(graph, workflow, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", type=Path, required=True)
    args = parser.parse_args()
    workflow, audit = build(json.loads(args.object_info.read_text(encoding="utf-8")))
    DEST.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"workflow": str(DEST), "audit": audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
