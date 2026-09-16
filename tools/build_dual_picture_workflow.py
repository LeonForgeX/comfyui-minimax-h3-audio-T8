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
T8_MEMORY_DEST = (
    ROOT
    / "examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json"
)
T8_TEMPORAL_COLOR_DEST = (
    ROOT
    / "examples/workflows/04-long-video/2026-09-16_H3_Dual_4plus4_T8_LowVRAM_Temporal_Color_EXP.json"
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


def t8_memory_recipe():
    """Same 8-second picture recipe with the user-preferred T8 h4 paths."""

    graph = recipe()
    graph["21"] = {
        "class_type": "MiniMaxH3LowVRAMAttentionT8Advanced",
        "inputs": {"model": ["30", 0], "head_chunks": 4},
    }
    graph["22"] = {
        "class_type": "MiniMaxH3LowVRAMAttentionT8Advanced",
        "inputs": {"model": ["31", 0], "head_chunks": 4},
    }
    graph["23"] = {
        "class_type": "MiniMaxH3ChunkFeedForwardT8Advanced",
        "inputs": {"model": ["21", 0], "chunks": 2, "seq_threshold": 4096},
    }
    graph["24"] = {
        "class_type": "MiniMaxH3ChunkFeedForwardT8Advanced",
        "inputs": {"model": ["22", 0], "chunks": 2, "seq_threshold": 4096},
    }
    graph["8"]["inputs"].update(
        model_pass1=["23", 0],
        model_pass2=["24", 0],
        chain_id="h3_t8_lowvram_h4c2_ramp_dual_8s_20260916",
        filename_prefix="H3_T8_LowVRAM_H4C2_Ramp_Dual_4plus4",
        video_context_mode="high_native_mask_ramp_exp",
    )
    return graph


def t8_temporal_color_recipe():
    """New chain identity for the optional temporal seam-color stabilizer."""

    graph = t8_memory_recipe()
    graph["8"]["inputs"].update(
        chain_id="h3_t8_lowvram_h4c2_ramp_color_temporal_8s_20260916",
        filename_prefix="H3_T8_LowVRAM_H4C2_Ramp_Color_Temporal_4plus4",
        color_match_mode="bounded_spatial_temporal_exp",
    )
    return graph


def t8_motion_color_recipe():
    graph = t8_temporal_color_recipe()
    graph['8']['inputs'].update(
        chain_id='h3_t8_lowvram_h4c2_ramp_motion_color_8s_20260916',
        filename_prefix='H3_T8_LowVRAM_H4C2_Ramp_Motion_Color_4plus4',
        color_match_mode='bounded_motion_color_exp',
    )
    return graph


def build_t8_memory_workflow():
    """Transform the saved accepted-picture graph without changing the KJ release."""

    workflow = json.loads(DEST.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    runner = nodes[8]
    note = nodes[15]

    for node_id, title in (
        (9, "一采 T8 LowVRAM Attention · 4组"),
        (10, "二采 T8 LowVRAM Attention · 4组"),
    ):
        node = nodes[node_id]
        node.update(
            type="MiniMaxH3LowVRAMAttentionT8Advanced",
            title=title,
            size=[390, 168],
            widgets_values=[4],
        )
        node["outputs"] = [
            {"name": "model", "type": "MODEL", "links": [node_id + 6]},
            {"name": "report_json", "type": "STRING", "links": None},
        ]
        node["properties"] = {
            "cnr_id": "minimax-h3-audio-T8",
            "Node name for S&R": "MiniMaxH3LowVRAMAttentionT8Advanced",
        }

    chunk_nodes = []
    for node_id, low_node_id, input_link, output_link, title, y in (
        (16, 9, 15, 3, "一采 T8 ChunkFFN · 2组/阈值4096", 0),
        (17, 10, 16, 4, "二采 T8 ChunkFFN · 2组/阈值4096", 430),
    ):
        chunk_nodes.append(
            {
                "id": node_id,
                "type": "MiniMaxH3ChunkFeedForwardT8Advanced",
                "title": title,
                "pos": [1900, y],
                "size": [390, 194],
                "flags": {},
                "order": 14 + (node_id - 16),
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": input_link}
                ],
                "outputs": [
                    {"name": "model", "type": "MODEL", "links": [output_link]},
                    {"name": "report_json", "type": "STRING", "links": None},
                ],
                "properties": {
                    "cnr_id": "minimax-h3-audio-T8",
                    "Node name for S&R": "MiniMaxH3ChunkFeedForwardT8Advanced",
                },
                "widgets_values": [2, 4096],
            }
        )

    links[3][1] = 16
    links[4][1] = 17
    workflow["links"].extend(
        [
            [15, 9, 0, 16, 0, "MODEL"],
            [16, 10, 0, 17, 0, "MODEL"],
        ]
    )
    runner.update(
        title="待验收 · 双路 T8 低显存 + HIGH 渐释接缝 · 4+4",
        pos=[2380, 0],
    )
    assert runner["widgets_values"][11] == "h3_accepted_picture_8s_20260913"
    assert runner["widgets_values"][43] == "H3_Accepted_Picture_4plus4"
    assert runner["widgets_values"][49] == "high_native_mask_exp"
    if len(runner["widgets_values"]) == 51:
        runner["widgets_values"].append("bounded_spatial_v2")
    if (len(runner["widgets_values"]) != 52
            or runner["widgets_values"][51] != "bounded_spatial_v2"):
        raise ValueError("Stored workflow has an incompatible Color Match widget layout")
    runner["widgets_values"][11] = "h3_t8_lowvram_h4c2_ramp_dual_8s_20260916"
    runner["widgets_values"][43] = "H3_T8_LowVRAM_H4C2_Ramp_Dual_4plus4"
    runner["widgets_values"][49] = "high_native_mask_ramp_exp"

    note.update(
        title="T8 双路低显存内循环 · 使用与验收边界",
        pos=[3320, 0],
        order=16,
        widgets_values=[
            "# 2026-09-16 T8 双路低显存 + HIGH 渐释接缝候选（待本次人审）\n\n"
            "本例保持原已验收配方：448×224 → 896×448、总长8秒/192帧、window124/context22、"
            "Prompt Relay、4+4采样、accepted_picture_low_context_v1。它实际生成两段，接缝约5.17秒；"
            "只需检查续接、声音和接缝，不做长片。\n\n"
            "两路必须独立：一采额外LoRA → 一采LowVRAM Attention → 一采ChunkFFN → model_pass1；"
            "二采额外LoRA → 二采LowVRAM Attention → 二采ChunkFFN → model_pass2。默认每路"
            "head_chunks=4、FFN chunks=2、seq_threshold=4096。上一组人审明确淘汰"
            "head_chunks=1：其续段接缝明显；head_chunks=4 保留。第二段 LOW 使用上一段实际成片"
            "末39帧重编码；HIGH 先精确锁住7个latent单元，再以0.25→0.5→0.75连续释放3个"
            "latent单元，避免硬边界把背景结构突然切换。音频mask和值完全不改。"
            "改变任一值会改变执行身份并使旧断点失效。\n\n"
            "不要再叠加KJ Memory、KJ Sage、另一套ChunkFFN或未知forward wrapper；SOL/Sage属于"
            "独立attention所有者，未经组合复核不要与本图同时接入。两个MODEL口仍可分别使用不同LoRA。\n\n"
            "首次保持resume_existing=false。中断后仅在模型、LoRA、内存参数、提示词、尺寸和时长"
            "完全不变时才开启续跑；任何变化都换chain_id。4+4保持EAV关闭、second_audio_source=auto。\n\n"
            "本图的同配置8秒渐释候选已完成机械审计，背景楼体／栏杆连续性较硬边界改善，"
            "但仍需人工完整播放并反复看约5.17秒，确认背景、人物、声音和口型后才能改成已验收。"
        ],
    )
    note_index = workflow["nodes"].index(note)
    workflow["nodes"][note_index:note_index] = chunk_nodes
    workflow["last_node_id"] = 17
    workflow["last_link_id"] = 16
    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "t8:accepted-picture-lowvram-dual:20260915")
    )
    workflow["extra"].pop("accepted_video_sha256", None)
    workflow["extra"].update(
        workflow_title="H3 4+4 · 双路 T8 低显存 + HIGH 渐释接缝 · 0.4MP / 8秒",
        acceptance_scope="mechanically-qualified two-segment 8s ramp candidate; human review required",
        t8_memory_paths={
            "pass1": ["extra_lora", "low_vram_attention_h4", "chunk_ffn_c2"],
            "pass2": ["extra_lora", "low_vram_attention_h4", "chunk_ffn_c2"],
        },
        seam_context={
            "low_context_source": "accepted_picture_low_context_v1",
            "high_video_context_mode": "high_native_mask_ramp_exp",
            "high_release_ramp": [0.25, 0.5, 0.75],
            "audio_unchanged": True,
        },
    )
    return workflow


def build_t8_temporal_color_workflow():
    """Append the reviewed CPU-only color stabilizer without rewriting the prior recipe."""

    workflow = build_t8_memory_workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    runner = nodes[8]
    note = nodes[15]
    runner.update(
        title="待验收 · HIGH 渐释 + 时间色彩稳定 · 4+4",
        size=[850, 1870],
    )
    runner["widgets_values"][11] = (
        "h3_t8_lowvram_h4c2_ramp_color_temporal_8s_20260916"
    )
    runner["widgets_values"][43] = (
        "H3_T8_LowVRAM_H4C2_Ramp_Color_Temporal_4plus4"
    )
    runner["widgets_values"][51] = "bounded_spatial_temporal_exp"
    note["widgets_values"] = [
        note["widgets_values"][0]
        + "\n\nColor Match仍开启；本图另外显式选择bounded_spatial_temporal_exp。"
          "它在既有Lab分布与8×5局部修色之后，只对续段开头12帧的低频RGB均值"
          "做有上限的时间中值稳定并渐退，用于压制接缝后的短促偏暗／回亮。"
          "不混帧、不改结构、不动音频。旧图仍使用bounded_spatial_v2；切换模式必须换chain_id。"
    ]
    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "t8:accepted-picture-lowvram-temporal-color:20260916")
    )
    workflow["extra"].update(
        workflow_title="H3 4+4 · 双路 T8 低显存 + HIGH 渐释 + 时间色彩稳定 · 0.4MP / 8秒",
        acceptance_scope=(
            "mechanically-qualified two-segment 8s ramp plus temporal color candidate; "
            "human review required"
        ),
    )
    workflow["extra"]["seam_context"] = {
        **workflow["extra"]["seam_context"],
        "color_match_mode": "bounded_spatial_temporal_exp",
    }
    return workflow


def write_t8_memory_workflow():
    workflow = build_t8_accepted_memory_workflow()
    T8_MEMORY_DEST.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return T8_MEMORY_DEST


def build_t8_motion_color_workflow():
    """Research candidate only; not auto-written or promoted into delivery examples."""
    workflow = build_t8_temporal_color_workflow()
    nodes = {node['id']: node for node in workflow['nodes']}
    nodes[8]['widgets_values'][11] = 'h3_t8_lowvram_h4c2_ramp_motion_color_8s_20260916'
    nodes[8]['widgets_values'][43] = 'H3_T8_LowVRAM_H4C2_Ramp_Motion_Color_4plus4'
    nodes[8]['widgets_values'][51] = 'bounded_motion_color_exp'
    nodes[8]['title'] = '待验收 · HIGH渐释 + 时间稳定 + 局部运动修色 · 4+4'
    nodes[15]['widgets_values'][0] += (
        '\n\n局部运动修色EXP需要OpenCV；只从可信运动对应估计前后帧共同支持的低频颜色异常。'
        '亮度上限0.006、色度上限0.004、RGB上限0.008，余弦渐退。光流只用于估计修色量，'
        '不挪动画面像素、不混帧、不改音频；不可信遮挡／切镜退出。必须使用新chain_id并审片。'
    )
    workflow['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 't8:lowvram-motion-color:20260916'))
    workflow['extra']['workflow_title'] = 'H3 4+4 · T8低显存 + 局部运动修色EXP · 8秒'
    workflow['extra']['acceptance_scope'] = 'motion color research candidate; human review required'
    workflow['extra']['seam_context']['color_match_mode'] = 'bounded_motion_color_exp'
    return workflow


def write_t8_temporal_color_workflow():
    workflow = build_t8_temporal_color_workflow()
    T8_TEMPORAL_COLOR_DEST.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return T8_TEMPORAL_COLOR_DEST


def t8_accepted_portrait_recipe():
    """Exact GPU portrait recipe plus the accepted post-decode C controls, new chain."""
    graph = json.loads((ROOT / 'tests/fixtures/dual_memory_accepted_portrait_api.json').read_text(encoding='utf8'))
    graph['8']['inputs'].update(
        chain_id='h3_t8_h4c2_portrait_motion_color_8s_20260916',
        filename_prefix='H3_T8_H4C2_Portrait_Motion_Color_4plus4',
        color_match_mode='bounded_motion_color_exp',
    )
    return graph


def build_t8_accepted_memory_workflow():
    """Accepted portrait control composition; not a new end-to-end GPU qualification."""
    from tools.package_progressive_candidate import T8_MEMORY_REVIEW_BINDING

    graph = t8_accepted_portrait_recipe()
    info = json.loads((ROOT / 'tests/fixtures/dual_memory_portrait_object_info.json').read_text(encoding='utf8'))
    optional = info[NODE]['input'].setdefault('optional', {})
    optional['color_match_mode'] = [
        ['bounded_spatial_v2', 'bounded_spatial_temporal_exp', 'bounded_motion_color_exp'],
        {'default': 'bounded_spatial_v2'},
    ]
    order = info[NODE].get('input_order', {}).get('optional')
    if order is not None and 'color_match_mode' not in order:
        order.append('color_match_mode')
    workflow = convert(graph, info, 'H3 · 双路T8 h4+c2 · 2:3首帧 · 局部修色 · 4+4/8秒')
    normalize_native_widget_inputs(workflow)
    audit_candidate(graph, workflow, info)
    ids = {key: index + 1 for index, key in enumerate(graph)}
    nodes = {node['id']: node for node in workflow['nodes']}
    for key, pos, size, title in (
        ('1', [0, 0], [390, 170], 'H3原生底模'),
        ('2', [440, 0], [390, 180], '一采EMA B · 强度1'),
        ('3', [440, 340], [390, 180], '二采EMA B · 强度1'),
        ('30', [880, 0], [390, 180], '一采额外LoRA · 默认0'),
        ('31', [880, 340], [390, 180], '二采额外LoRA · 默认0'),
        ('21', [1320, 0], [390, 170], '一采T8 LowVRAM · h4'),
        ('22', [1320, 340], [390, 170], '二采T8 LowVRAM · h4'),
        ('24', [1760, 0], [390, 190], '一采ChunkFFN · c2/4096'),
        ('25', [1760, 340], [390, 190], '二采ChunkFFN · c2/4096'),
        ('4', [0, 280], [390, 180], 'H3 Qwen CLIP'),
        ('5', [0, 530], [390, 140], '视频VAE'),
        ('6', [0, 760], [390, 140], '音频VAE'),
        ('23', [440, 650], [390, 500], '首帧 · 2:3 · 换图保持比例'),
        ('7', [880, 740], [1250, 850], 'Prompt Relay · global/local/时间线'),
        ('8', [2200, 0], [850, 1870], '已通过 · 双路h4+c2 · HIGH渐释＋局部修色'),
    ):
        nodes[ids[key]].update(pos=pos, size=size, title=title)
    text = (
        '# 2026-09-16 指定8秒C样片验收通过\n\n'
        '用户反馈：还是有一点，先这样吧，发布新版本，这个算通过，后续再研究接缝变色。'
        '轻微颜色跳变作为已知限制，不保证任意新素材无接缝。\n\n'
        '首帧1024×1536（2:3）→ LOW256×384 → HIGH512×768，禁止拉伸。'
        '先在Load Image选择自己的首帧；示例参考图不随包分发。换图后同时保持LOW/HIGH与图像比例，换chain_id。'
        '总长8秒192帧24fps、window124/context22，实际接缝约5.17秒，并非两段各4秒。\n\n'
        '两路EMA B强度1；额外LoRA默认0但仍需所选文件存在，可分别更换。每路额外LoRA→'
        'T8 LowVRAM h4→ChunkFFN c2/4096→对应MODEL口；不再叠加KJ同名内存/未知forward补丁。'
        '原learned 3D latent upscaler、4+4、12/3双shift、auto联合音频保持。EAV关闭。\n\n'
        'LOW参考上一段已完成画面；HIGH精确前缀后按0.25→0.50→0.75释放。color_match=true，'
        'color_match_mode=bounded_motion_color_exp：V2→全局时间稳定→运动置信局部修色。'
        '新增局部上限RGB0.008/亮度0.006/色度0.004，12帧余弦渐退；不混帧、不挪动输出像素、音频不改。'
        '此模式需要OpenCV；旧默认bounded_spatial_v2不变。正常舞台灯也可能被误修，换素材需审片。\n\n'
        '验收C为已完成GPU渐释源→离线B→同生产helper局部C的控制组合，不是新工作流完整GPU复跑或逐位复现保证。'
        '首次resume_existing=false；任何模型/LoRA/图像/提示词/模式/尺寸/时长变化都换chain_id，不搬旧缓存。'
        'audio_seam_policy=cosine_bridge、bridge_ms=5只作用音频；不是视频叠化。\n\n'
        'global只写贯穿人物/场景/音乐，一次性歌词仅进对应local事件；三事件percent时间线为'
        '0–43.75、43.75–87.5、87.5–100，length193覆盖192帧输出。改时长同时改时间线与length。\n\n'
        '详见docs/MOTION_COLOR_EXP.md、docs/H3_MEMORY_NODES_EXP.md与接缝防回归文档。'
    )
    nid = workflow['last_node_id'] + 1
    workflow['nodes'].append({'id': nid, 'type': 'MarkdownNote', 'title': '验收范围／参数／已知限制',
                             'pos': [3120, 0], 'size': [900, 1550], 'flags': {},
                             'order': len(graph), 'mode': 0, 'inputs': [], 'outputs': [],
                             'properties': {}, 'widgets_values': [text]})
    workflow['last_node_id'] = nid
    workflow['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 't8:accepted-portrait-motion-color:20260916'))
    workflow['extra'].update(
        acceptance_scope='Accepted bound portrait8s offline C and documented composition of accepted controls; slight seam color change remains.',
        t8_bound_review={'status': 'accepted_in_this_review_scope', 'date': '2026-09-16',
                        'not_universal_quality_claim': True, **T8_MEMORY_REVIEW_BINDING},
        t8_memory_paths={'pass1': ['extra_lora', 'low_vram_attention_h4', 'chunk_ffn_c2'],
                         'pass2': ['extra_lora', 'low_vram_attention_h4', 'chunk_ffn_c2']},
        seam_context={'low_context_source': 'accepted_picture_low_context_v1',
                      'high_video_context_mode': 'high_native_mask_ramp_exp',
                      'high_release_ramp': [.25, .5, .75], 'audio_unchanged': True,
                      'color_match_mode': 'bounded_motion_color_exp'},
        first_frame={'sha256': 'a89496c9bdc8cada29af5951ecf2f1f4f421142ce68ad3801e6ffe35797bf774',
                     'source_size': [1024, 1536], 'not_bundled': True},
        qualification='Original GPU ramp plus accepted offline post-decode C; not a fresh complete GPU workflow run.',
    )
    return workflow


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
    info[NODE]["input"].setdefault("optional", {})["color_match_mode"] = [
        ["bounded_spatial_v2", "bounded_spatial_temporal_exp", "bounded_motion_color_exp"],
        {"default": "bounded_spatial_v2"},
    ]
    if order is not None and "color_match_mode" not in order:
        order.append("color_match_mode")
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
    memory_workflow = write_t8_memory_workflow()
    print(json.dumps({"workflow": str(DEST), "audit": audit,
                      "t8_memory_workflow": str(memory_workflow)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
