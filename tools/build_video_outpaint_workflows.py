"""Build independent frontend drafts from the actual isolated probe graph/schema.

Does not queue inference, register nodes, install files or modify old workflows.
The output directory must be new. Promotion/mirroring remains a separate step.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid

try:
    from .api_to_frontend_workflow import convert
except ImportError:
    from api_to_frontend_workflow import convert


STAGES = [f"MiniMaxH3VideoOutpaint{stage}T8" for stage in ("Plan", "Prepare", "Sample", "Compose")]
NAMESPACE = uuid.UUID("fe3f9af9-106a-4a64-987e-54133ad1cb6b")
PREVIEW_NODE = "MiniMaxH3VideoOutpaintGeometryPreviewT8"
PREVIEW_TYPES = ("LoadVideo", STAGES[0], PREVIEW_NODE)
NOTE = """Video Outpaint · 独立草稿 / Independent draft

先在 Load Video 上传原片，再选择本机 H3 模型、文本编码器及视频/音频 VAE。
此示例：上下各扩 96px，原片不裁切；custom 才使用四边数值。
0.5MP 是内部生成预算，不是超分；输出尺寸与原片+扩边保持一致。
56 帧窗口依次处理整条输入，不是只输出56帧。原声音原包保留，不生成新配音。
默认 source_mode=joint_decode，整幅联合解码；原片也会经过VAE重建，原音频不变。
可选 preserve_source 精确保留原片像素，但接缝可能更明显。
Color Match 只在 preserve_source 模式调整扩区；joint_decode 跳过接缝修色/几何校正。

续跑：保持原片、模型、提示词、seed、扩边等设置不变，使用同一 run_name，
开启 Prepare.resume_audio 和 Sample.resume。修改生成设置请换 run_name。
只改 Color Match 时可复用完整采样缓存。已有成片不会覆盖，会另存新编号。

当前需 ComfyUI-KJNodes 的 H3 Memory Efficient Attention + FFN；未叠加 Turbo LoRA。
16GB测试使用启动参数 --reserve-vram 5；这是测试设置，不保证所有素材不爆显存。
Compose 本身保存视频，VIDEO输出可接后处理；无需再接 SaveVideo 重复编码。
这是未发布草稿，仅隔离测试环境已注册；正式安装、预览和完整人审还需完成。
折叠版：展开左侧模型节点可换模型；进入扩画子图可改镜头切点、逐镜提示词、锚点和步数。
"""


def build_workflow(prompt, object_info):
    graph = deepcopy(prompt)
    by_type = {}
    for key, node in graph.items():
        by_type.setdefault(node["class_type"], []).append(key)
    if any(len(by_type.get(stage, [])) != 1 for stage in STAGES):
        raise ValueError("needs exactly one of each outpaint stage")
    if any(node["class_type"] == "SaveVideo" for node in graph.values()):
        raise ValueError("outpaint Compose already saves the authoritative video")
    missing = {node["class_type"] for node in graph.values()} - set(object_info)
    if missing:
        raise ValueError(f"server is missing workflow schemas: {sorted(missing)}")
    plan, prepare, sample, compose = [graph[by_type[stage][0]]["inputs"] for stage in STAGES]
    plan.update(aspect="custom", left=0, right=0, top=96, bottom=96,
                generation_megapixels=0.5, window_frames="56", cut_frames_json="[]")
    prepare.update(run_name="outpaint_01", resume_audio=False, prompt="", shot_prompts_json="[]")
    sample.update(seed=20260808, steps=20, resume=False, noise_algorithm="t8.outpaint.native_cpu_noise/v1")
    compose.update(output_name="outpaint", color_match=True, geometry_align=False, source_mode="joint_decode")
    for node in graph.values():
        if node["class_type"] == "LoadVideo":
            node["inputs"]["file"] = "choose_your_video.mp4"
    workflow = convert(graph, object_info, "H3 Video Outpaint · Four Stages · Draft")
    workflow["id"] = str(uuid.uuid5(NAMESPACE, "four-stages-v1"))
    stages = []
    others = []
    for node in workflow["nodes"]:
        if node["type"] in STAGES:
            stages.append(node)
        else:
            others.append(node)
        if node["type"].startswith("MiniMaxH3VideoOutpaint"):
            node["properties"]["cnr_id"] = "minimax-h3-audio-t8"
        elif "KJ" in object_info[node["type"]].get("python_module", ""):
            node["properties"].pop("cnr_id", None)
    for index, node in enumerate(sorted(stages, key=lambda n: STAGES.index(n["type"]))):
        node.update(pos=[index*450, 750], size=[410, 650])
    for index, node in enumerate(others):
        node.update(pos=[index % 4 * 450, index // 4 * 300], size=[410, 260])
    note_id = workflow["last_node_id"]+1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "pos": [1820, 0], "size": [630, 1000],
        "flags": {}, "order": len(workflow["nodes"]), "mode": 0, "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [NOTE], "title": "使用说明 / Read first"})
    workflow["last_node_id"] = note_id
    workflow["extra"].update(ds={"scale": 0.48, "offset": [70, 70]},
        outpaint_delivery_status="draft_not_registered_not_human_accepted")
    return workflow


def build_preview_workflow(object_info):
    """A separate output graph with no sampler/model loader, not a pause branch."""
    missing = set(PREVIEW_TYPES) - set(object_info)
    if missing:
        raise ValueError(f"server is missing preview schemas: {sorted(missing)}")
    if not object_info[PREVIEW_NODE].get("output_node"):
        raise ValueError("geometry preview must be an executable output node")
    graph = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "choose_your_video.mp4"}},
        "2": {"class_type": STAGES[0], "inputs": {
            "source_video": ["1", 0], "aspect": "custom", "left": 0, "top": 96, "right": 0, "bottom": 96,
            "anchor_x": 0.5, "anchor_y": 0.5, "generation_megapixels": 0.5, "window_frames": "56",
            "cut_frames_json": "[]"}},
        "3": {"class_type": PREVIEW_NODE, "inputs": {"plan": ["2", 0], "frame_index": 0, "preview_max_edge": 768}},
    }
    workflow = convert(graph, object_info, "H3 Video Outpaint · 扩画范围预览 · Draft")
    workflow["id"] = str(uuid.uuid5(NAMESPACE, "geometry-preview-v1"))
    for node in workflow["nodes"]:
        node["pos"] = [(node["id"] - 1) * 470, 0]
        node["size"] = [430, 650 if node["type"] == STAGES[0] else 430]
        if node["type"].startswith("MiniMaxH3VideoOutpaint"):
            node["properties"]["cnr_id"] = "minimax-h3-audio-t8"
        if node["type"] == PREVIEW_NODE:
            node["size"] = [800, 960]
    workflow["nodes"].append({
        "id": 4, "type": "MarkdownNote", "pos": [0, 700], "size": [870, 330],
        "flags": {}, "order": 3, "mode": 0, "inputs": [], "outputs": [], "properties": {},
        "title": "先看范围，再开生成", "widgets_values": [
            "# 扩画范围预览（不生成）\n\n"
            "上传原片，调整 Plan 的四边像素或目标比例，然后点击运行。\n\n"
            "手动扩边使用 custom；选择固定比例时先把四边输入归零，原画面位置由 anchor_x/anchor_y 控制。\n\n"
            "照片是原片，蓝色棋盘是准备补出的区域，**不是扩画效果**。frame_index 从 0 开始；"
            "preview_max_edge 只控制预览大小，不影响成片或模型分辨率。\n\n"
            "确认范围后，把相同原片和 Plan 参数用于独立的四阶段或折叠生成工作流。"
            "此图不加载生成模型，不运行采样，也不保存成片；预览缓存图由 ComfyUI 保存到临时目录。\n\n"
            "这是隔离测试草稿，尚未发布。请勿把本图当成生成首帧的选择或验收。"
        ]})
    workflow["last_node_id"] = 4
    workflow["extra"].update(ds={"scale": 0.7, "offset": [50, 50]},
        outpaint_delivery_status="geometry_preview_draft_not_generated_not_registered")
    return workflow, graph


def build_compact(workflow, object_info):
    """Collapse only the four stages, leaving upload/model controls visible.

    Native subgraph serialization; no new monolithic runtime or hidden generator.
    Every external connection and promoted widget still targets its original node.
    """
    result = deepcopy(workflow)
    inner = [n for n in result["nodes"] if n["type"] in STAGES]
    if len(inner) != 4 or len({n["type"] for n in inner}) != 4:
        raise ValueError("compact workflow needs all four distinct stages")
    nodes = {n["id"]: n for n in inner}
    # Recreate named inner widget sockets in the live schema order. Outer ordinary
    # nodes keep native widget-only serialization from the existing converter.
    target_slots = {}
    for node in inner:
        saved = {i["name"]: i for i in node["inputs"]}
        schema = object_info[node["type"]]
        inputs = []
        for section in ("required", "optional"):
            specs = schema.get("input", {}).get(section, {})
            order = schema.get("input_order", {}).get(section, list(specs))
            for name in order:
                spec = specs[name]
                options = spec[1] if len(spec) > 1 else {}
                widget = not options.get("forceInput") and (isinstance(spec[0], list) or spec[0] in
                    {"INT", "FLOAT", "BOOLEAN", "STRING", "COMBO"})
                item = deepcopy(saved.get(name, {"name": name, "link": None}))
                item["type"] = "COMBO" if isinstance(spec[0], list) else spec[0]
                if widget:
                    item["widget"] = {"name": name}
                if item.get("link") is not None:
                    target_slots[item["link"]] = len(inputs)
                inputs.append(item)
        node["inputs"] = inputs
    internal = []
    outgoing = []
    external = {}
    untouched = []
    for raw in result["links"]:
        link_id, origin, origin_slot, target, target_slot, kind = raw
        if target in nodes:
            link = {"id": link_id, "origin_id": origin, "origin_slot": origin_slot,
                    "target_id": target, "target_slot": target_slots[link_id], "type": kind}
            internal.append(link)
            if origin not in nodes:
                external.setdefault((origin, origin_slot, kind), []).append(link)
        elif origin in nodes:
            outgoing.append(raw)
        else:
            untouched.append(raw)
    if outgoing:
        raise ValueError("compact draft currently expects standalone Compose outputs")
    graph_id = str(uuid.uuid5(NAMESPACE, "compact-four-stages-v1"))
    top_id = max(n["id"] for n in result["nodes"])+1
    public_inputs, top_inputs, proxies = [], [], []
    outer_links = untouched
    def expose(name, kind, links, widget=None):
        slot = len(public_inputs)
        public_inputs.append({"id": str(uuid.uuid5(NAMESPACE, "input:"+name)), "name": name,
            "type": kind, "linkIds": [link["id"] for link in links], "label": name, "pos": [20, 40+slot*20]})
        top_inputs.append({"name": name, "type": kind, "link": None, **({"widget": {"name": name}} if widget else {})})
        for link in links:
            link.update(origin_id=-10, origin_slot=slot)
        if widget:
            proxies.append(widget)
        return slot
    for (origin, origin_slot, kind), links in external.items():
        target = nodes[links[0]["target_id"]]["inputs"][links[0]["target_slot"]]
        slot = expose(target["name"], kind, links)
        outer_links.append([links[0]["id"], origin, origin_slot, top_id, slot, kind])
        top_inputs[slot]["link"] = links[0]["id"]
    next_link = max([link[0] for link in result["links"]]+[0])+1
    promote = {
        STAGES[0]: ["aspect", "left", "top", "right", "bottom", "generation_megapixels", "window_frames"],
        STAGES[1]: ["prompt", "run_name", "resume_audio"],
        STAGES[2]: ["seed", "resume"],
        STAGES[3]: ["source_mode", "color_match", "output_name"],
    }
    for node in sorted(inner, key=lambda n: STAGES.index(n["type"])):
        for name in promote[node["type"]]:
            slot = next(i for i, item in enumerate(node["inputs"]) if item["name"] == name)
            item = node["inputs"][slot]
            if item.get("link") is not None or "widget" not in item:
                raise ValueError(f"cannot promote {node['type']}.{name}")
            link = {"id": next_link, "target_id": node["id"], "target_slot": slot, "type": item["type"]}
            expose(name, item["type"], [link], [str(node["id"]), name])
            item["link"] = next_link
            internal.append(link)
            next_link += 1
    compose = next(n for n in inner if n["type"] == STAGES[3])
    public_outputs, top_outputs = [], []
    for slot, output in enumerate(compose["outputs"]):
        name, kind = output["name"], output["type"]
        public_outputs.append({"id": str(uuid.uuid5(NAMESPACE, "output:"+name)), "name": name,
            "type": kind, "linkIds": [next_link], "pos": [1900, 40+slot*20]})
        top_outputs.append({"name": name, "type": kind, "links": []})
        output["links"] = [next_link]
        internal.append({"id": next_link, "origin_id": compose["id"], "origin_slot": slot,
                         "target_id": -20, "target_slot": slot, "type": kind})
        next_link += 1
    top = {"id": top_id, "type": graph_id, "title": "H3 扩画 · Compact (Draft)", "pos": [480, 0],
        "size": [560, 800], "flags": {}, "order": len(result["nodes"]), "mode": 0,
        "inputs": top_inputs, "outputs": top_outputs, "widgets_values": [],
        "properties": {"proxyWidgets": proxies, "cnr_id": "minimax-h3-audio-t8"}}
    result["nodes"] = [n for n in result["nodes"] if n["id"] not in nodes]+[top]
    setup_row = 0
    for node in result["nodes"]:
        if node["type"] == "LoadVideo":
            node.update(pos=[0, 0], size=[410, 280])
        elif node["type"] == "MarkdownNote":
            node.update(pos=[1100, 0], size=[520, 900])
        elif node["id"] != top_id:
            node["pos"] = [0, 350+setup_row*70]
            node["flags"] = {**node.get("flags", {}), "collapsed": True}
            if node["type"] == "VAELoader":
                node["title"] = "Audio VAE" if "audio" in node["widgets_values"][0] else "Video VAE"
            setup_row += 1
    # Rebuild exterior source socket lists after fanout was grouped into one pin.
    for node in result["nodes"]:
        for output in node["outputs"]:
            output["links"] = []
    outside = {n["id"]: n for n in result["nodes"]}
    for link in outer_links:
        outside[link[1]]["outputs"][link[2]]["links"].append(link[0])
    result.update(id=str(uuid.uuid5(NAMESPACE, "compact-workflow-v1")), last_node_id=top_id,
                  last_link_id=max([link[0] for link in outer_links]+[0]), links=outer_links)
    result["definitions"] = {"subgraphs": [{"id": graph_id, "version": 1, "revision": 0,
        "name": "H3 Video Outpaint · Four Stages", "config": {}, "category": "MiniMax H3 T8/Video Outpaint EXP",
        "state": {"lastGroupId": 0, "lastNodeId": max(nodes), "lastLinkId": next_link-1, "lastRerouteId": 0},
        "inputNode": {"id": -10, "bounding": [-200, 750, 160, 600]},
        "outputNode": {"id": -20, "bounding": [1900, 750, 160, 100]},
        "inputs": public_inputs, "outputs": public_outputs, "widgets": [], "nodes": inner,
        "groups": [], "links": internal, "extra": {}, "description": "同一四阶段链的折叠入口；可进入子图查看。"}]}
    result["extra"]["ds"] = {"scale": 0.65, "offset": [80, 80]}
    return result


def execution_signature(workflow):
    """Validate socket ownership and flatten this one-level native subgraph.

    Compares actual widget values and named dependency sockets, not canvas layout.
    Does not replace importing/exporting the result through a real frontend.
    """
    top = {n["id"]: n for n in workflow["nodes"]}
    definitions = workflow.get("definitions", {}).get("subgraphs", [])
    definition_by_id = {d["id"]: d for d in definitions}
    real = {key: node for key, node in top.items() if node["type"] not in definition_by_id}
    edges = []
    outer = {link[0]: link for link in workflow["links"]}
    if len(outer) != len(workflow["links"]) or len(top) != len(workflow["nodes"]):
        raise ValueError("duplicate workflow node or link")
    for link_id, origin, origin_slot, target, target_slot, _kind in outer.values():
        if (link_id not in (top[origin]["outputs"][origin_slot].get("links") or [])
                or top[target]["inputs"][target_slot].get("link") != link_id):
            raise ValueError("outer workflow link/socket ownership mismatch")
        if top[target]["type"] not in definition_by_id:
            edges.append((origin, origin_slot, target, top[target]["inputs"][target_slot]["name"]))
    for top_node in top.values():
        if top_node["type"] not in definition_by_id:
            continue
        definition = definition_by_id[top_node["type"]]
        children = {n["id"]: n for n in definition["nodes"]}
        if set(children) & set(real):
            raise ValueError("compact draft child IDs overlap exterior IDs")
        real.update(children)
        inner_links = {link["id"]: link for link in definition["links"]}
        if len(inner_links) != len(definition["links"]):
            raise ValueError("duplicate inner link")
        public_names = [i["name"] for i in definition["inputs"]]
        if len(set(public_names)) != len(public_names):
            raise ValueError("ambiguous compact public inputs")
        for link in inner_links.values():
            link_id = link["id"]
            origin, origin_slot = link["origin_id"], link["origin_slot"]
            target, target_slot = link["target_id"], link["target_slot"]
            if origin == -10:
                if link_id not in definition["inputs"][origin_slot]["linkIds"]:
                    raise ValueError("compact input ownership mismatch")
            elif link_id not in (children[origin]["outputs"][origin_slot].get("links") or []):
                raise ValueError("inner origin ownership mismatch")
            if target == -20:
                if link_id not in definition["outputs"][target_slot]["linkIds"]:
                    raise ValueError("compact output ownership mismatch")
                continue
            if children[target]["inputs"][target_slot].get("link") != link_id:
                raise ValueError("inner target ownership mismatch")
            if origin == -10:
                public = top_node["inputs"][origin_slot]
                if "widget" in public:
                    continue  # Default lives on the unchanged real inner widget.
                external = outer[public["link"]]
                origin, origin_slot = external[1:3]
            edges.append((origin, origin_slot, target, children[target]["inputs"][target_slot]["name"]))
        proxies = top_node["properties"]["proxyWidgets"]
        if len(proxies) != sum("widget" in i for i in top_node["inputs"]):
            raise ValueError("proxy widget count mismatch")
        for node_id, name in proxies:
            if not any(i["name"] == name and "widget" in i for i in children[int(node_id)]["inputs"]):
                raise ValueError("proxy widget does not target a real inner control")
    return {"nodes": sorted((key, n["type"], n.get("widgets_values")) for key, n in real.items()
                             if n["type"] != "MarkdownNote"), "edges": sorted(edges)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--preview-only", action="store_true", help="Build a separate source/geometry-only workflow from live schema")
    parser.add_argument("--server", default="http://127.0.0.1:8191")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.preview_only and args.probe_report:
        parser.error("--preview-only does not consume an inference report")
    if not args.preview_only and not args.probe_report:
        parser.error("--probe-report is required for the four-stage inference drafts")
    if args.preview_only:
        with urllib.request.urlopen(args.server.rstrip("/")+"/object_info", timeout=30) as response:
            info = json.load(response)
        workflow, prompt = build_preview_workflow(info)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        path = args.output_dir / "2026-09-06_H3_Video_Outpaint_Geometry_Preview_DRAFT.json"
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        api_path = args.output_dir / "geometry_preview_api.json"
        api_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        selected = {key: info[key] for key in PREVIEW_TYPES}
        (args.output_dir / "schema_snapshot.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (args.output_dir / "provenance.json").write_text(json.dumps({
            "frontend_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "api_sha256": hashlib.sha256(api_path.read_bytes()).hexdigest(),
            "status": "built_from_actual_schema_not_browser_validated",
            "scope": "source_and_geometry_only_no_sampler", "registered_or_installed": False}, indent=2)+"\n", encoding="utf-8")
        print(path)
        return
    raw = args.probe_report.read_bytes()
    report = json.loads(raw)
    with urllib.request.urlopen(args.server.rstrip("/")+"/object_info", timeout=30) as response:
        info = json.load(response)
    selected = {node["class_type"]: info[node["class_type"]] for node in report["prompt"].values()}
    workflow = build_workflow(report["prompt"], selected)
    compact = build_compact(workflow, selected)
    if execution_signature(workflow) != execution_signature(compact):
        raise ValueError("compact workflow changed the expanded execution graph")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "2026-09-06_H3_Video_Outpaint_Four_Stages_DRAFT.json"
    path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    compact_path = args.output_dir / "2026-09-06_H3_Video_Outpaint_Compact_DRAFT.json"
    compact_path.write_text(json.dumps(compact, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "schema_snapshot.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "probe_report_sha256": hashlib.sha256(raw).hexdigest(),
        "frontend_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compact_sha256": hashlib.sha256(compact_path.read_bytes()).hexdigest(),
        "expanded_execution_signatures_equal": True,
        "status": "built_from_actual_schema_not_browser_validated",
        "registered_or_installed": False}, indent=2)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
