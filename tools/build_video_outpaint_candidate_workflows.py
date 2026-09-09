"""Build separate candidate/review/confirm/continue drafts from live node schemas.

Does not queue, install or change existing workflows. The output directory must
be new. CPU reload workflows deliberately have no MODEL, CLIP or VAE loaders.
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
    from .build_video_outpaint_workflows import NAMESPACE, STAGES, execution_signature
except ImportError:
    from api_to_frontend_workflow import convert
    from build_video_outpaint_workflows import NAMESPACE, STAGES, execution_signature


PREFIX = "MiniMaxH3VideoOutpaint"
CANDIDATE = PREFIX + "CandidateT8"
RELOAD = PREFIX + "LoadPreparedT8"
READ = PREFIX + "LoadCandidateT8"
SELECT = PREFIX + "SelectCandidateT8"
RESTORE = PREFIX + "LoadSelectionT8"
COMPLETED = PREFIX + "LoadCompletedSelectionT8"
CONTINUE = PREFIX + "ContinueCandidateT8"
COMPOSE = PREFIX + "ComposeCandidateT8"
GUIDANCE = PREFIX + "GuidanceT8"
GUIDED_PREPARE = PREFIX + "PrepareGuidedT8"
REGIONAL_MODEL = PREFIX + "RegionalModelT8"
DLSS_AUDIT = "MiniMaxH3DLSSNRRuntimeAuditT8Advanced"
DLSS_VIDEO = "MiniMaxH3DLSSNRVideoFileT8Advanced"
COMPATIBILITY = PREFIX + "CompatibilityAuditT8"
ROUTES = ("01_Generate_Candidate", "02_Review_Candidate", "03_Confirm_Candidate", "04_Continue_Selection",
          "05_Save_Completed_Selection", "06_Generate_Guided_Candidate", "07_Continue_Guided_Selection",
          "08_Save_Completed_Selection_DLSS_2x", "09_Model_Compatibility_Audit")
COMMON_NOTE = (
    "隔离测试草稿，尚未正式安装/发布。\n\n"
    "普通路线使用01→02→03→04（失败重存可用05）；区域引导路线使用06→02→03→07（失败重存仍可用05）。"
    "同一路线必须使用同一原片、Plan 参数、run_name、candidate_name，且连接同一 ComfyUI 输出缓存目录。"
    "只换 seed/步数/采样模型时请用新候选名；改变原片、扩边、提示词、VAE 或文本编码器时，"
    "请用新 run_name 重新准备缓存。"
    "候选和确认编号是完整的 64 位文本，请复制全部字符，不能填文件路径。\n\n"
    "读取缓存不重新编码提示词，不会静默补建丢失文件。缓存丢失或身份不符会报错。"
    "候选图只用于选定首个窗口，不能代替整片接缝、动作和声音验收。\n\n"
    "生成使用 Stock20 / 原生噪声和 KJ 显存补丁，未叠加 Turbo。默认joint_decode整幅联合解码，"
    "原片会经过VAE重建。可选preserve_source精确保留原片，但接缝可能更明显。"
    "Color Match只在preserve_source模式处理扩区；接续与保存继承候选模式，原片声音保留。"
)
NOTES = {
    ROUTES[0]: "生成只跑首个窗口，显示真实生成首帧后正常暂停，不接着跑整片。"
               "复制右侧 candidate_id 并记下 run_name/candidate_name，再打开 02 审图。"
               "不喜欢可换新 candidate_name 和 seed 重试；已接续的候选请用 02 读取，不要重跑本图。",
    ROUTES[1]: "填入原先的 run_name、candidate_name 和 candidate_id，运行即可读取原候选图。"
               "本图不加载生成模型、不采样、不确认，也不保存成片。喜欢后再打开 03。",
    ROUTES[2]: "先在 02 看过该 candidate_id 的图片，再填写同样参数，并手动打开 confirm_selection。"
               "默认关闭，未确认时运行会提示停止。确认只保存选择记录，不启动长视频生成。"
               "复制右侧 selection_id，下一步填到 04。",
    ROUTES[3]: "填入之前确认所得的 selection_id，并使用候选生成时的模型和视频 VAE。"
               "本图会真正接续生成整片；只想看图请用 02。"
               "复用所选首个窗口，已完成窗口不会重抽，取消后可按同样参数继续。"
               "颜色设置沿用候选，编码前核对首帧 RGB；不符不发布成片。Compose 已保存视频，无需再接 SaveVideo。",
    ROUTES[4]: "仅在 04 已完成全部窗口、但保存/编码失败或软件重启后使用。填入同一 selection_id，"
               "本图只加载视频 VAE，验证并读取已完成缓存后重新合成保存；不加载主模型、不采样、不改变 22 个窗口。"
               "如果缓存未完成会明确拒绝，请返回 04 接续。Compose 已保存视频，无需再接 SaveVideo。",
    ROUTES[5]: "区域引导首窗候选：先按输出画布设置regions_json；person_boxes_json使用原片坐标。"
               "示例把天空和地面分别路由到上下扩画区域。人物框只审计原片锁定和边界风险，不保证扩出区域绝不出现人物。"
               "必须使用独立run_name。生成候选后仍到02读取、03确认，再到07接续。",
    ROUTES[6]: "区域引导候选确认后，用和06相同的原片、Plan、run_name、candidate_name和selection_id。"
               "读取区域缓存后重新绑定同一MODEL，再串行接续整片。不能改区域提示、CLIP或模型组合；不支持的LoRA/Attention会明确拒绝。",
    ROUTES[7]: "独立后处理：只接受已经完成的普通或区域扩画selection_id，先无采样重合成，再用现有DLSS-NR文件视频节点做2x。"
               "需要自行安装models/DLSS-NR/1.3外部程序并遵守其许可，不需要新safetensors。新版运行时审计无需额外协议开关，实际一帧自检通过后才输出可用句柄。"
               "超分会改变整幅画面的像素和质感，因此输出不再声称原片区域像素精确保留；请单独人审。",
    ROUTES[8]: "只审计实际加载MODEL与已安装patch/wrapper，不生成视频。Stock20和锁定版本KJ低显存组合可通过；"
               "Turbo/SPEED/SLA/VDN/Fast H3/Prompt Relay在没有单独适配与扩画验收前明确不支持，不会按模型文件名猜测。",
}


def _node(kind, inputs, title=None):
    value = {"class_type": kind, "inputs": inputs}
    if title:
        value["_meta"] = {"title": title}
    return value


def _closure(graph, roots):
    result, visiting = {}, set()

    def visit(key):
        if key in visiting:
            raise ValueError("cyclic workflow")
        if key in result:
            return
        visiting.add(key)
        node = graph[key]
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph and isinstance(value[1], int):
                visit(str(value[0]))
        visiting.remove(key)
        result[key] = deepcopy(node)

    for root in roots:
        visit(root)
    return result


def _frontend(graph, info, route):
    missing = {n["class_type"] for n in graph.values()} - set(info)
    if missing:
        raise ValueError(f"server is missing candidate workflow schemas: {sorted(missing)}")
    workflow = convert(graph, info, f"H3 Video Outpaint · {route} · Draft")
    workflow["id"] = str(uuid.uuid5(NAMESPACE, "candidate-v1-" + route))
    for node in workflow["nodes"]:
        kind = node["type"]
        if kind.startswith(PREFIX):
            node["properties"]["cnr_id"] = "minimax-h3-audio-t8"
        elif "KJ" in (info[kind].get("python_module") or ""):
            node["properties"].pop("cnr_id", None)
        node["size"] = [410, 320]
        if kind == STAGES[0]:
            node["size"] = [410, 510]
        elif kind in (CANDIDATE, READ, RESTORE, COMPLETED):
            node["size"] = [620, 750]
        elif kind == "PreviewAny":
            node["size"] = [680, 260]
        if kind == CANDIDATE:
            # The installed frontend adds this widget for an input named seed,
            # even when the V3 schema omits control_after_generate. Keep the
            # converter's fixed control: removing it shifts steps/resume/color.
            values = node["widgets_values"]
            if (len(values) != 8 or not isinstance(values[0], str) or values[1:] !=
                    [20260808, "fixed", 20, False, True, False, "joint_decode"]):
                raise ValueError(f"unexpected candidate seed widget layout: {values!r}")
    # Preserve graph topology while making image/text widgets large enough.
    columns = {}
    for node in workflow["nodes"]:
        x = node["pos"][0]
        columns.setdefault(x, []).append(node)
    cursor = 0
    for x in sorted(columns):
        column = columns[x]
        y = 0
        for node in column:
            node["pos"] = [cursor, y]
            y += node["size"][1] + 60
        cursor += max(n["size"][0] for n in column) + 60
    note_id = workflow["last_node_id"] + 1
    workflow["nodes"].append({"id": note_id, "type": "MarkdownNote", "pos": [0, -440],
        "size": [1100, 400], "flags": {}, "order": len(workflow["nodes"]), "mode": 0,
        "inputs": [], "outputs": [], "properties": {},
        "title": route + " · 操作顺序", "widgets_values": [NOTES[route] + "\n\n" + COMMON_NOTE]})
    workflow["last_node_id"] = note_id
    workflow["extra"].update(ds={"scale": 0.65, "offset": [40, 480]},
        outpaint_delivery_status="candidate_workflow_draft_not_registered_not_human_accepted")
    execution_signature(workflow)
    return workflow


def build_candidate_workflows(prompt, object_info):
    graph = deepcopy(prompt)
    ids = {}
    for stage in STAGES:
        matches = [key for key, n in graph.items() if n["class_type"] == stage]
        if len(matches) != 1:
            raise ValueError("needs exactly one of each outpaint stage")
        ids[stage] = matches[0]
    if any(n["class_type"] == "SaveVideo" for n in graph.values()):
        raise ValueError("Compose already saves the authoritative video")
    reserved = {"candidate", "prepared_reload", "read", "select", "restore", "continue", "save", "id_text",
                "completed_restore", "save_completed", "guidance", "guided_prepare", "regional_model",
                "guided_candidate", "guided_id_text", "guided_prepared_reload", "guided_model",
                "guided_restore", "guided_continue", "guided_save"}
    reserved |= {"dlss_runtime", "dlss_completed_restore", "dlss_compose", "dlss_video"}
    reserved.add("compatibility")
    if set(graph) & reserved:
        raise ValueError("source graph uses reserved draft node IDs")
    plan_id, prepare_id, sample_id, compose_id = (ids[stage] for stage in STAGES)
    plan, prepare = graph[plan_id]["inputs"], graph[prepare_id]["inputs"]
    sample, compose = graph[sample_id]["inputs"], graph[compose_id]["inputs"]
    plan.update(aspect="custom", left=0, right=0, top=96, bottom=96,
        generation_megapixels=0.5, window_frames="56", cut_frames_json="[]", anchor_x=0.5, anchor_y=0.5)
    prepare.update(run_name="outpaint_01", resume_audio=False, prompt="", shot_prompts_json="[]")
    for node in graph.values():
        if node["class_type"] == "LoadVideo":
            node["inputs"]["file"] = "choose_your_video.mp4"
    model_link, vae_link = deepcopy(sample["model"]), deepcopy(compose["video_vae"])
    graph["candidate"] = _node(CANDIDATE, {"model": model_link, "prepared": [prepare_id, 0],
        "video_vae": vae_link, "candidate_name": "candidate_01", "seed": 20260808, "steps": 20,
        "resume": False, "color_match": True, "geometry_align": False, "source_mode": "joint_decode"})
    graph["id_text"] = _node("PreviewAny", {"source": ["candidate", 3]}, "candidate_id · 全选复制，下一步填到读取候选")
    routes = {ROUTES[0]: _closure(graph, ["candidate", "id_text"])}
    graph["prepared_reload"] = _node(RELOAD, {"plan": [plan_id, 0], "run_name": "outpaint_01"})
    graph["read"] = _node(READ, {"prepared": ["prepared_reload", 0], "candidate_name": "candidate_01", "candidate_id": ""})
    routes[ROUTES[1]] = _closure(graph, ["read"])
    graph["select"] = _node(SELECT, {"candidate": ["read", 0], "confirm_selection": False})
    graph["id_text"] = _node("PreviewAny", {"source": ["select", 2]}, "selection_id · 全选复制，下一步填到接续工作流")
    routes[ROUTES[2]] = _closure(graph, ["id_text"])
    graph["restore"] = _node(RESTORE, {"prepared": ["prepared_reload", 0], "candidate_name": "candidate_01", "selection_id": ""})
    graph["continue"] = _node(CONTINUE, {"model": model_link, "selected": ["restore", 0]})
    graph["save"] = _node(COMPOSE, {"sampled": ["continue", 0], "video_vae": vae_link, "output_name": "selected_outpaint"})
    routes[ROUTES[3]] = _closure(graph, ["save"])
    graph["completed_restore"] = _node(COMPLETED, {"prepared": ["prepared_reload", 0],
        "candidate_name": "candidate_01", "selection_id": ""})
    graph["save_completed"] = _node(COMPOSE, {"sampled": ["completed_restore", 0],
        "video_vae": vae_link, "output_name": "selected_outpaint_recovered"})
    routes[ROUTES[4]] = _closure(graph, ["save_completed"])
    graph["guidance"] = _node(GUIDANCE, {"plan": [plan_id, 0],
        "regions_json": "[{\"shot\":\"all\",\"region\":\"top\",\"prompt\":\"open sky continuing naturally\"},{\"shot\":\"all\",\"region\":\"bottom\",\"prompt\":\"ground continuing naturally\"}]",
        "person_boxes_json": "[]"})
    guided_inputs = deepcopy(prepare)
    guided_inputs.update({"plan": [plan_id, 0], "guidance": ["guidance", 0],
                          "run_name": "outpaint_guided_01"})
    graph["guided_prepare"] = _node(GUIDED_PREPARE, guided_inputs)
    graph["regional_model"] = _node(REGIONAL_MODEL, {"model": model_link,
        "prepared": ["guided_prepare", 0], "query_chunk_rows": 256})
    graph["guided_candidate"] = _node(CANDIDATE, {"model": ["regional_model", 0],
        "prepared": ["regional_model", 1], "video_vae": vae_link,
        "candidate_name": "guided_candidate_01", "seed": 20260808, "steps": 20,
        "resume": False, "color_match": True, "geometry_align": False, "source_mode": "joint_decode"})
    graph["guided_id_text"] = _node("PreviewAny", {"source": ["guided_candidate", 3]},
                                    "guided candidate_id · 复制到02读取")
    routes[ROUTES[5]] = _closure(graph, ["guided_candidate", "guided_id_text"])
    graph["guided_prepared_reload"] = _node(RELOAD, {"plan": [plan_id, 0],
        "run_name": "outpaint_guided_01"})
    graph["guided_model"] = _node(REGIONAL_MODEL, {"model": model_link,
        "prepared": ["guided_prepared_reload", 0], "query_chunk_rows": 256})
    graph["guided_restore"] = _node(RESTORE, {"prepared": ["guided_model", 1],
        "candidate_name": "guided_candidate_01", "selection_id": ""})
    graph["guided_continue"] = _node(CONTINUE, {"model": ["guided_model", 0],
        "selected": ["guided_restore", 0]})
    graph["guided_save"] = _node(COMPOSE, {"sampled": ["guided_continue", 0],
        "video_vae": vae_link, "output_name": "selected_guided_outpaint"})
    routes[ROUTES[6]] = _closure(graph, ["guided_save"])
    runtime_spec = object_info.get(DLSS_AUDIT, {}).get("input", {}).get("required", {}).get(
        "runtime_version", ["COMBO", {"options": ["1.3"]}])
    versions = runtime_spec[1].get("options", ["1.3"]) if len(runtime_spec) > 1 and isinstance(runtime_spec[1], dict) else ["1.3"]
    runtime_version = versions[-1] if isinstance(versions, list) and versions else "1.3"
    runtime_inputs = object_info.get(DLSS_AUDIT, {}).get("input", {})
    runtime_fields = {key: spec for group in ("required", "optional") for key, spec in runtime_inputs.get(group, {}).items()}
    probe_spec = runtime_fields.get("probe_mode", ["COMBO", {"default": "feature_probe_1_frame"}])
    probe_default = probe_spec[1].get("default", "feature_probe_1_frame") if len(probe_spec) > 1 else "feature_probe_1_frame"
    runtime = {"runtime_version": runtime_version, "probe_mode": probe_default,
               "dxgi_adapter_index": 0, "cuda_device_index": 0}
    if "accept_external_runtime_license" in runtime_fields:
        runtime["accept_external_runtime_license"] = False  # Only for an explicitly older installed schema.
    graph["dlss_runtime"] = _node(DLSS_AUDIT, runtime)
    graph["dlss_completed_restore"] = _node(COMPLETED, {"prepared": ["prepared_reload", 0],
        "candidate_name": "candidate_01", "selection_id": ""})
    graph["dlss_compose"] = _node(COMPOSE, {"sampled": ["dlss_completed_restore", 0],
        "video_vae": vae_link, "output_name": "selected_outpaint_for_dlss"})
    graph["dlss_video"] = _node(DLSS_VIDEO, {"dlss_nr_runtime": ["dlss_runtime", 0],
        "source_video": ["dlss_compose", 0], "mode": "sr_nr", "scale": "2.0",
        "quality_profile": "standard", "sr_preset": "default", "nr_style": "0 Default",
        "nr_preset": "0 Default", "nr_intensity": 1.5, "nr_detail": 1.0, "nr_color": 1.0,
        "nr_skin": -1.0, "nr_local_structure": 1.0, "nr_local_tone": 1.0,
        "nr_global_tone": -1.0, "nr_ui_correction": False, "nr_auto_mask": False,
        "motion_engine": "auto", "filename_prefix": "T8_H3_Outpaint_DLSS/selected_outpaint_2x",
        "crf": 18.0})
    routes[ROUTES[7]] = _closure(graph, ["dlss_video"])
    graph["compatibility"] = _node(COMPATIBILITY, {"model": model_link})
    routes[ROUTES[8]] = _closure(graph, ["compatibility"])
    result = {}
    for route, api in routes.items():
        result[route] = {"prompt": api, "workflow": _frontend(api, object_info, route)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prompt", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8192")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_prompt.read_bytes()
    source = json.loads(raw)
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    result = build_candidate_workflows(source.get("prompt", source), info)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for route, item in result.items():
        for suffix, value in (("DRAFT", item["workflow"]), ("api", item["prompt"])):
            path = args.output_dir / f"H3_Outpaint_{route}_{suffix}.json"
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    used = {n["class_type"] for v in result.values() for n in v["prompt"].values()}
    (args.output_dir / "schema_snapshot.json").write_text(json.dumps({k: info[k] for k in sorted(used)},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps({"source_prompt_sha256": hashlib.sha256(raw).hexdigest(),
        "files": hashes, "status": "built_from_actual_schema_not_browser_validated",
        "registered_or_installed": False, "queued_inference": False}, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
