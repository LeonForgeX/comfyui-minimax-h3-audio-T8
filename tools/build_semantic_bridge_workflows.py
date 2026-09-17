"""Build five opt-in release examples with scoped review notes; never queue or push."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_semantic_bridge_probe import build_graph  # noqa: E402
from tools.build_dual_model_workflows import build_prompt, defaults, PLAN  # noqa: E402
from tools.build_fast_h3_v2_workflows import selected_frontend_schema  # noqa: E402
from tools.api_to_frontend_workflow import convert  # noqa: E402
from tools.audit_progressive_workflows import audit_candidate  # noqa: E402

MODELS = {
    "original": "t8_compat/MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors",
    "bunny": "t8_compat/BUNNY_H3_ActionLogic_Bridge_V1_T8_Compat.safetensors",
}

DUAL_REVIEW = {
    "status": "accepted_in_this_review_scope",
    "date": "2026-09-17",
    "review_id": "6ace2aecb00e6fc1a01705304547728410b626e1869a13ac82a739349b95e895",
    "media_sha256": "6da418003510e040d2b3ccc7d9961dcb8d6b652395c0b50425f326159348ce9c",
    "scope": "User accepted repaired B: two-segment8s, LOW640x320, HIGH896x448, independent4+4, Original/BUNNY alpha0.10, active Relay. Not arbitrary-input qualification.",
    "not_universal_quality_claim": True,
}


def release_name(name):
    return name.removesuffix("_EXP") + "_Advanced" if "DualIndependent" in name else name

def configuration(variant):
    return {"class_type": "MiniMaxH3SemanticBridgeConfigT8", "inputs": {
        "model_name": MODELS[variant], "enabled": True, "alpha": .10,
        "magnitude_match": "per_token", "token_scope": "all_tokens", "device": "auto", "chunk_tokens": 256}}


def simple(variant, external=False):
    graph, _ = build_graph(variant, 91701)
    for key in ("30", "31", "32", "34", "35"):
        graph.pop(key)
    graph["14"]["inputs"]["av_latent"] = ["13", 0]
    graph["12"]["inputs"]["conditioning"] = ["9", 0]
    graph["33"] = configuration(variant)
    if external:
        graph["9"]["inputs"].pop("semantic_bridge")
        graph["40"] = {"class_type": "MiniMaxH3SemanticBridgeApplyT8", "inputs": {
            "conditioning": ["9", 0], "semantic_bridge": ["33", 0]}}
        graph["12"]["inputs"]["conditioning"] = ["40", 0]
    return graph


def recipes(info):
    original, bunny = simple("original"), simple("bunny", external=True)
    dual = build_prompt(info, relay=True)
    dual["33"], dual["34"] = configuration("original"), configuration("bunny")
    dual["8"]["inputs"].update(semantic_bridge_pass1=["33", 0], semantic_bridge_pass2=["34", 0],
        low_width=640, low_height=320, width=896, height=448, total_duration_seconds=8.,
        chain_id="semantic_bridge_dual_relay_640_exp", low_context_source="accepted_picture_low_context_v1",
        video_context_mode="high_native_mask_ramp_exp", color_match_mode="bounded_motion_color_exp")
    loop = deepcopy(dual)
    loop.pop("3")
    loop.pop("34")
    kind = "MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced"
    allowed = set(info[kind]["input"].get("required", {})) | set(info[kind]["input"].get("optional", {}))
    values = {key: value for key, value in loop["8"]["inputs"].items() if key in allowed}
    values.update(model=["2", 0], steps=8, shift_video=12., shift_audio=3.,
                  sampler_name="dual_clock_euler", scheduler="native_flow", model_id="native_h3_ema_b_exp",
                  semantic_bridge=["33", 0], chain_id="semantic_bridge_single_relay_exp")
    loop["8"] = {"class_type": kind, "inputs": {**defaults(info[kind]), **values}}
    relay = simple("original")
    relay["36"] = {"class_type": PLAN, "inputs": {**defaults(info[PLAN]),
        "global_prompt": "One woman in a blue coat, fixed studio lighting, quiet room, no subtitles.",
        "local_prompts": "She says <d>测试开始</d>.\nShe remains silent and raises her right hand.",
        "length": 73, "timing_mode": "auto_equal", "time_ranges": ""}}
    cond = "MiniMaxH3PromptRelayConditioningT8Advanced"
    relay["9"]["class_type"] = cond
    relay["9"]["inputs"].pop("prompt")
    relay["9"]["inputs"].pop("length")
    relay["9"]["inputs"] = {**defaults(info[cond]), **relay["9"]["inputs"], "model": ["1", 0]}
    relay["9"]["inputs"]["prompt_relay_plan"] = ["36", 0]
    relay["9"]["inputs"]["execution_mode"] = "apply_exp"
    # Controlled example binds the base first; user patches remain allowed with advisories.
    relay["2"]["inputs"]["model"] = ["9", 0]
    relay["10"]["inputs"].update(model=["2", 0], av_latent=["9", 2])
    relay["12"]["inputs"]["conditioning"] = ["9", 1]
    relay["13"]["inputs"]["latent_image"] = ["9", 2]
    return {"H3_SemanticBridge_Internal_EXP": original, "H3_BUNNY_External_EXP": bunny,
            "H3_SemanticBridge_Relay_EXP": relay, "H3_SemanticBridge_SingleLoop_8s_EXP": loop,
            "H3_SemanticBridge_DualIndependent_8s_EXP": dual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads(args.object_info.read_text(encoding="utf8"))
    results = {}
    for name, graph in recipes(info).items():
        selected, schemas = selected_frontend_schema(graph, info)
        released = release_name(name)
        workflow = convert(selected, schemas, released)
        accepted = "DualIndependent" in name
        text = (("Semantic Bridge / BUNNY · 正式发布的指定验收配方\n"
                 "2026-09-17 用户确认修复后B：可以了没问题了。两段共8秒，LOW640×320→HIGH896×448，两个独立MODEL及两阶段Bridge0.10，保留4+放大+4。\n\n"
                 if accepted else "Semantic Bridge / BUNNY · 正式发布的EXP示例\n指定短片测试之外的素材、声音和组合仍需自行对照验证。\n\n")
                + "默认 alpha=0.10，per_token，all_tokens。它不是 LoRA，不需要 SenseNova 教师。\n"
                "原版偏构图/关系；BUNNY 是独立训练的动作语义模型，不是同模型改格式。\n"
                "关闭 enabled 或 alpha=0 为原条件直通。参考音频、唱歌、Ref2VA 和内循环仍属 EXP，声音可能退化。\n"
                "Relay 必须使用条件/循环节点内部 Bridge 插口，不在绑定后重复 Apply。两采可分别接模型配置；"
                "enabled=false 只关闭对应阶段的 Bridge 增强，不关闭采样；双采始终保留两个独立MODEL与4+4。不要叠两次 Bridge。\n"
                "全局写贯穿场景；局部每行一个事件。循环默认两段共8秒，重点看约5.17秒接缝。"
                "改配置换 chain_id。验收双采使用640×320→896×448，保留两份Bridge；"
                "旧448×224戏院测试出现漂浮光斑，关闭Bridge也未消失，不作为画质推荐。"
                "提高一采尺寸的固定配方已验收，不是所有素材的保证。"
                "全局不重复一次性台词，局部按事件写；参考音频、歌唱和Hybrid仍为EXP。\n\n"
                "模型放 ComfyUI/models/semantic_bridge/t8_compat，保留模型相对路径，不是LoRA。\n"
                "https://huggingface.co/t8star/Semantic-Bridge-Comfy\n"
                "https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge\n"
                "https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge\n"
                "节点：https://github.com/T8mars/comfyui-minimax-h3-audio-T8\n"
                "详情：docs/SEMANTIC_BRIDGE_EXP.md")
        note = workflow["last_node_id"] + 1
        workflow["nodes"].append(dict(id=note, type="MarkdownNote", title="使用说明 / 指定验收范围" if accepted else "使用说明 / EXP范围",
            pos=[0, -630], size=[1000, 560], flags={}, order=len(graph), mode=0,
            inputs=[], outputs=[], properties={}, widgets_values=[text]))
        workflow["last_node_id"] = note
        workflow.setdefault("extra", {})["t8_semantic_bridge_status"] = "released_bound_dual_accepted" if accepted else "released_experimental_example"
        if accepted:
            workflow["extra"]["t8_bound_review"] = deepcopy(DUAL_REVIEW)
        results[name] = audit_candidate(selected, workflow, schemas)
        for suffix, content in (("api.json", selected), ("json", workflow)):
            with (args.output / f"2026-09-17_{released}.{suffix}").open("x", encoding="utf8") as stream:
                json.dump(content, stream, ensure_ascii=False, indent=2)
    with (args.output / "audit.json").open("x", encoding="utf8") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
