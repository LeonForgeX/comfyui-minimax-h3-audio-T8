"""Export both tested HyperFlow long-film routes as editable Comfy workflows.

Only reads Core object-info. It never queues a prompt or loads model weights.
The saved example chain IDs must be changed before a new independent run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import uuid

from tools.api_to_frontend_workflow import _get_json, convert
from tools.frontend_workflow_compat import normalize_native_widget_inputs
from tools.build_hyperflow_long_video_workflow import build_prompt as build_dual_prompt
from tools.build_hyperflow_single8_long_video_workflow import build_prompt as build_single_prompt


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
DEST = ROOT / "examples" / "workflows" / "04-long-video"
PROMPT = (
    "A single continuous cinematic camera move through a bright sunlit living "
    "room in clear late-morning daylight. Large windows evenly illuminate the "
    "entire room. Crisp visible details in the sofa fabric, wooden table grain, "
    "green plant leaves and picture frames. Natural realistic colors, stable slow "
    "movement, deep focus, clear bright exposure, quiet natural room tone. "
    "No darkness, no haze, no intentional blur."
)


def local_node_info() -> dict:
    sys.path.insert(0, str(CORE))
    package_name = "h3_t8_hyperflow_workflow_build"
    spec = importlib.util.spec_from_file_location(
        package_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    assert spec.loader is not None
    spec.loader.exec_module(package)
    from h3_t8_hyperflow_workflow_build.hyperflow_long_video_exp.nodes import (
        MiniMaxH3HyperFlowLongVideoEXPT8,
    )
    from h3_t8_hyperflow_workflow_build.hyperflow_long_video_exp.single8_node import (
        MiniMaxH3HyperFlowSingle8LongVideoEXPT8,
    )
    from h3_t8_hyperflow_workflow_build.nodes_h3_lora_compat_advanced import (
        MiniMaxH3LoRACompatibilityLoaderT8Advanced,
    )
    return {
        cls.define_schema().node_id: cls.GET_NODE_INFO_V1()
        for cls in (
            MiniMaxH3HyperFlowLongVideoEXPT8,
            MiniMaxH3HyperFlowSingle8LongVideoEXPT8,
            MiniMaxH3LoRACompatibilityLoaderT8Advanced,
        )
    }


def standard_node_info() -> dict:
    """Read the three unchanged Core loader schemas without a running server."""
    sys.path.insert(0, str(CORE))
    import nodes as core_nodes

    result = {}
    for name in ("UNETLoader", "CLIPLoader", "VAELoader"):
        cls = core_nodes.NODE_CLASS_MAPPINGS[name]
        result[name] = {
            "input": cls.INPUT_TYPES(),
            "output": list(cls.RETURN_TYPES),
            "output_name": list(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)),
            "display_name": name,
            "python_module": "nodes",
        }
    return result


def build_examples(info: dict) -> dict[str, dict]:
    recipes = {
        "2026-09-22_H3_HyperFlow_P7_Dual_0p6MP_8s_EXP.json": (
            build_dual_prompt(info, chain_id="hf_p7_example_change_me", prompt=PROMPT,
                              width=1024, height=576, low_width=512, low_height=288),
            "HyperFlow P7 · LOW 4 → latent 2× → HIGH 4",
            "每段 LOW 512×288 原生0:4，学习型3D latent放大，HIGH 1024×576原生4:8。"
            "这是两段共8秒的T2VA/native实验路线。",
        ),
        "2026-09-22_H3_HyperFlow_Native_Single8_0p6MP_8s_EXP.json": (
            build_single_prompt(info, chain_id="hf_single8_example_change_me",
                                prompt=PROMPT, width=1024, height=576),
            "HyperFlow 原生单次8步 · 无latent放大",
            "每段直接在1024×576运行原生完整0:8训练网格；不使用LOW/HIGH双采或latent放大。"
            "这是两段共8秒的T2VA/native实验路线。",
        ),
    }
    result = {}
    for filename, (graph, title, explanation) in recipes.items():
        workflow = convert(graph, info, title)
        normalize_native_widget_inputs(workflow)
        note_id = workflow["last_node_id"] + 1
        workflow["nodes"].append({
            "id": note_id, "type": "MarkdownNote", "title": "使用前请读",
            "pos": [1900, 20], "size": [600, 360], "flags": {},
            "order": len(graph), "mode": 0, "inputs": [], "outputs": [],
            "properties": {}, "widgets_values": [
                "# HyperFlow 长片 EXP\n\n" + explanation + "\n\n"
                "运行前必须改为新的 chain_id；不要把两条路线的缓存或旧成片混用。"
                "需安装独立 HyperFlow 原始权重、H3 底模/CLIP/VAE；双采路线另需3D latent放大权重。"
                "当前只验 T2VA/native、8秒、window124/context22。请先用禁用 comfy-aimdo"
                "编译器的隔离 Core；完整观看画面与声音，尤其是约5.17秒接缝。"
                "机械解码通过不是画质验收。"
            ],
        })
        workflow["last_node_id"] = note_id
        workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "t8:hf-long-film:" + filename))
        workflow["extra"]["t8_hyperflow_status"] = "experimental_8s_media_pass_human_review_required"
        result[filename] = workflow
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="",
                        help="Optional read-only Core object-info source; offline Core schemas otherwise")
    args = parser.parse_args()
    info = standard_node_info()
    if args.server:
        url = args.server.rstrip("/")
        for name in ("UNETLoader", "CLIPLoader", "VAELoader"):
            info.update(_get_json(f"{url}/object_info/{name}"))
    info.update(local_node_info())
    for filename, workflow in build_examples(info).items():
        path = DEST / filename
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
