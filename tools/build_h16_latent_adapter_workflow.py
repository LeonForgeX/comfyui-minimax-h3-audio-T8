"""Build a native SaveLatent bridge example; no inference and no model download."""
from copy import deepcopy
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.api_to_frontend_workflow import convert  # noqa: E402
from tools.audit_progressive_workflows import audit_candidate  # noqa: E402

NODE = "MiniMaxH3LTXLatentAdapterEXPT8"
NAME = "H3_to_LTX_Standard_LATENT_EXP"


def graph(latent_name="choose_h3_video.latent", source_directory="", model_directory=""):
    return {
        "1": {"class_type": "LoadLatent", "inputs": {"latent": latent_name}},
        "2": {"class_type": NODE, "inputs": {"h3_latent": ["1", 0],
            "source_frames": 73, "source_fps": 24.0, "frame_policy": "exact",
            "source_directory": source_directory, "model_directory": model_directory,
            "device": "cpu", "precision": "float32", "reference_prefix_latents": 0,
            "normalization": "comfy_normalized"}},
        "3": {"class_type": "SaveLatent", "inputs": {"samples": ["2", 0], "filename_prefix": "latents/T8_H3_to_LTX"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 4]}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["2", 2]}},
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["2", 3]}},
    }


def build(info, recipe=None):
    recipe = deepcopy(recipe if recipe is not None else graph())
    workflow = convert(recipe, info, NAME)
    layout = {
        1: ([0, 0], [300, 180]),
        2: ([380, 0], [640, 500]),
        3: ([1100, 0], [390, 180]),
        4: ([1100, 260], [390, 180]),
        5: ([1100, 520], [390, 180]),
        6: ([1100, 780], [390, 180]),
    }
    for node in workflow["nodes"]:
        node["pos"], node["size"] = deepcopy(layout[node["id"]])
    note = workflow["last_node_id"] + 1
    text = """# H3 → LTX 标准潜空间 · EXP

本图保存转换后的 LTX 视频 LATENT，不是 LTX 精修或成片生成模板。
1. 把原生 SaveLatent 保存的 H3 视频 .latent 放进 ComfyUI/input，选择文件。
   必须带 latent_format_version_0；旧 SD 格式会被 Core 自动缩放，不能冒充 H3。
2. 填写固定 Sana 的 h3_ltx_adapter 源码目录和官方模型目录（config.json + model.safetensors）。
3. source_frames 填实际帧数，不从latent猜时长。73→73可用exact；124需显式pad→129或crop→121。
4. source_fps当前24；保持32像素对齐的原画幅，不拉伸、不自动增加第二个放大器。

也可去掉 LoadLatent，把已有 H3 采样／learned 3D 放大后的联合AV直接接到本节点。
第1输出仅LTX视频latent，可接LTX解码或既有精修链。第2输出原样保留H3 AV。
音频不转换为LTX格式；pad/crop后必须明确处理原音尾部，不能用-shortest掩盖错位。
输出帧数/FPS以本节点实际输出为准；不直接沿用输入帧数。

CPU/F32为初始默认。CUDA/BF16已做固定73帧转换与解码测试，不是整条LTX的16GB/速度/画质保证。
只转换官方已核验权重和源码；缺文件不会影响其他节点导入。详细来源、兼容和限制见
docs/H16_STANDARD_LATENT_ADAPTER_EXP.md。未通过集中人审，禁止作为已验收画质推荐。
"""
    workflow["nodes"].append(dict(id=note, type="MarkdownNote", title="先看这里 / 输入与时间边界",
        pos=[0, 600], size=[1020, 680], flags={}, order=len(recipe), mode=0,
        inputs=[], outputs=[], properties={}, widgets_values=[text]))
    workflow["last_node_id"] = note
    workflow.setdefault("extra", {})["t8_h16_status"] = "development_exp_human_pending"
    return workflow, audit_candidate(recipe, workflow, info)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, audit = build(json.loads(args.object_info.read_text(encoding="utf8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(audit))
