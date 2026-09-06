from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = (
    ROOT
    / "examples"
    / "workflows"
    / "06-face-refine"
    / "2026-08-09_H3_Face_Refine_Parity_Advanced_EXP.json"
)
PARITY_API_PATH = ROOT / "tests" / "fixtures" / "api" / "face_refine_parity_advanced_api.json"
WINDOW_API_PATH = ROOT / "tests" / "fixtures" / "api" / "face_refine_window_advanced_api.json"
PATCH_NODE_ID = 32
PATCH_TYPE = "MiniMaxH3FaceRefineSamplerMaskPatchV11T8Advanced"
PATCH_NOTE = (
    "\n\n## H3 FaceRefine v1.1.1采样遮罩修正\n\n"
    "可选实验开关，enabled默认关闭，MODEL和LATENT原样直通，保持旧路线。"
    "2026-09-06盲评用户更喜欢旧路线；时序与接缝持平。"
    "手动开启后视频mask只走采样器、保留帧按当前sigma重加噪，音频锁定条件保留。"
    "开启时必须连接MODEL、去噪后的LATENT和对应报告；mask哈希不匹配或patch冲突时拒绝。"
)


def _properties(node_type: str) -> dict:
    return {
        "cnr_id": "minimax-h3-audio-T8",
        "Node name for S&R": node_type,
    }


def _patch_frontend_node() -> dict:
    return {
        "id": PATCH_NODE_ID,
        "type": PATCH_TYPE,
        "title": "H3 FaceRefine v1.1.1: optional / 默认关闭",
        "pos": [2640, 300],
        "size": [390, 210],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": 21},
            {"name": "av_latent", "type": "LATENT", "link": 44},
            {"name": "denoise_report_json", "type": "STRING", "link": 45},
        ],
        "outputs": [
            {"name": "model", "type": "MODEL", "links": [46]},
            {"name": "av_latent", "type": "LATENT", "links": [28]},
            {"name": "report_json", "type": "STRING", "links": None},
        ],
        "properties": _properties(PATCH_TYPE),
        "widgets_values": [False],
    }


def _rebuild_links(nodes: dict[int, dict], links: list[list]) -> None:
    for node in nodes.values():
        for output in node.get("outputs", []):
            output["links"] = None
        for input_socket in node.get("inputs", []):
            input_socket["link"] = None
    for link in links:
        link_id, source_id, source_slot, target_id, target_slot, _type = link
        output = nodes[int(source_id)]["outputs"][int(source_slot)]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(int(link_id))
        nodes[int(target_id)]["inputs"][int(target_slot)]["link"] = int(link_id)


def build_frontend() -> dict:
    workflow = json.loads(FRONTEND_PATH.read_text(encoding="utf-8"))
    if any(int(node["id"]) == PATCH_NODE_ID for node in workflow["nodes"]):
        for node in workflow["nodes"]:
            if int(node["id"]) == PATCH_NODE_ID:
                template = _patch_frontend_node()
                node["title"] = template["title"]
                node["inputs"] = node["inputs"][:3]
                node["widgets_values"] = [False]
            if int(node["id"]) == 25:
                note = str(node["widgets_values"][0]).split("\n\n## H3 FaceRefine v1.1.1")[0]
                node["widgets_values"] = [note + PATCH_NOTE]
        return workflow
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    for node_id in (15, 18, 19, 20, 21, 22, 23):
        nodes[node_id]["pos"][0] += 880
    nodes[25]["size"][0] += 880
    note = str(nodes[25]["widgets_values"][0])
    note += PATCH_NOTE
    nodes[25]["widgets_values"] = [note]
    nodes[PATCH_NODE_ID] = _patch_frontend_node()

    links = [list(link) for link in workflow["links"]]
    by_id = {int(link[0]): link for link in links}
    by_id[21][3:5] = [PATCH_NODE_ID, 0]
    by_id[28][1:3] = [PATCH_NODE_ID, 1]
    links.extend(
        [
            [44, 13, 0, PATCH_NODE_ID, 1, "LATENT"],
            [45, 13, 1, PATCH_NODE_ID, 2, "STRING"],
            [46, PATCH_NODE_ID, 0, 15, 0, "MODEL"],
        ]
    )
    _rebuild_links(nodes, links)
    old_order = [int(node["id"]) for node in workflow["nodes"]]
    old_order.insert(old_order.index(15), PATCH_NODE_ID)
    workflow["nodes"] = [nodes[node_id] for node_id in old_order]
    for order, node in enumerate(workflow["nodes"]):
        node["order"] = order
    workflow["links"] = sorted(links, key=lambda item: int(item[0]))
    workflow["last_node_id"] = PATCH_NODE_ID
    workflow["last_link_id"] = 46
    return workflow


def _patch_api(path: Path, *, model_node: str, denoise_node: str, guider_node: str, sampler_node: str) -> dict:
    graph = json.loads(path.read_text(encoding="utf-8"))
    graph[str(PATCH_NODE_ID)] = {
        "inputs": {
            "model": [model_node, 0],
            "av_latent": [denoise_node, 0],
            "denoise_report_json": [denoise_node, 1],
            "enabled": False,
        },
        "class_type": PATCH_TYPE,
        "_meta": {
            "title": (
                "H3 FaceRefine v1.1.1 sampler-only video mask and current-sigma re-noise"
            )
        },
    }
    graph[guider_node]["inputs"]["model"] = [str(PATCH_NODE_ID), 0]
    graph[sampler_node]["inputs"]["latent_image"] = [str(PATCH_NODE_ID), 1]
    return graph


def build_parity_api() -> dict:
    return _patch_api(
        PARITY_API_PATH,
        model_node="10",
        denoise_node="13",
        guider_node="15",
        sampler_node="18",
    )


def build_window_api() -> dict:
    return _patch_api(
        WINDOW_API_PATH,
        model_node="12",
        denoise_node="15",
        guider_node="17",
        sampler_node="20",
    )


def main() -> None:
    FRONTEND_PATH.write_text(
        json.dumps(build_frontend(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    PARITY_API_PATH.write_text(
        json.dumps(build_parity_api(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    WINDOW_API_PATH.write_text(
        json.dumps(build_window_api(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(FRONTEND_PATH)
    print(PARITY_API_PATH)
    print(WINDOW_API_PATH)


if __name__ == "__main__":
    main()
