"""Export four independent EXP drafts from captured live Comfy schemas.

Does not start a server, queue inference or install/promote user workflows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import build_vdn_two_pass_workflows as builder
    from . import run_openvdn_h3_multimodal_validation as multimodal
except ImportError:
    import build_vdn_two_pass_workflows as builder
    import run_openvdn_h3_multimodal_validation as multimodal


def candidate_graphs():
    result = {}
    for variant in ("t2va", "i2va"):
        for backend in ("vdn", "native_h3"):
            args = builder.base._parser().parse_args([])
            args.width, args.height, args.frame_count = 512, 256, 73
            args.variant, args.refine_backend = variant, backend
            args.two_pass, args.scale_by, args.refine_steps = True, 2., 4
            args.attention, args.video_save = "default", "isolated"
            args.image = multimodal.DEFAULT_IMAGE if variant == "i2va" else None
            args.native_refine_lora = "minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors"
            name = f"H3_OpenVDN_{variant.upper()}_{backend}_TwoPass_EXP"
            graph = builder.build_prompt(args, name)
            graph["18"]["inputs"]["filename_prefix"] = f"MiniMaxH3/VDN_TwoPass/{name}"
            graph["73"]["inputs"]["filename_prefix"] = f"MiniMaxH3/VDN_TwoPass/{name}_FirstPass"
            graph["90"] = builder.node("PrimitiveStringMultiline", value=graph["6"]["inputs"]["prompt"])
            graph["91"] = builder.node(
                "MiniMaxH3DurationPlannerT8", scene_start_seconds=0.,
                scene_duration_seconds=args.frame_count / 24., warmup_seconds=0., cooldown_seconds=0.,
                ensure_minimum_context=False, source_duration_seconds=0.)
            for node_id in ("6", "61"):
                graph[node_id]["inputs"].update(prompt=["90", 0], length=["91", 0])
            for node_id in ("12", "71"):
                graph[node_id]["inputs"]["duration_seconds"] = ["91", 1]
            result[name] = (graph, backend)
    return result


def export_candidates(schema_paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("workflow draft directory must be new")
    info, provenance = {}, []
    for path in map(Path, schema_paths):
        raw = path.read_bytes()
        captured = json.loads(raw)
        if not isinstance(captured, dict):
            raise ValueError("object_info must be a captured schema mapping")
        info.update(captured)
        provenance.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    candidates = {}
    for name, (graph, backend) in candidate_graphs().items():
        missing = {node["class_type"] for node in graph.values()} - info.keys()
        if missing:
            raise ValueError(f"missing captured schemas: {sorted(missing)}")
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                    outputs = info[graph[str(value[0])]["class_type"]]["output"]
                    if not isinstance(value[1], int) or not 0 <= value[1] < len(outputs):
                        raise ValueError("invalid source output slot in candidate graph")
        workflow = builder.build_frontend(graph, info, refine_backend=backend)
        for source_id, node_data in zip(graph, workflow["nodes"]):
            if source_id == "90":
                node_data.update(title="提示词（两次采样共用） / Shared prompt", pos=[0, -1120], size=[650, 400])
            elif source_id == "91":
                node_data.update(title="时长（生成与保存同步） / Shared duration", pos=[700, -1120], size=[350, 400])
        workflow["nodes"][-1]["widgets_values"][0] += (
            "\n\n提示词只在 Shared prompt 改一次；时长只在 Shared duration 改一次，两次采样和两份保存自动同步。"
            "时长按 H3 的 17n+5 帧网格向上对齐；本图从第0秒生成、无额外预热/尾段。"
            "任务类型是本图固定的 T2VA 或 I2VA，不要只改一个 Conditioning 的任务类型。")
        workflow.setdefault("extra", {})["vdn_delivery_status"] = "draft_pending_browser_roundtrip_and_human_review"
        candidates[name] = {"prompt": graph, "workflow": workflow}
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, candidate in candidates.items():
        for kind, data in candidate.items():
            path = output / f"{name}.{kind}.json"
            raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            path.write_bytes(raw)
            files[path.name] = hashlib.sha256(raw).hexdigest()
    manifest = {"status": "drafts_only", "schema_sources": provenance, "files": files,
                "routes": list(candidates), "installed": False, "gpu_quality_accepted": False,
                "browser_roundtrip_verified": False,
                "note": "Captured schema conversion is not browser import or generation validation."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info", nargs="+", type=Path, required=True,
                        help="Captured schemas in oldest-to-newest order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidates(args.object_info, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
