"""Create explicit tiny CPU fixtures for real browser archive/confirmation tests.

Not a learned model or a visual-quality sample. No server sampler is patched.
Writes only new fixture assets/workflows inside an explicitly named test root.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8192")
    args = parser.parse_args()
    root = args.test_root.resolve()
    if not all((root / name).is_dir() for name in ("input", "output", "user")):
        raise ValueError("requires an existing isolated test-server root")
    fixture_dir = root / "cpu_archive_fixture"
    fixture_dir.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "tests"))
    sys.path.insert(0, str(project.parents[1]))
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    package = types.ModuleType("h3_audio_t8_pkg")
    package.__path__ = [str(project / "h3_t8"), str(project)]
    sys.modules[package.__name__] = package
    stages = importlib.import_module(package.__name__ + ".nodes_video_outpaint")
    execution = importlib.import_module(package.__name__ + ".video_outpaint_candidate_execution")
    preview = importlib.import_module(package.__name__ + ".video_outpaint_candidate_preview")
    archive = importlib.import_module(package.__name__ + ".video_outpaint_candidate_archive")
    from test_video_outpaint_media import _clip
    from test_video_outpaint_compose import CombinedVAE
    from test_video_outpaint_execution import TinyClip, _model
    from test_video_outpaint_sampling import _backend
    from build_video_outpaint_candidate_workflows import _frontend, _closure, ROUTES
    import folder_paths
    folder_paths.set_output_directory(str(root / "output"))
    source_name = "cpu_archive_fixture_blue_90f.mp4"
    source = _clip(root / "input" / source_name, 32, 32, frames=90, sound=False)
    params = {"aspect": "custom", "left": 32, "top": 0, "right": 32, "bottom": 0,
        "anchor_x": 0.5, "anchor_y": 0.5, "generation_megapixels": 0.5,
        "window_frames": "73", "cut_frames_json": "[]"}
    plan = stages.MiniMaxH3VideoOutpaintPlanT8.execute(source_video=source, **params).result[0]
    prepared = stages.MiniMaxH3VideoOutpaintPrepareT8.execute(plan, TinyClip(), CombinedVAE(),
        "CPU fixture, not visual quality", "[]", "ui_cpu_fixture", 0, 64, False).result[0]
    cache = prepared["root"] / "candidates" / "cpu_fixture_a"
    settings = {"seed": 42, "steps": 20, "noise_algorithm": "t8.outpaint.native_cpu_noise/v1"}
    windows, candidate, _ = execution.sample_verified_first_candidate(model=_model(),
        conditioning=prepared["conditioning"], source_store=prepared["source"], audio=prepared["audio"],
        cache_root=cache, resume=False, sample_function=_backend, **settings)
    image, report = preview.render_candidate_first_frame(vae=CombinedVAE(), candidate=candidate,
        inspection=prepared["inspection"], source_store=prepared["source"], window_store=windows,
        color_match=True)
    handle = {"prepared": prepared, "windows": windows, "candidate": candidate,
        "preview_report": report, "settings": settings, "cache_root": cache}
    candidate_id = archive.save_candidate_archive(handle, image)
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info", timeout=30) as response:
        info = json.load(response)
    workflows = {}
    for route in ROUTES[1:]:
        prompt = json.loads((args.draft_dir / f"H3_Outpaint_{route}_api.json").read_bytes())
        if route == ROUTES[3]:
            prompt = _closure(prompt, ["restore"])
        for node in prompt.values():
            kind, inputs = node["class_type"], node["inputs"]
            if kind == "LoadVideo":
                inputs["file"] = source_name
            elif kind == "MiniMaxH3VideoOutpaintPlanT8":
                inputs.update(params)
            elif kind == "MiniMaxH3VideoOutpaintLoadPreparedT8":
                inputs["run_name"] = "ui_cpu_fixture"
            elif kind.endswith(("LoadCandidateT8", "LoadSelectionT8")):
                inputs["candidate_name"] = "cpu_fixture_a"
                if "candidate_id" in inputs:
                    inputs["candidate_id"] = candidate_id
        workflow = _frontend(prompt, info, route)
        workflow["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "t8-cpu-archive-fixture-" + route))
        note = next(n for n in workflow["nodes"] if n["type"] == "MarkdownNote")
        note["widgets_values"] = ["CPU 接口测试色块，不是学习模型效果图，不做画质验收。\n\n"
            "此测试只验证：已存候选PNG显示 → 手动确认 → 复制selection_id → 重新载入并恢复同图。"
            "不会在服务中加载生成模型、调用VAE或扩画采样。Confirm默认关闭；Restore只有读取节点，没有接续/保存节点。"]
        name = {ROUTES[1]: "CPU_Fixture_Review", ROUTES[2]: "CPU_Fixture_Confirm",
            ROUTES[3]: "CPU_Fixture_Restore_Only"}[route]
        # Bring the actual image / confirmation / text controls into view.
        for node in workflow["nodes"]:
            if node["type"] in ("MiniMaxH3VideoOutpaintLoadCandidateT8", "MiniMaxH3VideoOutpaintLoadSelectionT8"):
                node["pos"], node["size"] = [0, 0], [620, 550]
            elif node["type"] == "MiniMaxH3VideoOutpaintSelectCandidateT8":
                node["pos"], node["size"] = [680, 0], [470, 170]
            elif node["type"] == "PreviewAny":
                node["pos"], node["size"] = [680, 230], [760, 260]
            elif node["type"] == "MarkdownNote":
                node["pos"], node["size"] = [0, -230], [1000, 180]
            else:
                node["pos"][0] -= 1900
        workflow["extra"].update(ds={"scale": 0.8, "offset": [90, 280]},
            outpaint_delivery_status="cpu_fixture_ui_only_not_visual_acceptance")
        path = root / "user" / "default" / "workflows" / (name + ".json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(workflow, stream, ensure_ascii=False, indent=2)
        with (fixture_dir / (name + "_api.json")).open("x", encoding="utf-8") as stream:
            json.dump(prompt, stream, ensure_ascii=False, indent=2)
        workflows[name] = str(path)
    evidence = {"scope": "tiny_CPU_oracle_fixture_not_learned_model_not_quality_acceptance",
        "candidate_id": candidate_id, "candidate_name": "cpu_fixture_a", "run_name": "ui_cpu_fixture",
        "cache_root": str(cache), "plan": plan["plan"], "source_file": source_name,
        "preview_report": report, "sampled_windows": len(windows.snapshot()["committed"]),
        "selection_created": False, "server_sampler_overridden": False, "workflows": workflows}
    with (fixture_dir / "fixture.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"candidate_id": candidate_id, "sampled_windows": evidence["sampled_windows"],
        "fixture": str(fixture_dir), "selection_created": False}))


if __name__ == "__main__":
    main()
