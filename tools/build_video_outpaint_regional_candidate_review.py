"""Build a blinded plain-versus-regional H3 outpaint first-frame review."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image

try:
    from tools import outpaint_probe_cases as cases
except ModuleNotFoundError:
    import outpaint_probe_cases as cases


SCHEMA = "t8.h3.video_outpaint.regional_candidate_blind_review/v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _sealed(value):
    body = dict(value)
    digest = body.pop("sha256", None)
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != digest:
        raise ValueError("candidate review evidence seal mismatch")
    return body


def _base_model_contract(report, inputs, method):
    """Compare recorded base files/graph, not regional-composed MODEL hashes."""
    graph = report["prompt"]
    link = inputs.get("model")
    if not isinstance(link, list) or len(link) != 2 or link[1] != 0 or link[0] not in graph:
        raise ValueError("candidate model link is missing")
    root = graph[link[0]]
    if method == "regional":
        if (root["class_type"] != "MiniMaxH3VideoOutpaintRegionalModelT8"
                or set(root["inputs"]) != {"model", "prepared", "query_chunk_rows"}):
            raise ValueError("regional arm needs exactly the known regional wrapper")
        link = root["inputs"]["model"]
    visiting = set()
    files = []
    def expand(value):
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in graph:
            key, slot = value
            if key in visiting:
                raise ValueError("cyclic model graph")
            visiting.add(key)
            node = graph[key]
            if node["class_type"] == "MiniMaxH3VideoOutpaintRegionalModelT8":
                raise ValueError("unexpected nested regional wrapper")
            if node["class_type"] == "UNETLoader":
                name = Path(node["inputs"]["unet_name"]).name
                matches = [item for item in report["models"] if Path(item["path"]).name == name]
                if len(matches) != 1:
                    raise ValueError("base model file identity is missing or ambiguous")
                item = matches[0]
                files.append({"name": name, "bytes": item["bytes"], "sha256": item["sha256"].lower()})
            result = {"class_type": node["class_type"], "slot": slot,
                      "inputs": {k: expand(v) for k, v in node["inputs"].items()}}
            visiting.remove(key)
            return result
        return value
    tree = expand(link)
    if len(files) != 1:
        raise ValueError("review requires one recorded base UNET")
    return {"graph": tree, "files": files}


def _comparison_contract(report, rgb, plan, method):
    preview = _sealed(report["preview_report"])
    mode = cases.pixel_receipt_module().validate_source_mode(preview.get("source_mode", "preserve_source"))
    if (preview.get("schema") != "t8.h3.video_outpaint.candidate_first_frame/v1"
            or preview.get("plan_sha256") != plan["plan_sha256"]
            or preview.get("source_sha256") != plan["source"]["sha256"]
            or preview.get("rgb8_sha256") != hashlib.sha256(rgb.astype(np.uint8).tobytes()).hexdigest()
            or (preview.get("width"), preview.get("height")) != (rgb.shape[1], rgb.shape[0])
            or preview.get("source_exact_before_encoding") is not (mode == "preserve_source")
            or preview.get("source_reconstructed", False) is not (mode == "joint_decode")):
        raise ValueError("preview source/plan/pixels/mode evidence mismatch")
    candidates = [node["inputs"] for node in report["prompt"].values()
                  if node["class_type"] == "MiniMaxH3VideoOutpaintCandidateT8"]
    if len(candidates) != 1:
        raise ValueError("review needs exactly one candidate graph")
    inputs = candidates[0]
    if cases.candidate_finish_inputs(inputs)["source_mode"] != mode:
        raise ValueError("preview and graph source modes differ")
    cache = Path(report["cache_root"]).resolve(strict=True)
    manifest_path = cache / "outpaint_windows.json"
    if _sha256(manifest_path) != preview.get("window_manifest_sha256"):
        raise ValueError("preview sampling manifest changed")
    manifest = _sealed(json.loads(manifest_path.read_bytes()))
    records = manifest.get("committed", [])
    if (manifest.get("schema") != "t8.h3.video_outpaint.window_store/v1"
            or manifest.get("status") not in {"paused", "sampled"} or len(records) != 1
            or (records[0].get("shot"), records[0].get("window")) != (0, 0)
            or records[0].get("sha256") != report["new_first_window_sha256"]):
        raise ValueError("review requires a committed first-window candidate")
    record = records[0]
    payload = cache / f"window-{record['sha256']}.safetensors"
    if payload.stat().st_size != record["bytes"] or _sha256(payload) != record["sha256"]:
        raise ValueError("candidate sampled payload changed")
    identity = manifest["identity"]
    if identity.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("sampling plan differs")
    for field in ("seed", "steps"):
        if type(inputs.get(field)) is not int or inputs[field] != identity.get(field):
            raise ValueError("sampling settings differ from candidate graph")
    if type(inputs.get("color_match")) is not bool:
        raise ValueError("color_match must be boolean")
    required = {"plan_sha256", "model_sha256", "implementation_sha256", "source_cache_sha256",
                "audio_source_sha256", "conditioning_sha256", "seed", "steps", "noise_algorithm",
                "sampler_name", "scheduler"}
    if set(identity) != required:
        raise ValueError("sampling identity is incomplete")
    return {"source_mode": mode, "sampling": {k: v for k, v in identity.items()
                                               if k not in {"conditioning_sha256", "model_sha256"}},
            "base_model": _base_model_contract(report, inputs, method),
            "video_vae_sha256": preview["video_vae_sha256"], "color_match": inputs["color_match"],
            "geometry_align": cases.candidate_finish_inputs(inputs)["geometry_align"],
            "color_settings": preview["color_settings"], "geometry_settings": preview.get("geometry_settings", {})}


def _candidate(report_path, expected_status, method, plan):
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_bytes())
    if report.get("status") != expected_status or report.get("human_acceptance") is not False:
        raise ValueError(f"{method} report is not a mechanically complete candidate awaiting review")
    image = Path(report["image_path"]).resolve(strict=True)
    if _sha256(image) != image.stem.removeprefix("preview-"):
        raise ValueError(f"{method} preview filename/hash mismatch")
    with Image.open(image) as picture:
        dimensions = list(picture.size)
        rgb = np.asarray(picture.convert("RGB"), dtype=np.int16)
    contract = _comparison_contract(report, rgb, plan, method)
    return {
        "method": method,
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "image": str(image),
        "image_sha256": _sha256(image),
        "candidate_id": report["candidate_id"],
        "window_sha256": report["new_first_window_sha256"],
        "source_sha256": report["source_sha256"].lower(),
        "plan_sha256": report["plan_sha256"],
        "dimensions": dimensions,
        "rgb": rgb,
        "comparison_contract": contract,
    }


def build_review(plain_report, regional_report, output, seed=20260907, *, plan=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review evidence: {output}")
    if plan is None:
        raise ValueError("an explicit sealed plan is required; review margins are never guessed")
    plan = cases.plan_module().validate_outpaint_plan(plan)
    plain = _candidate(plain_report, "candidate_generated_human_review_pending", "plain", plan)
    regional = _candidate(
        regional_report, "regional_candidate_generated_human_review_pending", "regional", plan)
    if plain["source_sha256"] != regional["source_sha256"]:
        raise ValueError("candidate sources differ")
    if plain["plan_sha256"] != regional["plan_sha256"]:
        raise ValueError("candidate plans differ")
    if plain["dimensions"] != regional["dimensions"]:
        raise ValueError("candidate dimensions differ")
    if (plain["source_sha256"] != plan["source"]["sha256"]
            or plain["plan_sha256"] != plan["plan_sha256"]
            or plain["dimensions"] != [plan["output"]["width"], plan["output"]["height"]]):
        raise ValueError("candidate does not match the explicit review plan")
    if plain["comparison_contract"] != regional["comparison_contract"]:
        raise ValueError("candidate comparison settings differ; this is not a regional-only pair")
    width, height = plain["dimensions"]
    source_x0, source_y0, source_x1, source_y1 = plan["output"]["source_rect"]
    if source_x0 != 0 or source_x1 != width or not 0 < source_y0 < source_y1 < height:
        raise ValueError("this regional top/bottom review requires top and bottom expansion without side margins")
    diff = np.abs(plain["rgb"] - regional["rgb"])
    regions = {
        "top": diff[:source_y0],
        "source_rectangle": diff[source_y0:source_y1],
        "bottom": diff[source_y1:],
        "full": diff,
    }
    metrics = {
        name: {
            "mean_absolute_rgb_difference": float(values.mean()),
            "changed_rgb_values_percent": float((values != 0).mean() * 100),
            "max_absolute_rgb_difference": int(values.max()),
        }
        for name, values in regions.items()
    }
    public = output / "public"
    public.mkdir(parents=True)
    arms = [plain, regional]
    random.Random(seed).shuffle(arms)
    key = {}
    public_arms = []
    for label, arm in zip(("A", "B"), arms):
        filename = f"{label}.png"
        shutil.copyfile(arm["image"], public / filename)
        with Image.open(arm["image"]) as picture:
            rgb = picture.convert("RGB")
            rgb.crop((0, 0, width, source_y0)).save(public / f"{label}_top.png")
            rgb.crop((0, source_y1, width, height)).save(public / f"{label}_bottom.png")
        key[label] = {
            key_name: value for key_name, value in arm.items()
            if key_name not in {"rgb"}
        }
        public_arms.append({
            "label": label,
            "file": filename,
            "sha256": arm["image_sha256"],
            "candidate_id": arm["candidate_id"],
            "window_sha256": arm["window_sha256"],
        })
    review_id = hashlib.sha256(json.dumps({
        "schema": SCHEMA,
        "seed": seed,
        "source_sha256": plain["source_sha256"],
        "plan_sha256": plain["plan_sha256"],
        "arms": public_arms,
        "comparison_contract": plain["comparison_contract"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    public_manifest = {
        "schema": SCHEMA,
        "review_id": review_id,
        "source_sha256": plain["source_sha256"],
        "plan_sha256": plain["plan_sha256"],
        "dimensions": [width, height],
        "source_rectangle": [0, source_y0, width, source_y1],
        "source_mode": plain["comparison_contract"]["source_mode"],
        "arms": public_arms,
        "mapping_in_public_files": False,
        "mechanical_metrics_displayed_to_reviewer": False,
        "automatic_quality_ranking": False,
    }
    private = {
        "schema": "t8.h3.video_outpaint.regional_candidate_blind_key/v1",
        "review_id": review_id,
        "seed": seed,
        "mapping": key,
        "mechanical_difference_only_not_quality": metrics,
    }
    _atomic_json(public / "manifest.json", public_manifest)
    _atomic_json(output / "private_key.json", private)
    title = html.escape("H3 扩画区域提示 首窗 A/B 盲审")
    common = plain["comparison_contract"]
    mode_notice = ("两路均联合解码，原片区域会经 VAE 重建。" if common["source_mode"] == "joint_decode"
                   else "两路均使用原片像素保留模式。")
    settings_notice = html.escape(f"已核对同一原片、计划、底模文件与基础节点设置、seed {common['sampling']['seed']}、"
                                 f"{common['sampling']['steps']} 步及后处理设置；输出 {width}×{height}。{mode_notice}")
    page = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{title}</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:16px system-ui,-apple-system,sans-serif}}main{{max-width:1580px;margin:auto;padding:22px}}h1{{margin:0 0 8px}}.note{{color:#f2cc60}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px}}img{{display:block;width:100%;height:auto;background:#000}}.crop img{{image-rendering:auto}}.split{{position:relative;max-width:1100px;margin:auto;overflow:hidden}}.split img{{width:100%}}.split .over{{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}}.line{{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#ff4d4d;pointer-events:none}}input[type=range]{{width:100%}}select,textarea,button{{font:inherit;margin:6px 0;padding:8px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}textarea{{width:100%;min-height:72px}}label{{display:block;margin-top:10px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>{title}</h1><p class=\"note\">{settings_notice} 本页只比较首帧，不代表完整视频或最终验收。只看：上方扩出区、下方扩出区、与原图的衔接；不要根据标签猜实现。</p>
<div class=\"grid\"><section><h2>A</h2><img src=\"A.png\"></section><section><h2>B</h2><img src=\"B.png\"></section></div>
<h2>拖动半屏对比</h2><section><div class=\"split\"><img src=\"B.png\"><img class=\"over\" src=\"A.png\"><div class=\"line\"></div></div><input id=\"slider\" type=\"range\" min=\"0\" max=\"100\" value=\"50\"><div>A 在左侧覆盖，B 在右侧显示</div></section>
<h2>上方扩出区放大</h2><div class=\"grid crop\"><section><h3>A</h3><img src=\"A_top.png\"></section><section><h3>B</h3><img src=\"B_top.png\"></section></div>
<h2>下方扩出区放大</h2><div class=\"grid crop\"><section><h3>A</h3><img src=\"A_bottom.png\"></section><section><h3>B</h3><img src=\"B_bottom.png\"></section></div>
<section><h2>结论</h2><label><input id=\"viewed\" type=\"checkbox\"> 已完整看过全图、上方和下方</label><label>总体更自然 <select id=\"overall\"><option>未判断</option><option>A</option><option>B</option><option>差不多</option><option>都不行</option></select></label><label>上方扩出区 <select id=\"top\"><option>未判断</option><option>A</option><option>B</option><option>差不多</option><option>都不行</option></select></label><label>下方扩出区 <select id=\"bottom\"><option>未判断</option><option>A</option><option>B</option><option>差不多</option><option>都不行</option></select></label><label>接缝/颜色 <select id=\"seam\"><option>未判断</option><option>A</option><option>B</option><option>差不多</option><option>都不行</option></select></label><label>备注<textarea id=\"notes\"></textarea></label><button id=\"export\">导出审片结果</button></section>
<script>const rid={json.dumps(review_id)};const over=document.querySelector('.over'),line=document.querySelector('.line');document.getElementById('slider').oninput=e=>{{const x=e.target.value;over.style.clipPath=`inset(0 ${{100-x}}% 0 0)`;line.style.left=x+'%'}};document.getElementById('export').onclick=()=>{{const out={{schema:'{SCHEMA}',review_id:rid,exported_at:new Date().toISOString(),review_completed:document.getElementById('viewed').checked,overall:document.getElementById('overall').value,top:document.getElementById('top').value,bottom:document.getElementById('bottom').value,seam_color:document.getElementById('seam').value,notes:document.getElementById('notes').value}};const b=new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='outpaint_regional_candidate_blind_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};</script></main></body></html>"""
    (public / "blind_review.html").write_text(page, encoding="utf-8")
    return {"review_id": review_id, "page": str((public / "blind_review.html").resolve()),
            "public_manifest": public_manifest, "private_key": private}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-report", type=Path, required=True)
    parser.add_argument("--regional-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--plan", type=Path, required=True, help="sealed outpaint plan JSON; no guessed margins")
    arguments = parser.parse_args()
    print(json.dumps(build_review(arguments.plain_report, arguments.regional_report,
                                  arguments.output, arguments.seed,
                                  plan=json.loads(arguments.plan.read_bytes())), ensure_ascii=False, indent=2))
