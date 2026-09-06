from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from build_face_refine_window_blind_review import _probe, _sha256, _strict_decode


SCHEMA = "t8.face_refine_sampler_mask_blind_review.v1"


def build(
    source: Path,
    previous: Path,
    corrected: Path,
    analysis: Path,
    output: Path,
) -> dict:
    inputs = {
        "previous_without_v11_fix": previous.resolve(),
        "corrected_with_v11_fix": corrected.resolve(),
    }
    source = source.resolve()
    analysis = analysis.resolve()
    if not source.is_file() or not analysis.is_file():
        raise FileNotFoundError("source and analysis must both exist")
    source_probe = _probe(source)
    metadata = {name: _probe(path) for name, path in inputs.items()}
    if any(value != source_probe for value in metadata.values()):
        raise ValueError(
            f"Blind-review media contracts differ: source={source_probe}, candidates={metadata}"
        )
    for path in (source, *inputs.values()):
        _strict_decode(path)

    shas = {name: _sha256(path) for name, path in inputs.items()}
    source_sha = _sha256(source)
    analysis_sha = _sha256(analysis)
    review_id = hashlib.sha256(
        "|".join((source_sha, *shas.values(), analysis_sha)).encode()
    ).hexdigest()[:16]
    methods = list(inputs)
    random.Random(int(review_id, 16)).shuffle(methods)
    mapping = dict(zip(("A", "B"), methods, strict=True))
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "source.mp4")
    for label, method in mapping.items():
        shutil.copy2(inputs[method], output / f"{label}.mp4")

    public = {
        "schema": "t8.face_refine_sampler_mask_blind_package.v1",
        "review_id": review_id,
        "target_range_abs_inclusive": [0, 23],
        "media_contract": source_probe,
        "source": {"file": "source.mp4", "sha256": _sha256(output / "source.mp4")},
        "sides": {
            label: {"file": f"{label}.mp4", "sha256": _sha256(output / f"{label}.mp4")}
            for label in ("A", "B")
        },
        "analysis_sha256": analysis_sha,
        "mapping_disclosed": False,
    }
    private = {
        "schema": "t8.face_refine_sampler_mask_blind_key.v1",
        "review_id": review_id,
        "mapping": mapping,
        "inputs": {
            name: {"path": str(path), "sha256": shas[name], "probe": metadata[name]}
            for name, path in inputs.items()
        },
        "source": {"path": str(source), "sha256": source_sha, "probe": source_probe},
        "analysis": {"path": str(analysis), "sha256": analysis_sha},
    }
    (output / "public_manifest.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "private_key.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    page_path = output / "blind_review.html"
    page_path.write_text(render_page(public), encoding="utf-8")
    return {"public": public, "private": private, "page": str(page_path.resolve())}


def render_page(public: dict) -> str:
    review_id = public["review_id"]
    sides = json.dumps(public["sides"], ensure_ascii=False)
    review_id_json = json.dumps(review_id)
    page = rf"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Face Refine v1.1 修正盲审</title><style>
body{{margin:0;background:#10141c;color:#eef3ff;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:20px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}section{{background:#1a2230;border:1px solid #344158;border-radius:10px;padding:10px}}video{{width:100%;background:#000}}.source{{max-width:360px;margin-top:16px}}h1{{font-size:24px;margin:0 0 8px}}h2{{margin:0 0 8px}}.grid video{{display:block;max-height:58vh;object-fit:contain}}summary{{cursor:pointer}}button,select,textarea{{font:inherit;padding:8px}}button{{margin:4px}}label{{display:block;margin:8px 0}}textarea{{width:100%;min-height:80px;box-sizing:border-box}}.warn{{color:#ffda74}}@media(max-width:600px){{main{{padding:10px}}.grid{{gap:6px}}section{{padding:6px}}}}</style></head><body><main>
<h1>Face Refine v1.1.1采样遮罩修正 A/B 盲审</h1><p class="warn">原片公开显示；A/B是同一模型、素材、seed和Stock20设置下的旧路线与v1.1.1修正路线，映射已隐藏。请先完整观看，再反复看前0–23帧（约1秒）的眼、口、鼻、皮肤网格感、抖动以及第24帧边界。素材静音，只评画面。</p>
<p><button id="play">A/B 同步播放</button><button id="pause">全部暂停</button><button id="start">回到开头</button><button id="one">定位1.0秒边界</button><span id="status" role="status"></span></p>
<div class="grid" id="videos"><section><h2>A · 左边</h2><video id="video-a" controls muted playsinline preload="auto" src="A.mp4"></video></section><section><h2>B · 右边</h2><video id="video-b" controls muted playsinline preload="auto" src="B.mp4"></video></section></div>
<details class="source"><summary>展开原片参考</summary><video controls muted playsinline preload="metadata" src="source.mp4"></video></details>
<h2>结论</h2><label>总体更好 <select id="overall"><option value="pending">未判断</option><option>A</option><option>B</option><option value="tie">差不多/平</option><option value="none">都不行</option></select></label>
<label>前24帧五官更自然 <select id="features"><option value="pending">未判断</option><option>A</option><option>B</option><option value="tie">差不多/平</option><option value="none">都不行</option></select></label>
<label>网格/发虚/噪点更少 <select id="grid"><option value="pending">未判断</option><option>A</option><option>B</option><option value="tie">差不多/平</option><option value="none">都不行</option></select></label>
<label>时序和接缝更自然 <select id="temporal"><option value="pending">未判断</option><option>A</option><option>B</option><option value="tie">差不多/平</option><option value="none">都不行</option></select></label>
<label>三路是否完整看过 <input id="watched" type="checkbox"></label><label>备注/问题时间点<textarea id="notes"></textarea></label><button id="export">导出评审JSON</button>
<script>const sides={sides},reviewId={review_id_json};const vids=[...document.querySelectorAll('video')];function sync(t){{for(const v of vids)v.currentTime=t}}document.getElementById('play').onclick=async()=>{{sync(vids[0].currentTime);await Promise.all(vids.map(v=>v.play().catch(e=>{{document.getElementById('status').textContent='播放失败：'+e.message}})))}};document.getElementById('pause').onclick=()=>vids.forEach(v=>v.pause());document.getElementById('start').onclick=()=>sync(0);document.getElementById('one').onclick=()=>sync(1);document.getElementById('export').onclick=()=>{{const value={{schema:'{SCHEMA}',review_id:reviewId,exported_at:new Date().toISOString(),review_completed:document.getElementById('watched').checked,overall:document.getElementById('overall').value,features_first24:document.getElementById('features').value,grid_or_blur:document.getElementById('grid').value,temporal_seam:document.getElementById('temporal').value,notes:document.getElementById('notes').value}};const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='face_refine_sampler_mask_blind_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};</script></main></body></html>"""
    return page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.previous, args.corrected, args.analysis, args.output)
    print(
        json.dumps(
            {"page": result["page"], "review_id": result["public"]["review_id"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
