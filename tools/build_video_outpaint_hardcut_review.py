"""Build a synchronized source-versus-outpaint hard-cut review page."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

try:
    from tools.run_dlss_nr_validation import _video_screen
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_dlss_nr_validation import _video_screen


SCHEMA = "t8.h3.video_outpaint.hardcut_review/v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_review(report_path, window_manifest_path, output):
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_bytes())
    if report.get("status") != "media_pass_human_review_pending":
        raise ValueError("hard-cut run must be mechanically complete and awaiting human review")
    source = Path(report["case"]["source_path"]).resolve(strict=True)
    candidate = Path(report["media"]["path"]).resolve(strict=True)
    if _sha256(source).lower() != report["source_sha256"].lower():
        raise ValueError("source hash differs from the run binding")
    if _sha256(candidate).lower() != report["media"]["sha256"].lower():
        raise ValueError("candidate hash differs from the run report")
    plan = report["case"]["plan"]
    cuts = plan["request"]["cut_frames"]
    if len(cuts) != 1 or len(plan["shots"]) != 2:
        raise ValueError("review requires one hard cut and exactly two shots")
    manifest_path = Path(window_manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_bytes())
    committed = manifest.get("committed", [])
    expected_windows = [(shot, window)
                        for shot, planned_shot in enumerate(plan["shots"])
                        for window, _ in enumerate(planned_shot["windows"])]
    actual_windows = [(row.get("shot"), row.get("window")) for row in committed]
    if manifest.get("status") != "sampled" or actual_windows != expected_windows:
        raise ValueError("sampled windows do not match the two-shot plan")
    receipt_sha = report["delivery"]["report"]["pixel_receipt"]["window_manifest_sha256"]
    if _sha256(manifest_path) != receipt_sha:
        raise ValueError("window manifest differs from the published pixel receipt")
    if any(window["context_video_latents"] or window["context_audio_latents"]
           for shot in plan["shots"] for window in shot["windows"]):
        raise ValueError("hard-cut review requires zero cross-shot continuation context")
    color_path = Path(__file__).resolve().parents[1] / 'h3_t8/video_outpaint_color.py'
    delivered_color_sha = report["delivery"]["report"]["composition_implementation_sha256"][color_path.name]
    if _sha256(color_path) != delivered_color_sha:
        raise ValueError("color implementation differs from the generated artifact")
    metrics = _video_screen(source, candidate, hard_cut=True)
    cut = metrics["hard_cut"]
    if not all(cut[field] for field in (
        "source_has_mechanical_hard_cut", "candidate_preserves_cut_transition",
        "post_cut_closer_to_current_source_than_previous_source",
    )):
        raise ValueError("candidate failed the mechanical hard-cut screen")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review evidence: {output}")
    public = output / "public"
    public.mkdir(parents=True)
    shutil.copyfile(source, public / "source.mp4")
    shutil.copyfile(candidate, public / "outpaint.mp4")
    fps = 24
    cut_frame = cuts[0]
    review_id = hashlib.sha256(json.dumps({
        "schema": SCHEMA,
        "source_sha256": _sha256(source),
        "candidate_sha256": _sha256(candidate),
        "plan_sha256": plan["plan_sha256"],
        "window_manifest_sha256": receipt_sha,
        "cut_frame": cut_frame,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_out = {
        "schema": SCHEMA,
        "review_id": review_id,
        "source": {"file": "source.mp4", "sha256": _sha256(source)},
        "outpaint": {"file": "outpaint.mp4", "sha256": _sha256(candidate)},
        "cut_frame": cut_frame,
        "cut_time_seconds": cut_frame / fps,
        "plan_sha256": plan["plan_sha256"],
        "two_independent_shots": actual_windows == [(0, 0), (1, 0)],
        "cross_shot_video_context_latents": 0,
        "cross_shot_audio_context_latents": 0,
        "color_match_enabled": report["delivery"]["report"]["color_match_enabled"],
        "color_reset_evidence": {
            "kind": "content_bound_implementation_and_plan_inference",
            "runtime_counter_available_for_this_legacy_artifact": False,
            "future_reports_emit_explicit_shot_reset_count": True,
            "color_implementation_sha256": delivered_color_sha,
            "shot_starts": [shot["start"] for shot in plan["shots"]],
        },
        "mechanical_screen": metrics,
        "automatic_quality_ranking": False,
    }
    _atomic_json(public / "manifest.json", manifest_out)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 扩画硬切镜头审片</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:16px system-ui,-apple-system,sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}.note{{color:#f2cc60}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px}}video{{display:block;width:100%;background:#000}}button,select,textarea{{font:inherit;margin:6px;padding:8px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}textarea{{display:block;width:calc(100% - 20px);min-height:72px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>H3 扩画硬切镜头同步审片</h1><p class="note">左边是原片，右边是扩画成片。硬切发生在第 {cut_frame} 帧（{cut_frame / fps:.3f} 秒）。重点看右侧扩画区域在切点处是否自然换场，不能残留上一镜头的颜色或内容。为避免双音轨叠加，左侧原片静音，只播放右侧扩画成片的音频。</p>
<p><button id="play">同步播放</button><button id="pause">全部暂停</button><button data-time="{max(0, (cut_frame-3)/fps):.6f}">切点前 3 帧</button><button data-time="{cut_frame/fps:.6f}">切点</button><button data-time="{(cut_frame+3)/fps:.6f}">切点后 3 帧</button><span id="status"></span></p>
<div class="grid"><section><h2>原片（静音）</h2><video id="source" controls muted preload="metadata" src="source.mp4"></video></section><section><h2>扩画成片（播放此音轨）</h2><video id="outpaint" controls preload="metadata" src="outpaint.mp4"></video></section></div>
<section><h2>结论</h2><label><input id="viewed" type="checkbox"> 已同步看过切点前后</label><label>扩画整体 <select id="overall"><option>未判断</option><option>通过</option><option>不通过</option></select></label><label>切点是否残留上一镜头 <select id="history"><option>未判断</option><option>没有</option><option>有</option></select></label><label>切点颜色是否自然 <select id="color"><option>未判断</option><option>自然</option><option>有跳变但可接受</option><option>不可接受</option></select></label><label>备注<textarea id="notes"></textarea></label><button id="export">导出审片结果</button></section>
<script>const videos=[document.getElementById('source'),document.getElementById('outpaint')],status=document.getElementById('status');function seek(t){{videos.forEach(v=>v.currentTime=t);status.textContent=` 已定位 ${{t.toFixed(3)}} 秒`}}document.getElementById('play').onclick=async()=>{{const t=Math.min(...videos.map(v=>v.currentTime));seek(t);await Promise.all(videos.map(v=>v.play()))}};document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{{videos.forEach(v=>v.pause());seek(Number(b.dataset.time))}});videos[1].ontimeupdate=()=>{{if(!videos[0].paused&&Math.abs(videos[0].currentTime-videos[1].currentTime)>.08)videos[0].currentTime=videos[1].currentTime}};document.getElementById('export').onclick=()=>{{const out={{schema:'{SCHEMA}',review_id:'{review_id}',exported_at:new Date().toISOString(),review_completed:document.getElementById('viewed').checked,overall:document.getElementById('overall').value,previous_shot_history:document.getElementById('history').value,cut_color:document.getElementById('color').value,notes:document.getElementById('notes').value}};const blob=new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='outpaint_hardcut_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};</script></main></body></html>'''
    (public / "review.html").write_text(page, encoding="utf-8")
    return {"review_id": review_id, "page": str(public / "review.html"), "manifest": manifest_out}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_review(arguments.report, arguments.window_manifest, arguments.output),
                     ensure_ascii=False, indent=2))
