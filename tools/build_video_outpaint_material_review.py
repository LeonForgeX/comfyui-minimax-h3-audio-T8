"""Build a synchronized source-versus-outpaint material review page."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

try:
    from tools.run_dlss_nr_validation import _video_screen
    from tools.outpaint_probe_cases import verify_delivery_report
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from run_dlss_nr_validation import _video_screen
    from outpaint_probe_cases import verify_delivery_report


SCHEMA = "t8.h3.video_outpaint.material_review/v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_review(report_path, window_manifest_path, output, *, title, focus):
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_bytes())
    if report.get("status") != "media_pass_human_review_pending":
        raise ValueError("material run must be mechanically complete and awaiting human review")
    source = Path(report["case"]["source_path"]).resolve(strict=True)
    candidate = Path(report["media"]["path"]).resolve(strict=True)
    if _sha256(source).lower() != report["source_sha256"].lower():
        raise ValueError("source hash differs from the run binding")
    if _sha256(candidate).lower() != report["media"]["sha256"].lower():
        raise ValueError("candidate hash differs from the run report")
    plan = report["case"]["plan"]
    requested_mode = report.get("prompt", {}).get("13", {}).get("inputs", {}).get("source_mode")
    delivery = verify_delivery_report(report["delivery"]["report"], _sha256(candidate), _sha256(source),
        plan["plan_sha256"], expected_plan=plan, expected_source_mode=requested_mode)
    mode = delivery.get("source_mode", "preserve_source")
    manifest_path = Path(window_manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_bytes())
    expected = [(shot, window)
                for shot, planned_shot in enumerate(plan["shots"])
                for window, _ in enumerate(planned_shot["windows"])]
    actual = [(row.get("shot"), row.get("window")) for row in manifest.get("committed", [])]
    receipt_sha = report["delivery"]["report"]["pixel_receipt"]["window_manifest_sha256"]
    if manifest.get("status") != "sampled" or actual != expected or _sha256(manifest_path) != receipt_sha:
        raise ValueError("sampled windows do not match the published plan/receipt")
    metrics = _video_screen(source, candidate, hard_cut=False)
    if metrics["decoded_frame_count"] != plan["source"]["frames"]:
        raise ValueError("mechanical screen decoded an unexpected frame count")
    if metrics["black_regression_frames"] or metrics["freeze_regression_frames"]:
        raise ValueError("candidate failed the black/freeze mechanical screen")

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review evidence: {output}")
    public = output / "public"
    public.mkdir(parents=True)
    shutil.copyfile(source, public / "source.mp4")
    shutil.copyfile(candidate, public / "outpaint.mp4")
    player = Path(__file__).with_name("outpaint_material_player.js")
    review_id = hashlib.sha256(json.dumps({
        "schema": SCHEMA, "source": _sha256(source), "candidate": _sha256(candidate),
        "plan": plan["plan_sha256"], "windows": receipt_sha, "title": title, "focus": focus,
        "source_mode": mode, "delivery_report_sha256": delivery["delivery_report_sha256"],
        "player_sha256": _sha256(player),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    public_manifest = {
        "schema": SCHEMA, "review_id": review_id, "title": title, "focus": focus,
        "source": {"file": "source.mp4", "sha256": _sha256(source)},
        "outpaint": {"file": "outpaint.mp4", "sha256": _sha256(candidate)},
        "plan_sha256": plan["plan_sha256"], "committed_windows": len(actual),
        "output_geometry": [plan["output"]["width"], plan["output"]["height"]],
        "source_rectangle": plan["output"]["source_rect"],
        "source_mode": mode, "source_exact_before_encoding": mode == "preserve_source",
        "source_reconstructed": mode == "joint_decode",
        "frames": plan["source"]["frames"], "fps": 24,
        "delivery_report_sha256": delivery["delivery_report_sha256"],
        "player_sha256": _sha256(player),
        "color_match_report": delivery.get("color_match_report"),
        "mechanical_screen": metrics, "automatic_quality_ranking": False,
    }
    _atomic_json(public / "manifest.json", public_manifest)
    shutil.copyfile(player, public / "player.js")
    config_json = json.dumps(public_manifest, ensure_ascii=False).replace("<", "\\u003c")
    safe_title, safe_focus = html.escape(title), html.escape(focus)
    safe_focus += (" 当前为联合解码：原片区域也经过 VAE 重建，不承诺原像素不变。" if mode == "joint_decode"
                   else " 当前为原片保留：编码前源区域像素保持不变，视频编码本身有损。")
    duration = plan["source"]["frames"] / 24
    buttons = "".join(
        f'<button data-transport disabled data-time="{value:.6f}">{value:.1f} 秒</button>'
        for value in sorted({0.0, min(1.0, duration), min(2.0, duration), duration / 2, max(0.0, duration - 0.25)})
    )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{safe_title}</title><style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:16px system-ui,-apple-system,sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}.note{{color:#f2cc60}}.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}}section{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:12px;min-width:0}}video{{display:block;width:100%;max-height:60vh;background:#000}}canvas{{display:block;max-width:100%;max-height:45vh;width:auto;height:auto}}#crops{{margin:16px 0}}button,select,textarea{{font:inherit;margin:6px;padding:8px;background:#21262d;color:#e6edf3;border:1px solid #484f58;border-radius:6px}}button:disabled{{opacity:.45}}#scrub{{width:100%}}textarea{{display:block;width:calc(100% - 20px);min-height:72px}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>{safe_title}</h1><p class="note">左边原片，右边扩画成片。{safe_focus} 为避免双音轨叠加，左侧原片静音，只播放右侧扩画成片的音频。</p><p><button id="restart" data-transport disabled>从头同步播放</button><button id="play" data-transport disabled>同步播放</button><button id="pause" data-transport disabled>全部暂停</button>{buttons}<span id="status" role="status">正在校验视频，请稍候…</span></p><label>同步定位<input id="scrub" data-transport disabled type="range" min="0" value="0"></label>
<div class="grid"><section><h2>原片（静音）</h2><video id="source" controls muted preload="auto"></video></section><section><h2>扩画成片（播放此音轨）</h2><video id="outpaint" controls preload="auto"></video></section></div><div class="grid" id="crops"></div>
<section><h2>结论</h2><label><input id="viewed" type="checkbox"> 已从头到尾同步看完</label><label>扩画整体 <select id="overall"><option>未判断</option><option>通过</option><option>不通过</option></select></label><label>运动/连续性 <select id="motion"><option>未判断</option><option>自然</option><option>轻微问题可接受</option><option>不可接受</option></select></label><label>边界/颜色 <select id="seam"><option>未判断</option><option>自然</option><option>轻微问题可接受</option><option>不可接受</option></select></label><label>原画内容是否保持 <select id="sourceKeep"><option>未判断</option><option>保持正常</option><option>有可见损坏</option></select></label><label>备注<textarea id="notes"></textarea></label><button id="export">导出审片结果</button></section>
<script id="review-data" type="application/json">{config_json}</script><script src="player.js"></script></main></body></html>'''
    (public / "review.html").write_text(page, encoding="utf-8")
    return {"review_id": review_id, "page": str(public / "review.html"), "manifest": public_manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--focus", required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(args.report, args.window_manifest, args.output,
                                  title=args.title, focus=args.focus), ensure_ascii=False, indent=2))
