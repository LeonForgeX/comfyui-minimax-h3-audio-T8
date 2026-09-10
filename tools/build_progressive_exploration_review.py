"""One inline review page for every fixed short exploration pair plus I2VA.

Build only after all ten jobs complete. Original MP4 bytes, independent blind
mapping per pair, no auto-open/download, and no inferred human acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import secrets
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_progressive_pilot import RESEARCH, write_json  # noqa: E402
from run_progressive_exploration import frozen_plan, validate_job  # noqa: E402
from progressive_pilot_analysis import load_run, compare_runs  # noqa: E402


def render_sections(sections):
    if not sections or len({s["id"] for s in sections}) != len(sections):
        raise ValueError("Unique review section ids required")
    result = []
    for section in sections:
        key = section["id"]
        if not key.startswith("pair-") or not key[5:].isdigit():
            raise ValueError("Unsafe review id")
        cards = []
        for label in ("A", "B"):
            cards.append(f'<div><h3>{label}</h3><video data-side="{label}" controls playsinline preload="metadata" '
                f'src="{key}-{label}.mp4" {"muted" if label == "B" else ""}></video>'
                f'<p>{label}：中央原始像素局部（不另做锐化）</p><canvas width="512" height="384"></canvas></div>')
        review = ('<p class="note">这组已收到你的整体通过反馈，仅保留作参照，不要求重审。</p>' if section["accepted"] else
            '<label>画面：<select data-answer="visual"><option value="">未评价</option><option>A 更好</option>'
            '<option>B 更好</option><option>差不多，都可以</option><option>都有问题</option></select></label> '
            '<label>声音：<select data-answer="audio"><option value="">未评价</option><option>两条正常</option>'
            '<option>A 有问题</option><option>B 有问题</option><option>两条有问题</option></select></label> '
            '<label>备注（对白组也请看口型）：<input data-answer="notes" type="text"></label>')
        result.append(f'<section id="{key}" data-review="{str(not section["accepted"]).lower()}"><h2>{escape(section["title"])}</h2>'
            f'<p>{escape(section["description"])}</p><div class="controls"><button data-action="play">本组从头同步播放</button> '
            '<button data-action="pause">暂停本组</button> 声音 <select data-audio><option>A</option><option>B</option></select>'
            ' <input aria-label="本组时间" data-seek type="range" min="0" max="3.0416667" step="0.001" value="0">'
            '<span data-clock>0.000 秒</span></div><p data-status>点击本组同步播放；建议使用 Chrome。</p>'
            f'<div class="pair">{"".join(cards)}</div><div class="verdict">{review}</div></section>')
    return "\n".join(result)


def build(suite_root, output):
    suite_root, output = Path(suite_root).resolve(strict=True), Path(output).resolve()
    if output.exists() or output == RESEARCH or not output.is_relative_to(RESEARCH):
        raise ValueError("Use a new dedicated review directory")
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    terminal, frozen = read(suite_root / "terminal.json"), read(suite_root / "plan.json")
    plan = frozen_plan()
    if (terminal["status"] != "ten_serial_jobs_five_matched_pairs_human_pending" or
            len(terminal["completed"]) != 10 or frozen["jobs"] != plan):
        raise ValueError("Exploration is incomplete or differs from its frozen ten-job plan; retain failure evidence")
    pairs = [("人物对白 · seed 2609032101（已接受参照）", "音乐、你在哪里；此组沿用既有反馈。", True,
              RESEARCH / "gpu-t2va-native8-v2", RESEARCH / "gpu-t2va-progressive6plus2-v3"),
             ("图生视频 · 女性与机器人兔子", "看首帧一致性、眨眼与运动、环境声；本组无对白，不验收口型。", False,
              RESEARCH / "gpu-i2va-native8-v1", RESEARCH / "gpu-i2va-progressive6plus2-v1")]
    for a, b in zip(plan[::2], plan[1::2]):
        for item in (a, b):
            validate_job(load_run(suite_root / item["folder"]), item, frozen)
        content, seed = a["exploration_case"].split("_")
        title = ("人物对白" if content == "portrait" else "游戏骑士运动") + " · seed " + seed
        description = ("请听古典音乐和“你在哪里”，看脸、口型、闪烁。" if content == "portrait" else
                       "请看盔甲、披风、脚步、背景移动与细节，听脚步和环境声；无对白或音乐。同 seed 不保证相同构图，也请留意是否保持了全身画面。")
        pairs.append((title, description, False, suite_root / a["folder"], suite_root / b["folder"]))
    validated = []
    for title, description, accepted, native, progressive in pairs:
        a, b = load_run(native), load_run(progressive)
        comparison = compare_runs(a, b)
        if accepted:
            feedback = read(RESEARCH / "t2va-pilot-review/human_feedback_v1.json")
            if feedback.get("overall") != "tie_both_accepted" or {a["media"]["file"]["sha256"], b["media"]["file"]["sha256"]} != {
                    feedback["A_sha256"], feedback["B_sha256"]}:
                raise ValueError("Prior acceptance is not bound to these exact media")
        validated.append((title, description, accepted, a, b, comparison))
    public = output / "public"
    public.mkdir(parents=True)
    sections, private = [], []
    for index, (title, description, accepted, a, b, comparison) in enumerate(validated):
        section = {"id": f"pair-{index}", "title": title, "description": description, "accepted": accepted}
        sides = [a, b]
        secrets.SystemRandom().shuffle(sides)
        mapping = {}
        for label, run in zip(("A", "B"), sides):
            target = public / (section["id"] + "-" + label + ".mp4")
            shutil.copyfile(run["media"]["file"]["path"], target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != run["media"]["file"]["sha256"]:
                raise RuntimeError("Review copy changed the original video/audio bytes")
            mapping[label] = {"root": run["root"], "sha256": digest}
        sections.append(section)
        private.append({"id": section["id"], "mapping": mapping, "comparison": comparison})
    template = Path(__file__).with_name("progressive_review_hub_template.html").read_text(encoding="utf-8")
    if template.count("<!-- PAIRS -->") != 1:
        raise ValueError("Review template must have exactly one insertion point")
    if template.count("REVIEW_ID_PLACEHOLDER") != 1:
        raise ValueError("Review export must bind one unique page identity")
    review_id = secrets.token_hex(16)
    html = template.replace("<!-- PAIRS -->", render_sections(sections)).replace("REVIEW_ID_PLACEHOLDER", review_id)
    (public / "review.html").write_text(html, encoding="utf-8")
    write_json(output / "private-receipt.json", {"status": "built_not_browser_or_human_qualified",
        "review_id": review_id, "sections": private, "original_media_bytes": True, "all_required_short_pairs_present": True,
        "review_html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "scope": "Short exploration only; not the final long-video/attention/FI delivery gate."})
    return {"public": str(public), "pairs": len(sections), "new_human_reviews": len(sections)-1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.suite_root, args.output), ensure_ascii=False))
