"""Publish a private-mapped pilot pair using a previously browser-tested template.

Only copies original media after the strict run comparison succeeds. No inference,
transcoding, quality verdict, source workflow changes or external publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progressive_pilot_analysis import compare_runs, load_run  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_template(template, hashes, task):
    if task not in {"T2VA", "I2VA"}:
        raise ValueError("Only the qualified fixed pilot task types are supported")
    if set(hashes) != {"A", "B"} or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                                       for v in hashes.values()):
        raise ValueError("Exactly two lowercase SHA-256 media identities are required")
    # No shell, template execution or untrusted text interpolation.
    html, count = re.subn(r"const hashes=\{[^\n]*?\},blobUrls=\[\];",
                         "const hashes=" + json.dumps(hashes) + ",blobUrls=[];", template)
    if count != 1:
        raise ValueError("Browser-tested template hash binding was not found exactly once")
    if task == "I2VA":
        replacements = {
            "H3 渐进首采：第一组 A/B 审看": "H3 渐进首采：图生视频 A/B 审看",
            "H3 渐进首采 · 第一组 A/B 审看": "H3 渐进首采 · 图生视频 A/B 审看",
            "同模型、同新版 EMA B、同提示词和 seed": "同一首帧、同模型、同新版 EMA B、同提示词和 seed",
            "这是第一组短片，不是32秒验收。两条独立生成，人物和声音不要求逐像素/逐采样相同；请听古典音乐和“你在哪里”，看脸、嘴形、动作与闪烁。只播放所选一路声音，避免回声。":
                "这是图生视频的3秒短片，不是32秒验收。同一首帧：白色背景中的机械风格女性与机器人兔子。请看人物与参考的一致性、细节、运动和闪烁，并听环境声有无杂音；本组无对白，不验收口型。只播放所选一路声音，避免回声。",
            "看完请回复：画面 A 更好 / B 更好 / 差不多 / 都不行；并分别说两条的音乐、人声、音量和口型是否正常。允许画面通过但声音不通过，不会自动替你判定。":
                "看完请回复：画面 A 更好 / B 更好 / 差不多 / 都不行；人物是否稳定，环境声是否有杂音。此组没有说话，不用于口型验收；不会自动替你判定。",
        }
        for old, new in replacements.items():
            if html.count(old) != 1:
                raise ValueError("Browser-tested template task text changed; review the template first")
            html = html.replace(old, new)
    # A new page must be actually tested before claiming playback qualification.
    html = html.replace("此页已在独立 Chrome 中验证两路完整播放。", "建议使用独立 Chrome 播放原视频。")
    return html


def build(baseline, candidate, template_path, output):
    research = Path(__file__).resolve().parents[1] / "artifacts/acceleration-research-20260909"
    output = Path(output).resolve()
    if output == research or not output.is_relative_to(research) or output.exists():
        raise ValueError("Use a new dedicated review directory within acceleration research")
    a, b = load_run(baseline), load_run(candidate)
    comparison = compare_runs(a, b)
    if any((r["media"]["timeline"]["width"], r["media"]["timeline"]["height"],
            r["media"]["timeline"]["frames"], r["media"]["timeline"]["fps"]) != (1024, 512, 73, "24")
           for r in (a, b)):
        raise ValueError("Template only supports the verified 1024x512 73-frame 24fps pilot")
    template_path = Path(template_path).resolve(strict=True)
    if not template_path.is_relative_to(research):
        raise ValueError("Template must be the local research review page")
    task = a["terminal"]["case"].split("_", 1)[0]
    sides = [a, b]
    secrets.SystemRandom().shuffle(sides)
    hashes = {label: r["media"]["file"]["sha256"] for label, r in zip(("A", "B"), sides)}
    html = render_template(template_path.read_text(encoding="utf-8"), hashes, task)
    public = output / "public"
    public.mkdir(parents=True)
    for label, r in zip(("A", "B"), sides):
        source = Path(r["reports"]["save"]["output"]).resolve(strict=True)
        shutil.copyfile(source, public / f"{label}.mp4")
        if digest(public / f"{label}.mp4") != hashes[label] or digest(source) != hashes[label]:
            raise RuntimeError("Media changed during byte-identical review copy")
    (public / "review.html").write_text(html, encoding="utf-8")
    receipt = {"status": "built_not_browser_or_human_validated", "task": task,
               "template_sha256": digest(template_path), "html_sha256": digest(public / "review.html"),
               "original_media_sha256": hashes, "original_audio_video_unchanged": True,
               "private_mapping": {label: r["root"] for label, r in zip(("A", "B"), sides)},
               "comparison": comparison, "gpu_started": False}
    (output / "private_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": receipt["status"], "public": str(public), "task": task}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "template", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.candidate, args.template, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
