"""Aggregate existing qualified short/Sage/FI media without rerunning generation.

Builds a new public-only page, does not open a browser or assert acceptance. H3
pairs are blind; FI explicitly labels source/interpolated and never calls it SR.
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
from build_progressive_exploration_review import build as build_base, render_sections  # noqa: E402
from progressive_pilot_analysis import load_run, compare_runs  # noqa: E402
from run_progressive_pilot import RESEARCH  # noqa: E402
from progressive_probe_control import file_identity  # noqa: E402


def add_lipsync_questions(html, pair_ids):
    """Require an explicit mouth-sync answer only for new dialogue cases."""
    question = ('<label>口型：<select data-answer="lipsync"><option value="">未评价</option>'
                '<option>两条正常</option><option>A 有问题</option><option>B 有问题</option>'
                '<option>两条有问题</option><option>看不清／无法判断</option></select></label> ')
    for key in pair_ids:
        marker = f'<section id="{key}" '
        if html.count(marker) != 1:
            raise ValueError('Ambiguous dialogue section')
        start = html.index(marker)
        end = html.index('</section>', start)
        section = html[start:end]
        target = '<label>备注（对白组也请看口型）：'
        if section.count(target) != 1 or 'data-answer="lipsync"' in section:
            raise ValueError('Dialogue form changed')
        html = html[:start]+section.replace(target, question+target)+html[end:]
    return html


def render_fi(section):
    section = dict(section, accepted=False)
    result = render_sections([section])
    result = result.replace('max="3.0416667"', f'max="{section["duration"]:.9f}"')
    result = result.replace('<h3>A</h3>', f'<h3>A：原片 {escape(section["source_fps"])}fps</h3>')
    result = result.replace('<h3>B</h3>', f'<h3>B：插帧 {escape(section["target_fps"])}fps</h3>')
    if not section['has_audio']:
        start = result.index('<label>声音：')
        end = result.index('</label>', start)+len('</label>')
        result = result[:start]+'<span>本组原片没有声音，不评价音频。</span>'+result[end:]
    return result


def verified_pair(root, status):
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    terminal, source, postflight = read('terminal.json'), read('source.json'), read('postflight-v1.json')
    if terminal['status'] != status or terminal['server_stop']['owned_children_remaining']:
        raise ValueError('FI run did not complete its owned cleanup')
    if postflight['status'] not in ('actual_media_records_audio_and_full_decode_pass', 'fixed_boundary_media_and_records_pass'):
        raise ValueError('FI independent postflight is missing')
    if postflight['candidate'] != terminal['candidate']:
        raise ValueError('FI postflight is not bound to this candidate')
    for identity in (source['file'], terminal['candidate']):
        if file_identity(identity['path']) != identity:
            raise ValueError('Review media changed since qualification')
    return source, terminal, postflight


def build(suite, output):
    build_base(suite, output)
    output = Path(output).resolve(strict=True)
    public = output/'public'
    html_path = public/'review.html'
    html = html_path.read_text(encoding='utf8')
    original = json.loads((output/'private-receipt.json').read_text(encoding='utf8'))
    sections, extras = [], []

    def copy_pair(index, identities, metadata):
        mapping = {}
        for side, identity in zip(('A', 'B'), identities):
            if file_identity(identity['path']) != identity:
                raise ValueError('Original media identity changed')
            target = public/f'pair-{index}-{side}.mp4'
            if target.exists():
                raise FileExistsError(target)
            shutil.copyfile(identity['path'], target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != identity['sha256']:
                raise ValueError('Review copy changed original video or audio')
            mapping[side] = identity
        extras.append({'id': f'pair-{index}', 'mapping': mapping, **metadata})

    sage = RESEARCH/'qualification-sage-short-gpu-20260910-v1'
    postflight = json.loads((sage/'postflight-v1.json').read_text(encoding='utf8'))
    if postflight['status'] != 'two_runs_eight_queues_independent_postflight_pass':
        raise ValueError('Actual Sage pair not qualified')
    a, b = load_run(sage/'native8'), load_run(sage/'progressive6plus2')
    comparison = compare_runs(a, b)
    if comparison != postflight['comparison']:
        raise ValueError('Sage comparison differs from independent postflight')
    sides = [a['media']['file'], b['media']['file']]
    secrets.SystemRandom().shuffle(sides)
    index = len(original['sections'])
    copy_pair(index, sides, {'kind': 'sage_blind', 'comparison': comparison})
    sections.append(render_sections([{'id': f'pair-{index}', 'accepted': False,
        'title': 'Sage 注意力 · 人物对白',
        'description': '本组1024×512、73帧、24fps。请听古典音乐、“你在哪里”，看脸、嘴形与闪烁；A/B仍隐藏采样路线。'}]))

    cases = [('dlss-fi-file-probe-20260910-v1', 'actual_73_source_72_generated_1_tail_audio_exact',
              '游戏人物 · 24→48fps', '已有H3游戏成片，只做插帧，不做超分。看运动、遮挡和重影，原音轨保持。', True),
             ('dlss-fi-hud30-gpu-20260910-v1', 'actual_boundary_complete', '固定字幕 · 30→60fps',
              'CPU合成诊断片：看固定文字和移动方块边缘。不是H3生成，也未声称字幕保护。', False),
             ('dlss-fi-flash30-gpu-20260910-v1', 'actual_boundary_complete', '闪光、低纹理 · 30→60fps',
              'CPU合成诊断片：看圆形轮廓及短暂闪光前后是否拖影。闪光没有被当作切镜。', False),
             ('dlss-fi-hardcut24-gpu-20260910-v1', 'actual_boundary_complete', '硬切 · 24→48fps',
              'CPU合成诊断片：1秒位置显式切镜，不生成跨镜帧；不是自动切镜检测。', False)]
    for folder, status, title, description, has_audio in cases:
        index += 1
        source, terminal, postflight = verified_pair(RESEARCH/folder, status)
        copy_pair(index, [source['file'], terminal['candidate']], {'kind': 'fi_labeled', 'postflight': postflight})
        sections.append(render_fi({'id': f'pair-{index}', 'title': title, 'description': description,
            'has_audio': has_audio, 'duration': float(__import__('fractions').Fraction(postflight['duration'])),
            'source_fps': source['plan']['source_rate'], 'target_fps': postflight['fps']}))

    old_intro = '<p>所有视频就在这一页。每组都是 1024×512、73 帧、24fps、约 3 秒；同组模型、提示词、seed 与总求值次数相同。A/B 顺序各组独立打乱。</p>'
    old_notice = '<p class="note">这是短片探索汇总，不是完整 32 秒或插帧验收。只播放当前组的一路声音；检查画面时可暂停拖动时间。建议使用 Chrome，内置浏览器此前有黑屏问题。</p>'
    if html.count(old_intro) != 1 or html.count(old_notice) != 1:
        raise ValueError('Base review template changed')
    html = html.replace('H3 渐进采样 · 短片集中审看', 'H3 渐进采样与插帧 · 集中审看')
    html = html.replace(old_intro, '<p>全部对照直接显示在本页。前面是渐进采样A/B盲测，后面是明确标注原片／插帧的对照。每组尺寸、时长和有无声音单独说明。</p>')
    html = html.replace(old_notice, '<p class="note">待审素材汇总，不代表全部开发或发布完成。32秒测试因显存余量保护停止，没有合格视频，不列为通过。TRT未安装／测试。建议使用Chrome；只播放当前组一路声音。</p>')
    marker = '<p>填完可导出一份反馈，也可以直接在聊天中按组回复。不会自动提交或替你判定。</p>'
    if html.count(marker) != 1:
        raise ValueError('Feedback insertion point changed')
    html = html.replace(marker, '\n'.join(sections)+'\n'+marker)
    # Fixed builder order: first accepted portrait, I2VA, two new portraits,
    # three games, then Sage. Do not request lipsync for silent/game cases.
    html = add_lipsync_questions(html, ('pair-2', 'pair-3', 'pair-7'))
    html = html.replace('t8.progressive.exploration-human.v1', 't8.acceleration.combined-human.v1')
    html = html.replace('progressive_exploration_human_review.json', 'acceleration_combined_human_review.json')
    html_path.write_text(html, encoding='utf8')
    with (output/'combined-private-receipt.json').open('x', encoding='utf8') as receipt:
        json.dump({'status': 'built_not_browser_or_human_qualified', 'review_id': original['review_id'],
            'sections': original['sections']+extras, 'pairs': index+1, 'original_media_bytes': True,
            'review_html_sha256': hashlib.sha256(html_path.read_bytes()).hexdigest(),
            'not_qualified': ['long32', 'TRT', 'public_FI_node', 'final_release']}, receipt, ensure_ascii=False, indent=2)
    return {'status': 'built_no_browser_action', 'public': str(public), 'pairs': index+1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.suite_root, args.output), ensure_ascii=False))
