"""Append audited video-reference and static-text checks, reuse prior media bytes."""
import argparse
import base64
from copy import deepcopy
import json
from pathlib import Path
import secrets
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'artifacts/acceleration-research-20260909'
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "h3_t8"))
from trt_vae_build import digest_file, write_new_json  # noqa: E402
from tools.build_progressive_exploration_review import render_sections  # noqa: E402


def text_section(root):
    audit = json.loads((root/'independent-text-audit.json').read_text(encoding='utf8'))
    if audit['status'] != 'static_text_actual_public_vae_tensor_png_lifecycle_pass_human_pending':
        raise ValueError('Independent text evidence required')
    cards = []
    for key, title in [('source','输入测试图'),('native','原生编码＋原生解码'),
                       ('trt_same_latent','同一原生潜空间＋TRT解码'),('trt_full','TRT编码＋TRT解码')]:
        path = root/(key+'.png')
        if digest_file(path) != audit['files'][path.name]:
            raise ValueError('Text preview identity differs')
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        cards.append(f'<h3>{title}</h3><div style="overflow:auto"><img alt="{title}" width="1024" height="512" src="data:image/png;base64,{data}"></div>')
    return ('<section id="text-check" data-review="true"><h2>文字专项 · 原尺寸静态对照</h2>'
        '<p>固定中英文测试图，不是模型生成文字。四图均1024×512原尺寸，窄窗口可横向滚动。'
        '先看原生重建比输入损失了多少，再看TRT是否额外损伤笔画。此项不验证视频闪烁，也没有用PSNR代替可读性。</p>'
        + ''.join(cards)
        + '<label>文字：<select data-answer="visual"><option value="">未评价</option>'
        '<option>TRT与原生差不多，可以接受</option><option>TRT解码额外损失明显</option>'
        '<option>TRT全流程额外损失明显</option><option>两种TRT都有问题</option><option>无法判断</option></select></label>'
        '<label>备注：<input data-answer="notes" type="text"></label></section>')


def build(previous, video_audit, text_root, output):
    previous, video_audit, text_root = (Path(p).resolve(strict=True) for p in (previous,video_audit,text_root))
    output = Path(output).resolve()
    if output.exists() or output == RESEARCH or not output.is_relative_to(RESEARCH):
        raise ValueError('New dedicated research review directory required')
    for path in (previous,video_audit,text_root):
        if not path.is_relative_to(RESEARCH):
            raise ValueError('Evidence outside research')
    old = json.loads((previous/'private-receipt.json').read_text(encoding='utf8'))
    if old['status'] != 'seven_short_pairs_built_human_pending':
        raise ValueError('Expected existing seven-pair review')
    html_path = previous/'public/review.html'
    if digest_file(html_path) != old['review_html_sha256']:
        raise ValueError('Previous page changed')
    pair = json.loads(video_audit.read_text(encoding='utf8'))
    if pair['status'] != 'encoder_fixed_sampler_pair_evidence_pass_human_review_pending' or pair.get('reference_kind') != 'video':
        raise ValueError('Audited complete video-reference pair required')
    text_html = text_section(text_root)
    public = output/'public'
    public.mkdir(parents=True)
    sections = deepcopy(old['sections'])
    sources = [run['media']['file'] for run in pair['runs']]
    secrets.SystemRandom().shuffle(sources)
    sections.append({'id':'pair-8','mapping':dict(zip(('A','B'),sources,strict=True))})
    for section in sections:
        for side, item in section['mapping'].items():
            source = Path(item['path']).resolve(strict=True)
            if not source.is_relative_to(RESEARCH) or digest_file(source) != item['sha256']:
                raise ValueError('Audited media changed')
            target = public/(section['id']+'-'+side+'.mp4')
            shutil.copyfile(source,target)
            if digest_file(target) != item['sha256']:
                raise ValueError('Media copy differs')
    extra = render_sections([{'id':'pair-8','title':'短参考视频编码 · 后续生成影响',
        'description':'完整73帧参考视频，原生/TRT已存实际编码分别进入视频参考条件后独立8步生成。最终均原生解码。模型、提示词、seed相同；看人物、动作和背景，听环境声。无对白，不验口型；不是编码测速。',
        'accepted':False}])
    html = html_path.read_text(encoding='utf8')
    marker = '<p>填完可导出一份反馈'
    if html.count(marker) != 1 or html.count('for(const section of sections){') != 1:
        raise ValueError('Unexpected old review structure')
    html = html.replace(marker,extra+text_html+marker)
    html = html.replace('for(const section of sections){',
                        "for(const section of sections){\n  if(!section.querySelector('video'))continue;")
    review_id = secrets.token_hex(16)
    html = html.replace(old['review_id'],review_id)
    html = html.replace('第7组看人脸裁剪回贴。','第7组看人脸裁剪回贴；第8组新增完整参考视频；最后是原尺寸文字专项。')
    (public/'review.html').write_text(html,encoding='utf8')
    receipt = {'status':'eight_short_pairs_and_static_text_human_pending','review_id':review_id,
        'pairs':8,'public':str(public),'sections':sections,'previous_review_sha256':digest_file(html_path),
        'video_audit_sha256':digest_file(video_audit),'text_audit_sha256':digest_file(text_root/'independent-text-audit.json'),
        'text_root':str(text_root),'review_html_sha256':digest_file(public/'review.html'),
        'limits':'Seven earlier pairs reused byte-for-byte with unchanged A/B labels. No new generation from review builder, no long tests, no browser action, no human acceptance.'}
    write_new_json(output/'private-receipt.json',receipt)
    return {k:v for k,v in receipt.items() if k != 'sections'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','video-audit','text-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.previous,args.video_audit,args.text_root,args.output),ensure_ascii=False))
