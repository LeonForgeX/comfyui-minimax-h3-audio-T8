"""One final short-clip review page from audited saved media; CPU only, no browser."""
import argparse
from html import escape
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'artifacts/acceleration-research-20260909'
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'tools'))
from trt_vae_build import digest_file,write_new_json  # noqa: E402
from build_progressive_exploration_review import render_sections  # noqa: E402
from build_acceleration_review import add_lipsync_questions  # noqa: E402
from prepare_trt_vae_video_media import inspect_video,encode_rgb  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_identity(entry):
    path = Path(entry['path']).resolve(strict=True)
    if not path.is_relative_to(RESEARCH) or digest_file(path) != entry['sha256']:
        raise ValueError('Review media is missing or changed')
    return {'path':str(path),'sha256':entry['sha256']}


def decode_pair(folder):
    root = RESEARCH/folder
    report = read(root/'media-audit.json')
    if report['status'] != 'complete_video_and_original_audio_verified_not_human_qualification':
        raise ValueError('Whole video/original audio audit required')
    rgb_audit = root/('independent-saved-reference-audit.json' if report['native_review_reused'] else 'independent-video-audit.json') if 'native_review_reused' in report else root/'independent-video-audit.json'
    if digest_file(rgb_audit) != report['rgb_audit_sha256']:
        raise ValueError('RGB provenance audit changed')
    return [verify_identity(report['routes'][label]) for label in ('native','trt')]


def render(title,review_id,sections):
    template = (PROJECT/'tools/progressive_review_hub_template.html').read_text(encoding='utf8')
    template = template.replace('H3 渐进采样 · 短片集中审看',escape(title))
    template = template.replace('所有视频就在这一页。每组都是 1024×512、73 帧、24fps、约 3 秒；同组模型、提示词、seed 与总求值次数相同。A/B 顺序各组独立打乱。',
        '所有对照直接显示在这一页，均为约3秒短片。每组A/B独立打乱。第1组是当前公共解码节点；第2—4组复用已测路线样片；第5组看量化；第6组看参考编码；第7组看人脸裁剪回贴。')
    template = template.replace('这是短片探索汇总，不是完整 32 秒或插帧验收。只播放当前组的一路声音；检查画面时可暂停拖动时间。建议使用 Chrome，内置浏览器此前有黑屏问题。',
        '长片测试按你的要求取消。请用Chrome：逐组播放、切换A/B声音，暂停检查脸部、细节与闪烁。这里只比较质量，不表示TRT一定更快；加入加载和保存后，本机短片总耗时没有明显收益。尚未替你判定通过或发布。')
    template = template.replace('<!-- PAIRS -->',render_sections(sections)).replace('REVIEW_ID_PLACEHOLDER',review_id)
    template = add_lipsync_questions(template,('pair-1','pair-3','pair-4','pair-5','pair-7'))
    template = template.replace('t8.progressive.exploration-human.v1','t8.trt-vae.combined-human.v1')
    template = template.replace('progressive_exploration_human_review.json','trt_vae_combined_human_review.json')
    template = template.replace('<option>都有问题</option>','<option>都有问题</option><option>看不清／无法判断</option>')
    return template


def build(output):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    from safetensors.torch import load_file,save_file
    from dlss_nr_advanced import _audio_packet_digests,_audio_pcm_digests,_validate_audio_identity,_packet_copy_video_and_audio
    torch.set_num_threads(2)
    output = Path(output).resolve()
    if output.exists() or output == RESEARCH or not output.is_relative_to(RESEARCH):
        raise ValueError('New dedicated review directory required')
    pairs = []
    timing = RESEARCH/'trt-public-timing73-gpu-v1'
    timing_audit = read(timing/'independent-public-timing-audit.json')
    if timing_audit['status'] != 'six_serial_public_output_stage_media_and_timings_verified_not_human':
        raise ValueError('Current public output-stage independent audit required')
    sources = []
    for case in ('05-native','06-trt'):
        item = next(row for row in timing_audit['media'] if row['case']==case)
        sources.append(verify_identity({'path':str(timing/case/'review.mp4'),'sha256':item['file_sha256']}))
    pairs.append(('当前公共解码节点 · 人物对白','同一已存潜空间，原生与当前TRT公共解码器。古典音乐＋“你在哪里”；音轨均直接复制原片。',sources))
    for folder,title,note in (
        ('trt-i2va-0p5mp-decode-gpu-v1','图生视频路线','此前完整解码兼容样片，无对白，不验口型。看眨眼、动作与背景，听环境声。'),
        ('trt-progressive-0p5mp-decode-gpu-v1','渐进采样路线','此前渐进6+2采样结果的同潜空间解码对照。听音乐、人声，看脸和口型。'),
        ('trt-vdn-0p5mp-decode-gpu-v1','VDN二采路线','此前VDN8步→潜空间放大→VDN4步结果的同潜空间解码对照。听音乐、人声，看口型。'),
        ('trt-w4-saved-native-0p5mp-gpu-v1','W4量化解码','同潜空间的原生与W4解码，不是FP16与W4互比。着重看皮肤、眼睛、纹理与闪烁；音轨相同。')):
        pairs.append((title,note,decode_pair(folder)))
    encoder_audit = read(RESEARCH/'trt-encoder-condition-pair-audit-v1.json')
    if encoder_audit['status'] != 'encoder_fixed_sampler_pair_evidence_pass_human_review_pending':
        raise ValueError('Actual reference-condition pair audit required')
    pairs.append(('参考图编码 · 后续生成影响','同一参考图分别经原生/TRT编码后独立生成，其余条件相同。最终均原生解码。画面和环境声不要求逐像素相同；无对白，不验口型。',
        [verify_identity(item['media']['file']) for item in encoder_audit['runs']]))
    face = RESEARCH/'trt-public-face73-gpu-v1'
    face_audit = read(face/'independent-public-face-audit.json')
    if face_audit['status'] != 'public_full_decoder_face_roundtrip_tensor_and_cleanup_pass_not_human':
        raise ValueError('Actual Full/FaceRefine audit required')
    for name,digest in face_audit['files'].items():
        if Path(name).name != name or digest_file(face/name) != digest:
            raise ValueError('Face interface tensor evidence changed')
    public = output/'public'
    public.mkdir(parents=True)
    prepared = output/'face-media'
    prepared.mkdir()
    original_report = read(RESEARCH/'trt-t2va-0p5mp-decode-gpu-v1/media-audit.json')
    source = verify_identity({'path':original_report['source'],'sha256':original_report['source_sha256']})
    original = inspect_video(source['path'])
    packets,pcm = _audio_packet_digests(source['path']),_audio_pcm_digests(source['path'])
    tensors = load_file(str(face/'outputs.safetensors'),device='cpu')
    face_sources = []
    for label in ('native','trt'):
        tensor = prepared/(label+'.safetensors')
        rgb = tensors[label+'_stitched'].movedim(-1,0).unsqueeze(0).contiguous()
        save_file({'rgb':rgb},str(tensor))
        encode_rgb(prepared/(label+'-video.mp4'),tensor,tuple(rgb.shape),original)
        final = prepared/(label+'.mp4')
        _packet_copy_video_and_audio(prepared/(label+'-video.mp4'),Path(source['path']),final)
        if inspect_video(final) != original:
            raise ValueError('Face media timing/geometry changed')
        audio = _validate_audio_identity(packets,_audio_packet_digests(final),pcm,_audio_pcm_digests(final))
        face_sources.append({'path':str(final),'sha256':digest_file(final),'audio':audio})
    del tensors,rgb
    write_new_json(output/'face-media-audit.json',{'status':'actual_roundtrip_stitch_media_audio_pass_not_face_sampling',
        'original':source,'routes':face_sources,'tensor_audit_sha256':digest_file(face/'independent-public-face-audit.json')})
    pairs.append(('人脸裁剪与回贴 · Full接口','只把真实人物短片裁剪后编码、解码、回贴，没有重新采样修脸。看脸部重建差异与闪烁；原音轨不变，遮罩外原片像素已单独核对。',face_sources))
    sections,private = [],[]
    for index,(title,note,sides) in enumerate(pairs,1):
        sides = list(sides)
        secrets.SystemRandom().shuffle(sides)
        mapping = {}
        for label,identity in zip(('A','B'),sides,strict=True):
            identity = verify_identity(identity)
            target = public/f'pair-{index}-{label}.mp4'
            shutil.copyfile(identity['path'],target)
            if digest_file(target) != identity['sha256']:
                raise ValueError('Public media copy mismatch')
            mapping[label] = identity
        sections.append({'id':f'pair-{index}','title':title,'description':note,'accepted':False})
        private.append({'id':f'pair-{index}','mapping':mapping})
    review_id = secrets.token_hex(16)
    html = render('H3 TRT VAE · 集中审片',review_id,sections)
    with (public/'review.html').open('x',encoding='utf8') as target:
        target.write(html)
    if torch.cuda.is_initialized():
        raise ValueError('Review preparation must remain CPU-only')
    receipt = {'status':'seven_short_pairs_built_human_pending','review_id':review_id,'pairs':7,
        'public':str(public),'sections':private,'cuda_initialized':False,'no_browser_action':True,
        'review_html_sha256':digest_file(public/'review.html'),
        'limits':'No long-video tests.Three prior route decoder pairs retained as explicitly labeled earlier evidence; currentpublic loader+pixelbuffer path independently exercised bypair1. No claim of new FaceRefine sampling,all-route latestGPU reruns or human acceptance.'}
    write_new_json(output/'private-receipt.json',receipt)
    return {k:v for k,v in receipt.items() if k!='sections'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),ensure_ascii=False,indent=2))
