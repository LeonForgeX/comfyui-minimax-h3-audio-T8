"""Bind completed independent audits into a new local five-track review."""
import argparse
import json
from pathlib import Path

from tools.build_candidate_combined_review import PROJECT, build, digest


def voice_review_configuration(ablation_audit):
    """Keep v1 immutable; opt in to a separately qualified reference control cohort."""
    if ablation_audit is None:
        return 'five-track-review-input-v1.json', [
            ('native-voice-media-audit-v2', 'voice_neutral_00001_', 'A：普通平静对白'),
            ('native-voice-media-audit-v2', 'voice_emotion_00001_', 'B：轻喜悦／克制释然')]
    if ablation_audit != 'native-voice-ablation-media-audit-v1':
        raise ValueError('Use the explicitly bound reference-ablation audit')
    return 'five-track-review-input-v2.json', [
        (ablation_audit, 'voice_reference_with_00001_', 'A：普通平静对白／有音色参考'),
        ('native-voice-media-audit-v2', 'voice_emotion_00001_', 'B：轻喜悦／克制释然／有参考'),
        (ablation_audit, 'voice_reference_without_00001_', 'C：普通平静对白／无音色参考')]


def review_manifest_name(ablation_audit, joint_audit, long_audit=None, window_audit=None, dialogue_owner_audit=None):
    name, _ = voice_review_configuration(ablation_audit)
    if dialogue_owner_audit is not None:
        if (dialogue_owner_audit != 'native-voice-dialogue-owner-media-audit-v1'
                or window_audit != 'native-voice-window-media-audit-v1'
                or long_audit != 'native-voice-long-media-audit-v1'
                or joint_audit != 'native-voice-dual-joint-media-audit-v2' or not ablation_audit):
            raise ValueError('Dialogue owner requires the explicitly bound fresh v6 cohort')
        return 'five-track-review-input-v6.json'
    if window_audit is not None:
        if (window_audit != 'native-voice-window-media-audit-v1'
                or long_audit != 'native-voice-long-media-audit-v1'
                or joint_audit != 'native-voice-dual-joint-media-audit-v2' or not ablation_audit):
            raise ValueError('Window text requires the explicitly bound fresh v5 cohort')
        return 'five-track-review-input-v5.json'
    if long_audit is not None:
        if long_audit != 'native-voice-long-media-audit-v1' or joint_audit != 'native-voice-dual-joint-media-audit-v2' or not ablation_audit:
            raise ValueError('Single-model candidate requires the explicitly bound fresh v4 cohort')
        return 'five-track-review-input-v4.json'
    if joint_audit is None:
        return name
    if not ablation_audit or joint_audit not in ('native-voice-dual-joint-media-audit-v1', 'native-voice-dual-joint-media-audit-v2'):
        raise ValueError('Joint candidate requires the explicit fresh v3 review cohort')
    return 'five-track-review-input-v3.json'


def validate_single_model_terminal(receipt):
    if (receipt.get('status') != 'actual_native_reference_single_model_two_segment20_8s_complete_not_human'
            or receipt.get('single_model') is not True or receipt.get('query_route') != 'joint_av_exp'
            or receipt.get('completed_segments') != 2 or receipt.get('human_qualified') is not False
            or receipt.get('sources_unchanged') is not True):
        raise ValueError('Complete independent single-model mechanical evidence is required')
    if 'resumed_accepted_segments_before' not in receipt:
        if receipt.get('actual_forwards') != 40:
            raise ValueError('A fresh two-segment probe must have 40 actual forwards')
        return False
    if (receipt.get('actual_forwards') != 20 or receipt.get('resumed_accepted_segments_before') != 1
            or receipt.get('reused_first_segment_bytes_unchanged') is not True
            or receipt.get('assets_and_Core_unchanged') is not True
            or receipt.get('prior_job_actual_forwards', 0) < 20
            or receipt.get('all_jobs_actual_forwards') != receipt['prior_job_actual_forwards'] + 20
            or receipt.get('discarded_timeout_partial_forwards') != receipt['prior_job_actual_forwards'] - 20):
        raise ValueError('Keep exact prefix preservation and honest discarded-compute evidence')
    return True


def validate_window_text_terminal(receipt):
    _validate_controlled_text_terminal(receipt, 'accepted_window_text_exp')


def validate_dialogue_owner_terminal(receipt):
    _validate_controlled_text_terminal(receipt, 'dialogue_start_owner_exp')


def _validate_controlled_text_terminal(receipt, policy):
    validate_single_model_terminal(receipt)
    if (receipt.get('window_text_policy') != policy
            or receipt.get('prior_job_controlled_stop') is not True
            or receipt.get('prior_job_actual_forwards') != 20
            or receipt.get('all_jobs_actual_forwards') != 40
            or receipt.get('discarded_timeout_partial_forwards') != 0):
        raise ValueError('Require the controlled20 plus resumed20 window probe, not an inherited qualification')


def validate_window_words(receipt, media_sha):
    if (receipt.get('status') != 'complete_offline_CPU_ASR_observations_not_human_acceptance'
            or receipt.get('input_and_model_bytes_unchanged') is not True
            or receipt.get('human_qualified') is not False or receipt.get('GPU_used') is not False
            or any(receipt.get('configuration', {}).get(key) is not None
                   for key in ('initial_prompt', 'prefix', 'hotwords'))):
        raise ValueError('Require fresh offline no-hint words evidence, not human acceptance')
    rows = [row for row in receipt.get('media', [])
            if row.get('sha256') == media_sha and row.get('reference_recording') is False]
    if len(rows) != 1 or len(rows[0].get('observations', [])) != 2:
        raise ValueError('Words evidence must bind the exact completed candidate')
    return [item['text'] for item in rows[0]['observations']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-ablation-audit', choices=['native-voice-ablation-media-audit-v1'])
    parser.add_argument('--joint-route-audit', choices=['native-voice-dual-joint-media-audit-v1', 'native-voice-dual-joint-media-audit-v2'])
    parser.add_argument('--single-model-audit', choices=['native-voice-long-media-audit-v1'])
    parser.add_argument('--single-model-terminal', choices=['native-voice-long-gpu-v1', 'native-voice-long-gpu-v2'], default='native-voice-long-gpu-v1')
    parser.add_argument('--window-text-audit', choices=['native-voice-window-media-audit-v1'])
    parser.add_argument('--dialogue-owner-audit', choices=['native-voice-dialogue-owner-media-audit-v1'])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Never overwrite an existing bound review')
    evidence = PROJECT / 'artifacts/five-track-development-20260918'
    manifest_name, voice_configuration = voice_review_configuration(args.reference_ablation_audit)
    manifest_name = review_manifest_name(args.reference_ablation_audit, args.joint_route_audit,
                                         args.single_model_audit, args.window_text_audit, args.dialogue_owner_audit)
    if args.reference_ablation_audit:
        ablation = json.loads((evidence / 'native-voice-ablation-gpu-v1/terminal.json').read_text(encoding='utf8'))
        assert ablation['status'] == 'actual_native_Ref2VA_reference_ablation_two73frame_generated_AV_complete_not_human'
        assert ablation['human_qualified'] is False

    def relative(path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(evidence):
            raise ValueError('Use only this task evidence')
        return path.relative_to(PROJECT).as_posix()

    def entry(audit_name, stem, label, *, audio=True):
        receipt = evidence / audit_name / 'report.json'
        report = json.loads(receipt.read_text(encoding='utf8'))
        assert report['status'] == 'complete_media_integrity_not_human_quality'
        assert report['original_files_unchanged'] and report['human_qualified'] is False
        rows = [row for row in report['media'] if Path(row['path']).stem == stem]
        if len(rows) != 1:
            raise ValueError('Expected exactly one independently audited media')
        row = rows[0]
        assert row['all_video_decoded']
        if audio:
            assert row['all_audio_decoded'] and row['audio_blocks'] > 0
        else:
            assert row['audio_absent'] and row['audio_blocks'] == 0
        return dict(label=label, path=relative(row['path']), sha256=row['sha256'],
                    audit=relative(receipt), audit_sha256=digest(receipt),
                    poster=dict(path=relative(row['poster']), sha256=digest(Path(row['poster']))),
                    last=dict(path=relative(row['last']), sha256=digest(Path(row['last']))))

    dual = json.loads((evidence / 'native-voice-dual-gpu-v2/terminal.json').read_text(encoding='utf8'))
    assert dual['status'] == 'actual_native_reference_two_segment20plus4_8s_complete_not_human'
    assert dual['actual_forwards'] == 48 and dual['completed_segments'] == 2
    groups = [
        dict(id='avatar', title='1. Avatar 录音驱动：原生8步 vs 渐进4+4',
             note='同首帧／录音／seed，73帧／24fps／512×768。两路明确交付原录音；右路录音 latent 同时参与采样，不是仅贴音轨。看人物、口型、动作和画质。右路是新的独立进程 TAEH3 候选完整成片；不是旧 v6 失败预览对照。此单段不评价长视频接缝。',
             clips=[entry('avatar-v6-media-audit-v1', 'native8_control_00001_', 'A：原生8步'),
                    entry('avatar-fresh-media-audit-v1', 'Avatar_fresh_preview_73f', 'B：录音驱动 LOW4→learned3D→HIGH4')]),
        dict(id='voice', title='2. 原生音色参考：普通／轻喜悦' + ('／无参考对照' if args.reference_ablation_audit else ''),
             note='人物图片、声音参考与 seed 相同，各20次真实 Stock20 前向。目标新对白都是“你终于回来了。”；参考录音只作音色条件，保存的是原生生成声音。两路声音逐个听：中文清晰度、是否噪音／过轻、音色身份、轻喜悦是否有区别及口型。英文参考→中文生成属跨语言实验，不保证克隆或逐字。此单段接缝不适用。',
             clips=[entry(audit, stem, label) for audit, stem, label in voice_configuration]),
        dict(id='voice-dual', title='3. 原生音色＋独立二采：两段内循环8秒',
             note='同一长视频的连续两段，不是两条独立短片。各段原生一采20步完整生成AV→原learned3D→HIGH4；总48次真实前向，192帧。只用 reference 音色、不贴原录音；H4/C2 是计算分块，不是空间采样tile。Global固定身份／场景，Local依次“你终于回来了。”／“这次别再走了。”／安静聆听。Timeline百分比0–35/35–70/70–100，输入193自动对齐为209帧计划：事件约0–3.042/3.042–6.083/6.083–8.708秒，交付裁成8秒。重点拖到5.17秒看背景／人物／颜色是否跳变，听对白与环境声音是否正常；不把机械48步／能解码当作接缝通过。',
             clips=[entry('native-voice-dual-media-audit-v2', Path(dual['media']['path']).stem, '原生 reference-only 20+4，8秒完整成片')]),
        dict(id='meridian', title='4. Meridian P2：原生 ConvRot INT8 最小可行性',
             note='正确已授权 Omega1B512 真实几何、73帧真实warp、250层 native INT8／3次真实前向，原潜空间同权重完整VAE解码；没有原全精度 Meridian 测试模型。镜头轻向右平移，人物参考冻结；上游此最小链没有音频，“声音／口型／接缝”可选不适用。输出864×1184是上游训练bucket与明确FOV裁切，不拉伸，也不是完整2:3原图全画布。看身份／空间几何／镜头跟随与伪影；不是官方原随机流的逐值复现。P3编辑器／完整公开节点及P4性能验证仍保留本组P2人审门禁，不冒称都完成。',
             clips=[entry('meridian-media-audit-v1', 'Meridian_native_INT8_73f', 'Meridian 原生INT8完整73帧', audio=False)]),
    ]
    if args.reference_ablation_audit:
        groups[1]['note'] = ('A与C是新的一组有／无参考受控对照：相同首帧、seed、Stock20、目标对白“你终于回来了。”，'
                            '只删除音色音频连接与<Audio 1>绑定说明；实际20次前向音频条件数分别1和0。'
                            'B是之前独立生成的轻喜悦样片，不是这次同一批次的受控对照。'
                            '所有音轨都是原生新生成，不贴源录音。有参考路整体幅度明显较低，未认定音色效果通过。'
                            '逐个听中文、噪音／音量、音色、情绪，另看口型；英文参考→中文属跨语言实验。'
                            'CPU识别只是辅助文字证据，不代表音色／情绪／口型通过。单段接缝不适用。')
    if args.joint_route_audit:
        joint = json.loads((evidence / 'native-voice-dual-joint-gpu-v1/terminal.json').read_text(encoding='utf8'))
        assert joint['status'] == 'actual_native_reference_two_segment20plus4_8s_complete_not_human'
        assert joint['query_route'] == 'joint_av_exp' and joint['actual_forwards'] == 48 and joint['completed_segments'] == 2
        assert joint['human_qualified'] is False and joint['sources_unchanged']
        groups[2]['title'] = '3. 原生音色＋独立二采：视频路由 vs 联合音画路由'
        groups[2]['clips'][0]['label'] = 'A：旧视频路由基线／ASR发现重复对白，未通过时序'
        groups[2]['clips'].append(entry(args.joint_route_audit, Path(joint['media']['path']).stem,
                                       'B：显式 joint_av_exp 两段8秒候选／待人审'))
        groups[2]['note'] += ('\nA的ASR在约4.04–6.4秒再次识别第一句，只保留为未通过时序的对照。'
                              'B只显式增加已有Query Route节点，以原生音频时间轴加入相同事件偏置；'
                              '旧节点默认、采样步数/sigma/learned3D/遮罩/音频交付/接缝不改。'
                              '该音频扩展未被Prompt Relay论文验证，不自动保证逐字或精确事件时间；'
                              '两个候选都必须从头听，特别检查尾部是否仍重复、应安静处有无多余对白。')
    if args.single_model_audit:
        long = json.loads((evidence / args.single_model_terminal / 'terminal.json').read_text(encoding='utf8'))
        resumed = validate_single_model_terminal(long)
        groups.append(dict(id='voice-long', title='5. 原生音色＋单模型：两段内循环8秒',
            note='独立单模型原生Stock20内循环，两个成片段各配置20步，没有learned3D或HIGH二采。'
                 '人物图／参考录音／新对白／seed与上述双采实验相同，显式joint_av_exp，仅H4/C2计算分块。'
                 '交付原生新生成音轨，不贴参考录音；192帧512×768，接缝仍约5.17秒。'
                 '检查两句“你终于回来了。”、“这次别再走了。”，随后应安静；有无重复、噪音、过轻、'
                 '身份／情绪／口型／背景或颜色跳变。单模型没有独立二采的质量资格，未经人审，不当正式推荐。'
                 '沿用193→209帧Relay计划和最终8秒裁切，不保证精确对白时间。'
                 + (f'\n首次作业在{long["prior_job_actual_forwards"]}次前向后触及超时保护；'
                    f'保留第一段／上下文原字节，新作业只补第二段20次真实前向。'
                    f'跨作业实际{long["all_jobs_actual_forwards"]}次计算，其中{long["discarded_timeout_partial_forwards"]}次未完成段废弃；'
                    '不能把整次实验写成只算了40次或首次超时通过。' if resumed else '\n独立新作业实际40次前向完成；不代表质量已通过。'),
            clips=[entry(args.single_model_audit, Path(long['media']['path']).stem,
                         '原生单模型20步／显式联合音画路由／待人审')]))
    if args.window_text_audit:
        window = json.loads((evidence / 'native-voice-window-gpu-v2/terminal.json').read_text(encoding='utf8'))
        validate_window_text_terminal(window)
        clip = entry(args.window_text_audit, Path(window['media']['path']).stem,
                     'B：仅窗口文本策略 EXP／待人审')
        if clip['sha256'] != window['media']['sha256']:
            raise ValueError('Window media audit must bind the actual completed GPU result')
        words_path = evidence / 'native-voice-window-words-cpu-v1/report.json'
        texts = validate_window_words(json.loads(words_path.read_text(encoding='utf8')), clip['sha256'])
        group = groups[-1]
        group['title'] = '5. 单模型两段8秒：旧全部文本 vs 窗口文本实验'
        group['clips'][0]['label'] = 'A：旧全部文本／ASR发现第二句重复，不推荐'
        group['clips'].append(clip)
        group['note'] += ('\nB保持原193→209计划、三Local事件、seed、Stock20、AV上下文、H4C2和接缝；'
                          '只显式过滤不与当前交付窗口相交的局部文本，跨段事件原时间和sigma不改。'
                          '第一段提交后在自有进程取消，再只续跑第二段；20+20次真实前向，无废弃partial。'
                          '不是论文all-key路线，也不是无重复或逐字保证。B无提示离线ASR观察：'
                          + ' / '.join(texts) + '。识别不是试听、声纹、情绪、口型或接缝验收；请两路从头听。')
        group['words_audit'] = dict(path=relative(words_path), sha256=digest(words_path))
    if args.dialogue_owner_audit:
        owner = json.loads((evidence / 'native-voice-dialogue-owner-gpu-v2/terminal.json').read_text(encoding='utf8'))
        validate_dialogue_owner_terminal(owner)
        clip = entry(args.dialogue_owner_audit, Path(owner['media']['path']).stem,
                     'B：一次台词起始段归属 EXP／ASR含额外英文，不推荐')
        if clip['sha256'] != owner['media']['sha256']:
            raise ValueError('Dialogue owner audit must bind the completed independent GPU result')
        words_path = evidence / 'native-voice-dialogue-owner-words-cpu-v1/report.json'
        texts = validate_window_words(json.loads(words_path.read_text(encoding='utf8')), clip['sha256'])
        group = groups[-1]
        previous_words = group.pop('words_audit')
        previous_clip = group['clips'][-1]
        previous_clip['label'] = 'A：仅过滤不相交文本／ASR第二句仍重复，不推荐'
        group['clips'] = [previous_clip, clip]
        group['title'] = '5. 单模型两段8秒：窗口过滤 vs 一次台词归属'
        group['note'] = (
            '这是新的独立单变量对照，不是旧全部文本A/B标签的重新使用；旧v5审片保留。'
            '两路Stock20、512×768、H4C2、193→209计划、三Local、seed、AV尾部/接缝相同。'
            'A只过滤不相交局部事件，跨5.17秒的第二句仍送给两段，ASR发现重复。'
            'B另外只在事件起始交付窗口保留其<d>字面台词；后段跨段视觉/情绪和时间/sigma不改，'
            '不重启台词。不是裁音、静音、增益或换轨；正在续接的已知音频不截断。'
            '两次自有Job首段20提交/受控取消+只续后段20，共40实际前向，无废弃partial。'
            '这不是原论文all-key方法，不保证逐字/一句完成/未标记对白。B新无提示CPU ASR观察：'
            + ' / '.join(texts) + '。识别不等于试听、音色身份、情绪、口型或接缝验收。'
            '本例B未重复读出第二句，却识别到多余英文；与参考录音部分文句相似，疑似参考内容泄漏。'
            'ASR仍可能出错，不能宣判全部根因，也不能判本组对白通过或推荐；未裁音/换轨掩盖。'
            '请从头逐个听，检查少词/多词，5.17秒边界及片尾是否再次说话。'
        )
        group['previous_words_audit'] = previous_words
        group['words_audit'] = dict(path=relative(words_path), sha256=digest(words_path))
    manifest = evidence / manifest_name
    if manifest.exists():
        raise ValueError('Never overwrite the bound review manifest')
    manifest.write_text(json.dumps(dict(groups=groups), ensure_ascii=False, indent=2), encoding='utf8')
    result = build(manifest, args.output, template_name='five_track_review.html')
    print(json.dumps(dict(review_id=result['review_id'], groups=len(groups), clips=sum(len(g['clips']) for g in groups),
                         published=False, human_qualified=False)))


if __name__ == '__main__':
    main()
