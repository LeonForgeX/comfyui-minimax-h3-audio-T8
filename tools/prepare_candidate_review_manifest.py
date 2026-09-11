"""Select only named, independently audited H3 cases for collective review."""
import argparse
import json
from pathlib import Path

from tools.build_candidate_combined_review import PROJECT, digest


def clip(folder, label, audit='independent-audit-v1.json'):
    receipt = PROJECT / 'artifacts' / folder / audit
    data = json.loads(receipt.read_text(encoding='utf-8'))
    if data['status'] not in {'short_mechanical_audit_pass_human_pending',
                             'legacy_single_loop_mechanical_pass_human_pending',
                             'multi_segment_mechanical_pass_human_pending',
                             'two_segment_resume_mechanical_pass_human_pending'}:
        raise ValueError('Case lacks its independent completed-media audit')
    media = data['media']
    path = Path(media['path']).resolve(strict=True)
    if not path.is_relative_to(PROJECT / 'artifacts') or digest(path) != media['sha256']:
        raise ValueError('Audited case media is missing or changed')
    return {'label': label, 'path': path.relative_to(PROJECT).as_posix(), 'sha256': media['sha256'],
            'audit': receipt.relative_to(PROJECT).as_posix(), 'audit_sha256': digest(receipt)}


def manifest():
    return {'scope': 'H3 machine-qualified candidate cases; Topaz2x/Starlight still unavailable, not full goal completion',
        'groups': [
        {'id': 'long24', 'title': '1. 24秒六段 · 事件时间线与接缝',
         'note': '这条确实是24秒，不是3秒短片。预期先说“你在哪里”，随后安静倾听、转头、走向门边并停下。请特别检查重复对白、字幕和约5.17、9.42、13.67、17.92、22.17秒接缝。全片机械检查通过，不代表剧情已跟随。',
         'clips': [clip('dual-relay-long24-gpu-v1', 'Relay · 4+放大+4')]},
        {'id': 'backend', 'title': '2. 同条件后端对照 · 原生、KJ、Sol',
         'note': '同底模、EMA B、提示词、seed、1024×512和4+4。古典音乐＋“你在哪里”。三条独立生成，逐一选声音看口型。实际后端各自200+200调用已审计；不是Relay/Sol兼容性试验。',
         'clips': [clip('dual-backend-benchmark-pytorch-gpu-v1', '原生PyTorch', 'audit-cold.json'),
                   clip('dual-backend-benchmark-kj-gpu-v1', 'KJ Sage', 'audit-cold.json'),
                   clip('dual-backend-benchmark-sol-gpu-v1', 'Sol', 'audit-cold.json')]},
        {'id': 'memory', 'title': '3. KJ省显存组合 · 同条件分块对照',
         'note': '两条均为省显存Sage＋Relay、4+4。右边额外head4/FFN2。请比较人物、音色、噪声、嘴型和闪烁；不能仅根据调用成功就认定质量一致。',
         'clips': [clip('dual-kj-memory-relay-gpu-v1', 'head1／无FFN分块'),
                   clip('dual-memory-head4-ffn2-gpu-v1', 'head4／FFN2')]},
        {'id': 'distinct', 'title': '4. 独立二采底模 · FL2VA与Ref2VA',
         'note': '左边两采共用FL2VA底模；右边一采FL2VA、二采Ref2VA，各自EMA B。检查模型切换后的画面和声音。右边使用关闭锁页缓存的独立启动配置；本组不是公平速度对照，也不证明任意底模组合。',
         'clips': [clip('dual-backend-benchmark-kj-gpu-v1', '同底模独立LoRA', 'audit-cold.json'),
                   clip('dual-distinct-base-unpinned-gpu-v1', '独立FL2VA／Ref2VA底模')]},
        {'id': 'eav', 'title': '5. EAV＋Relay＋KJ省显存组合',
         'note': '一采不挂Turbo，用原始20步；二采EMA B 4步。head4/FFN2。不是4步Turbo启用EAV。请看人物、听古典音乐和对白，检查口型。',
         'clips': [clip('dual-eav-stock20-memory-gpu-v1', 'Stock20＋放大＋EMA4')]},
        {'id': 'resume', 'title': '6. 两段8秒 · 中断后续跑',
         'note': '真实中断后在新进程恢复，一采与二采LoRA强度为1／0.9。请检查约5.17秒的接缝以及声音连续性。缓存未重用错位已由文件身份检查，不由画面推断。',
         'clips': [clip('dual-continuation-resume-gpu-v2', '8秒恢复成片')]},
        {'id': 'i2va', 'title': '7. 首帧参考 · 图生视频',
         'note': '真实首帧分别进入低／高分辨率条件；3秒4+4。提示词同样要求古典音乐与“你在哪里”。请实际听看：有对白才评价口型，没有说出来则记录对白未跟随。不是长片I2VA验证。',
         'clips': [clip('dual-i2va-reference-gpu-v1', 'I2VA双模型候选', 'independent-audit-v2.json')]},
        {'id': 'game', 'title': '8. 游戏场景／角色与纹理',
         'note': '3秒4+4游戏素材，检查盔甲、高频细节和运动，听环境声是否异常。没有合格的配对基线，不借这一条宣称比其他方案更好。',
         'clips': [clip('dual-game-short-gpu-v1', '游戏候选')]},
        {'id': 'legacy', 'title': '9. 旧单模型内循环 · KJ＋Relay＋EAV',
         'note': '直接运行旧节点，不是新双模型替代。512×256、原始20步、不挂Turbo，KJ省显存head4/FFN2。此低分辨率样片用于确认旧组合能实际生成和声音是否正常，不能当作高清细节对照。',
         'clips': [clip('legacy-loop-kj-eav-gpu-v1', '旧内循环Stock20')]},
    ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(manifest(), stream, ensure_ascii=False, indent=2)
