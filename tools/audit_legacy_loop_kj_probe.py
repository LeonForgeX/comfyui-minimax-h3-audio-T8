"""Independent old single-MODEL Stock20/Relay/KJ evidence, CPU media checks only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.build_candidate_combined_review import digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_effects(data, state, segment):
    unsigned = dict(data)
    claimed = unsigned.pop('audit_sha256')
    actual = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if claimed != actual or data['contract_sha256'] != state['contract_sha256']:
        raise ValueError('Effects audit identity mismatch')
    if data['segment_index'] != 0 or data['candidate_id'] != segment['candidate_id']:
        raise ValueError('Effects audit is bound to a different segment')
    eav = data['enhance_a_video_audit']
    if (eav['status'] != 'apply_exp_long_video_segment_verified' or eav['aborted']
            or eav['model_forward_count'] != 20 or eav['attention_measurement_count'] != 1000
            or eav['attention_calls_per_active_forward'] != [50] * 20
            or data['sampling_plan']['second_pass_nfe'] != 0):
        raise ValueError('Old loop did not complete the declared single Stock20 schedule')
    backend = eav['config']['composed_attention_backend']
    calls, memory = backend['completed_calls'], backend['memory_composition']
    if (backend['kind'] != 'audited_kj_selector' or set(calls) != {'sage:biased', 'sage:unbiased'}
            or min(calls.values()) <= 0 or memory['head_chunks'] != 4
            or memory['ffn_settings'] != [2, 4096] or memory['kind'] != 'kj_memory_sage'):
        raise ValueError('KJ/Relay memory composition is missing or fell back')
    return backend


def audit(root):
    root = root.resolve(strict=True)
    terminal = read(root / 'terminal.json')
    if (not terminal['status'].startswith('generation_completed')
            or terminal['server_stop']['owned_children_remaining']
            or terminal['resources']['status'] != 'observations_within_policy'):
        raise ValueError('Generation/cleanup/resource checks incomplete')
    graph = read(root / 'generation/prompt.json')
    node = graph['8']
    if (node['class_type'] != 'MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced'
            or node['inputs']['model'] != ['26', 0]
            or any('LoRA' in value['class_type'] or 'DualModel' in value['class_type']
                   for value in graph.values())
            or graph['21']['inputs']['model'] != ['1', 0]):
        raise ValueError('Not the declared old public node without Turbo')
    chain = root / 'output/minimax_h3_t8_long_video' / node['inputs']['chain_id']
    state, manifest = read(chain / 'in_node_loop_effects_state.json'), read(chain / 'manifest.json')
    if state['status'] != 'complete' or state['accepted_count'] != 1 or len(manifest['segments']) != 1:
        raise ValueError('One complete accepted segment required')
    segment = manifest['segments'][0]
    paths = list(chain.rglob('effects_audit.json'))
    if len(paths) != 1:
        raise ValueError('Ambiguous effects receipt')
    backend = check_effects(read(paths[0]), state, segment)
    accepted = (chain / segment['video_path']).resolve(strict=True)
    media = Path(state['final_video_path']).resolve(strict=True)
    if (not accepted.is_relative_to(chain) or not media.is_relative_to(chain)
            or digest(accepted) != segment['video_sha256'] or digest(media) != state['final_video_sha256']):
        raise ValueError('Accepted/assembled media identity mismatch')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(media)],
                           capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    if (video['width'], video['height'], video['nb_read_frames'], video['avg_frame_rate']) != (512, 256, '72', '24/1'):
        raise ValueError('Old loop output geometry or timeline changed')
    for maps in (['0:v:0'], ['0:a:0'], ['0:v:0', '0:a:0']):
        command = ['ffmpeg', '-v', 'error', '-threads', '2', '-xerror', '-err_detect', 'explode', '-i', str(media)]
        for item in maps:
            command += ['-map', item]
        result = subprocess.run([*command, '-f', 'null', '-'], capture_output=True, text=True)
        if result.returncode or result.stderr.strip():
            raise ValueError('Strict media decode failed')
    return {'status': 'legacy_single_loop_mechanical_pass_human_pending',
        'node': node['class_type'], 'actual_forwards': 20, 'eav_measurements': 1000,
        'backend': backend, 'effects_sha256': digest(paths[0]), 'no_turbo_or_second_model': True,
        'media': {'path': str(media), 'sha256': digest(media), 'frames': 72, 'width': 512, 'height': 256,
                  'fps': '24/1', 'video_seconds': video['duration'], 'audio_seconds': audio['duration']},
        'strict_decode': 'video_audio_joint_all_pass',
        'scope': 'old public single-model Stock20 combined route; no upscale, no new long-video/quality/speed claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
