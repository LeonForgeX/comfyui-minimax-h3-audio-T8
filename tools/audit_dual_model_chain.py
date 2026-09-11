"""CPU readback of an uninterrupted multi-segment dual-model pilot.

Does not substitute checkpoint validity for subjective seam/lipsync acceptance.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

from audit_dual_continuation import bound, one, read, sha, stage
from dual_audio_chain_contract import audit_audio_stages, audit_saved_contexts


def validate_backend(report, backend, relay):
    if report['completed_network_forwards'] != 4:
        raise ValueError('Expected exactly four completed model forwards per stage')
    actual = report['backend']
    expected = {'kj': ('audited_kj_selector', 'sage:unbiased'),
        'sol': ('audited_sol_attn_selector', 'sol:completed'),
        'pytorch': ('native_core_pytorch_selector', 'pytorch:completed')}
    kind, counter = expected[backend]
    if actual['kind'] != kind:
        raise ValueError('Unexpected measured backend owner')
    counts = actual['completed_calls']
    if relay:
        if backend != 'kj':
            raise ValueError('This auditor does not qualify Relay Sol fallback as Sol')
        if (counts.get('sage:unbiased') != 200 or counts.get('sage:biased', 0) <= 0
                or set(counts) != {'sage:biased', 'sage:unbiased'}):
            raise ValueError('Relay backend is incomplete or fell back')
    elif counts != {counter: 200}:
        raise ValueError('Actual backend count differs or fell back')


def audit(root):
    terminal = read(root / 'terminal.json')
    if (not terminal['status'].startswith('generation_completed')
            or terminal['server_stop']['owned_children_remaining']):
        raise ValueError('Generation or owned cleanup incomplete')
    duration = terminal['duration']
    if duration not in (8, 24):
        raise ValueError('Use short auditor for the one-segment case')
    output = (root / 'output').resolve(strict=True)
    manifest_path = one(output.glob('**/manifest.json'))
    chain = manifest_path.parent
    manifest = read(manifest_path)
    state = read(chain / 'in_node_loop_effects_state.json')
    segments = manifest['segments']
    if (state['status'] != 'complete' or len(segments) < 2
            or state['accepted_count'] != len(segments)):
        raise ValueError('Multi-segment chain is incomplete')
    previous = None
    end = 0
    summaries = []
    for index, segment in enumerate(segments):
        if segment['index'] != index or segment['timeline_start_frame'] != end:
            raise ValueError('Timeline gap, overlap or segment-order mismatch')
        end = segment['timeline_end_frame']
        if sha(bound(chain, segment['video_path'])) != segment['video_sha256']:
            raise ValueError('Segment media changed')
        rlow, low = stage(chain, index, 'low_x0')
        rinput, prepared = stage(chain, index, 'high_input')
        rhigh, high = stage(chain, index, 'high_output')
        if (low['samples_video'].shape[-2:] != (16, 32)
                or high['samples_video'].shape[-2:] != (32, 64)
                or high['samples_video'].shape != prepared['samples_video'].shape):
            raise ValueError('Low/high geometry mismatch')
        audio_contract = audit_audio_stages(
            {'low_x0': rlow, 'high_input': rinput, 'high_output': rhigh},
            {'low_x0': low, 'high_input': prepared, 'high_output': high})
        for receipt in (rinput, rhigh):
            if receipt['contract']['low_tensor_sha256'] != rlow['tensor_sha256']:
                raise ValueError('High stage is not bound to this low stage')
        for receipt in (rlow, rhigh):
            validate_backend(receipt['report'], terminal['backend'], terminal['relay'])
        candidate = bound(chain, 'candidates/' + f'segment_{index:05d}/' + segment['candidate_id'])
        dual = read(candidate / 'effects_audit.json')['sampling_plan']['dual_model']
        if dual['low_reused'] or dual['high_reused']:
            raise ValueError('Uninterrupted pilot unexpectedly reused stages')
        if dual.get('low_context') and audio_contract['mode'] == 'joint_audio_with_locked_context':
            audit_saved_contexts(low, high,
                load_file(str(bound(chain, dual['low_context']['path'])), device='cpu'),
                load_file(str(bound(chain, segment['context_path'])), device='cpu'))
        if previous is not None:
            old_segment, old_dual = previous
            low_context = bound(chain, old_dual['low_context']['path'])
            high_context = bound(chain, old_segment['context_path'])
            if (rlow['contract']['parent_high'] != old_segment['candidate_id']
                    or rlow['contract']['parent_low_sha256'] != sha(low_context)
                    or sha(high_context) != old_segment['context_sha256']):
                raise ValueError('Continuation context content identity mismatch')
            for path, geometry in ((low_context, (16, 32)), (high_context, (32, 64))):
                if load_file(str(path), device='cpu')['video_tail'].shape[-2:] != geometry:
                    raise ValueError('Separate low/high continuation geometry lost')
        summaries.append({'index': index, 'end_frame': end,
            'audio_contract': audio_contract,
            'stage_sha256': [rlow['tensor_sha256'], rinput['tensor_sha256'], rhigh['tensor_sha256']],
            'backend_calls': [rlow['report']['backend'], rhigh['report']['backend']]})
        previous = segment, dual
    media = bound(chain, state['final_video_path'])
    if end != duration * 24 or sha(media) != state['final_video_sha256']:
        raise ValueError('Final identity/duration differs')
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
        '-of', 'json', str(media)], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    if (video['width'], video['height'], int(video['nb_read_frames']),
            Fraction(video['avg_frame_rate'])) != (1024, 512, duration * 24, Fraction(24)):
        raise ValueError('Final decoded frame count/geometry/rational rate differs')
    for maps in (['0:v:0'], ['0:a:0'], ['0:v:0', '0:a:0']):
        command = ['ffmpeg', '-v', 'error', '-threads', '2', '-xerror', '-err_detect',
            'explode', '-i', str(media)]
        for selected in maps:
            command += ['-map', selected]
        decoded = subprocess.run(command + ['-f', 'null', '-'], capture_output=True, text=True)
        if decoded.returncode or decoded.stderr.strip():
            raise ValueError('Strict media decode failed: ' + decoded.stderr)
    return {'status': 'multi_segment_mechanical_pass_human_pending', 'segments': summaries,
        'media': {'path': str(media), 'sha256': sha(media), 'frames': end, 'fps': '24/1',
            'video_seconds': video['duration'], 'audio_seconds': audio['duration']},
        'cuda_initialized': torch.cuda.is_initialized(),
        'scope': 'one actual multi-segment route; not subjective quality or comparative speed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new audit output')
    torch.set_num_threads(2)
    result = audit(args.root.resolve(strict=True))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
