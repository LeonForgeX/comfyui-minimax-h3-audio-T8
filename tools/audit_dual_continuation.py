"""Independent CPU readback of two-segment interrupted dual-model delivery."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file
from dual_audio_chain_contract import audit_audio_stages, audit_saved_contexts


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Evidence path leaves its job directory')
    return path


def one(paths):
    values = list(paths)
    if len(values) != 1:
        raise ValueError(f'Expected exactly one evidence item; found {len(values)}')
    return values[0]


def stage(root, index, name):
    path = one((root / 'dual_stages' / f'segment_{index:05d}').glob('*/' + name + '-*.json'))
    receipt = read(path)
    data = bound(path.parent, receipt['tensor_file'])
    if sha(data) != receipt['tensor_sha256'] or receipt['stage'] != name or receipt['contract']['segment'] != index:
        raise ValueError('Stage identity or index mismatch')
    tensors = load_file(str(data), device='cpu')
    if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError('Nonfinite stage tensor')
    if [list(tensors['samples_' + key].shape) for key in ('video', 'audio')] != receipt['shapes']:
        raise ValueError('Recorded stage geometry differs from actual bytes')
    return receipt, tensors


def audit(root):
    terminal = read(root / 'terminal.json')
    if terminal['status'] != 'two_segment_resume_completed_independent_audit_and_human_pending':
        raise ValueError('Controller has not completed both attempts')
    if [item['status'] for item in terminal['attempts']] != [
            'intentional_core_interrupt_after_saved_low_and_upscale',
            'generation_completed_independent_audit_pending']:
        raise ValueError('Missing real interrupt or fresh-process generation')
    if any(item['server_stop']['owned_children_remaining'] for item in terminal['attempts']):
        raise ValueError('Owned child process remains')
    output = Path(read(root / 'output-location.json')['output']).resolve(strict=True)
    if not output.is_relative_to(Path(__file__).resolve().parents[1] / 'artifacts'):
        raise ValueError('Unexpected artifact source')
    preserved = read(root / 'before-resume.json')
    if len(preserved) != 4 or any(sha(bound(output, name)) != digest for name, digest in preserved.items()):
        raise ValueError('Resume changed saved first-pass/upscale evidence')
    manifest_path = one(output.glob('**/manifest.json'))
    chain = manifest_path.parent
    manifest = read(manifest_path)
    state = read(chain / 'in_node_loop_effects_state.json')
    segments = manifest['segments']
    if state['status'] != 'complete' or len(segments) != 2 or state['accepted_count'] != 2:
        raise ValueError('Two-segment chain is incomplete')
    summaries = []
    previous = None
    end = 0
    for index, segment in enumerate(segments):
        if segment['index'] != index or segment['timeline_start_frame'] != end:
            raise ValueError('Accepted video timeline has a gap or overlap')
        end = segment['timeline_end_frame']
        media = bound(chain, segment['video_path'])
        if sha(media) != segment['video_sha256']:
            raise ValueError('Accepted media changed')
        rlow, low = stage(chain, index, 'low_x0')
        rinput, prepared = stage(chain, index, 'high_input')
        rhigh, high = stage(chain, index, 'high_output')
        if (low['samples_video'].shape[-2:] != (16, 32)
                or high['samples_video'].shape[-2:] != (32, 64)
                or high['samples_video'].shape != prepared['samples_video'].shape):
            raise ValueError('Independent low/high stage geometry mismatch')
        audio_contract = audit_audio_stages(
            {'low_x0': rlow, 'high_input': rinput, 'high_output': rhigh},
            {'low_x0': low, 'high_input': prepared, 'high_output': high})
        for receipt in (rinput, rhigh):
            if receipt['contract']['low_tensor_sha256'] != rlow['tensor_sha256']:
                raise ValueError('High stage belongs to another first pass')
        for receipt in (rlow, rhigh):
            report = receipt['report']
            counts = report['backend']['completed_calls']
            if report['completed_network_forwards'] != 4 or not counts.get('sage:biased') or set(counts) - {'sage:biased', 'sage:unbiased'}:
                raise ValueError('Actual4+4 biased Sage route differs from requested route')
        candidate = chain / 'candidates' / f'segment_{index:05d}' / segment['candidate_id']
        effect = read(candidate / 'effects_audit.json')
        dual = effect['sampling_plan']['dual_model']
        if dual.get('low_context') and audio_contract['mode'] == 'joint_audio_with_locked_context':
            audit_saved_contexts(low, high,
                load_file(str(bound(chain, dual['low_context']['path'])), device='cpu'),
                load_file(str(bound(chain, segment['context_path'])), device='cpu'))
        if index == 0 and (not dual['low_reused'] or dual['high_reused']):
            raise ValueError('Fresh process did not resume saved first pass and compute unfinished high pass')
        if previous:
            old_segment, old_dual = previous
            low_context = bound(chain, old_dual['low_context']['path'])
            high_context = bound(chain, old_segment['context_path'])
            if (rlow['contract']['parent_high'] != old_segment['candidate_id']
                    or rlow['contract']['parent_low_sha256'] != sha(low_context)
                    or sha(high_context) != old_segment['context_sha256']):
                raise ValueError('Continuation parent context identity mismatch')
            lo = load_file(str(low_context), device='cpu')['video_tail']
            hi = load_file(str(high_context), device='cpu')['video_tail']
            if lo.shape[-2:] != (16, 32) or hi.shape[-2:] != (32, 64):
                raise ValueError('Low and high continuation tails were not independently retained')
        summaries.append({'index': index, 'low_reused': dual['low_reused'], 'high_reused': dual['high_reused'],
            'audio_contract': audio_contract,
            'low_sha256': rlow['tensor_sha256'], 'high_sha256': rhigh['tensor_sha256'],
            'backend_calls': [rlow['report']['backend']['completed_calls'], rhigh['report']['backend']['completed_calls']]})
        previous = segment, dual
    media = bound(chain, state['final_video_path'])
    if sha(media) != state['final_video_sha256'] or end != 192:
        raise ValueError('Final media hash or8-second accepted timeline differs')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(media)],
        capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    audio = next(s for s in streams if s['codec_type'] == 'audio')
    if (video['width'], video['height'], int(video['nb_read_frames']), Fraction(video['avg_frame_rate'])) != (1024, 512, 192, Fraction(24)):
        raise ValueError('Final video geometry or rational timeline differs')
    for maps in (['0:v:0'], ['0:a:0'], ['0:v:0', '0:a:0']):
        command = ['ffmpeg', '-v', 'error', '-threads', '2', '-xerror', '-err_detect', 'explode', '-i', str(media)]
        for selected in maps:
            command += ['-map', selected]
        process = subprocess.run(command + ['-f', 'null', '-'], capture_output=True, text=True)
        if process.returncode or process.stderr.strip():
            raise ValueError('Strict media decode failed: ' + process.stderr)
    return {'status': 'two_segment_resume_mechanical_pass_human_pending', 'segments': summaries,
        'preserved_checkpoint_files': preserved, 'media': {'path': str(media), 'sha256': sha(media),
            'frames': 192, 'fps': '24/1', 'video_seconds': video['duration'], 'audio_seconds': audio['duration']},
        'cuda_initialized': torch.cuda.is_initialized(),
        'scope': 'actual two-segment resume and delivery; not human quality, complete long video, or speed comparison'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new audit filename')
    torch.set_num_threads(2)
    report = audit(args.root.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
