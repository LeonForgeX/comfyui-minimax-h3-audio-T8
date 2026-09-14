"""CPU-only matched previews from a completed public Topaz receipt; no UI or AI."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve(strict=True)
    output = args.output.resolve()
    artifacts = PROJECT / 'artifacts'
    if not receipt_path.is_relative_to(artifacts) or not output.is_relative_to(artifacts) or output.exists():
        raise ValueError('Use a candidate receipt and a new candidate artifact directory')
    receipt = json.loads(receipt_path.read_text(encoding='utf8'))
    if receipt.get('status') != 'public_nodes_real_execution_media_pass_human_pending':
        raise ValueError('Completed public-node media receipt required')
    report = receipt['result']
    if report.get('status') != 'media_audit_pass_human_pending':
        raise ValueError('Master media must pass before preparing review')
    source, master = Path(report['source']['path']), Path(report['output']['path'])
    if any(not p.resolve(strict=True).is_relative_to(artifacts) for p in (source, master)):
        raise ValueError('This review tool only handles isolated candidate media')
    package = importlib.util.module_from_spec(importlib.util.spec_from_loader(
        'topaz_preview_pkg', loader=None, is_package=True))
    package.__path__ = [str(PROJECT / 'h3_t8')]
    sys.modules['topaz_preview_pkg'] = package
    from topaz_preview_pkg import topaz_media as media
    if media.file_identity(source) != report['source'] or media.file_identity(master) != report['output']:
        raise ValueError('Bound source/master identity changed')
    # Media-only use of ordinary CPU FFmpeg. Enhancement already happened through
    # the official runtime; this tool never invokes tvai_up or replaces its result.
    ffmpeg = Path('C:/ProgramData/chocolatey/bin/ffmpeg.exe').resolve(strict=True)
    ffprobe = Path('C:/ProgramData/chocolatey/bin/ffprobe.exe').resolve(strict=True)
    class MediaRuntime:
        install = ffmpeg.parent

        def executable(self, name):
            return {'ffmpeg.exe': ffmpeg, 'ffprobe.exe': ffprobe}[name]
    runtime = MediaRuntime()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    output.mkdir(parents=True)
    def run(command):
        return subprocess.run(command, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            shell=False, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=900)
    def probe(path, **options):
        return json.loads(run(media.probe_command(runtime, path, **options)).stdout)
    original_video = media.analyze_video(
        probe(source, frames=True), allowed_bit_depths=media.AUTOMATIC_H264_INPUT_BIT_DEPTHS
    )
    native_video = media.analyze_video(
        probe(master, frames=True), allowed_bit_depths=media.AUTOMATIC_H264_INPUT_BIT_DEPTHS
    )
    width, height = native_video['width'], native_video['height']
    media.compare_video(original_video, native_video, width, height)
    original_audio = probe(source, packets=True)
    original_pcm = media.decoded_pcm_digests(runtime, source, original_audio, env, output / 'source_pcm.stderr')
    clips = []
    for name, input_path, filters, label in (
        ('lanczos', source, f'scale={width}:{height}:flags=lanczos:threads=2,format=yuv420p', '普通Lanczos放大'),
        ('topaz', master, 'format=yuv420p', '正式Topaz增强结果')):
        target = output / (name + '.mp4')
        command = [str(ffmpeg), '-v', 'error', '-nostdin', '-n', '-copyts', '-start_at_zero',
            '-i', str(input_path), '-map', '0:v:0', '-map', '0:a?', '-vf', filters,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-threads', '2',
            '-fps_mode', 'passthrough', '-enc_time_base:v', 'demux', '-c:a', 'copy',
            '-movflags', '+faststart', str(target)]
        run(command)
        video = media.analyze_video(
            probe(target, frames=True), allowed_bit_depths=media.AUTOMATIC_H264_INPUT_BIT_DEPTHS
        )
        audio = probe(target, packets=True)
        va = media.compare_video(original_video, video, width, height)
        aa = media.compare_audio_packets(original_audio, audio, va['common_shift'])
        aa['pcm'] = media.compare_pcm_digests(original_pcm,
            media.decoded_pcm_digests(runtime, target, audio, env, output / (name + '_pcm.stderr')))
        run([str(ffmpeg), '-v', 'error', '-xerror', '-err_detect', 'explode', '-threads', '2',
             '-i', str(target), '-map', '0:v:0', '-map', '0:a?', '-f', 'null', '-'])
        evidence = {'status': 'preview_media_pass_not_visual_acceptance', 'output': media.file_identity(target),
            'source': report['source'], 'source_master': report['output'] if name == 'topaz' else None,
            'upscale_receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            'video': va, 'audio': aa, 'command': command, 'gpu_inference_executed': False}
        audit_path = output / (name + '.audit.json')
        audit_path.write_text(json.dumps(evidence, indent=2), encoding='utf8')
        clips.append({'label': label, 'path': target.relative_to(PROJECT).as_posix(),
            'sha256': evidence['output']['sha256'], 'audit': audit_path.relative_to(PROJECT).as_posix(),
            'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest()})
    if media.file_identity(source) != report['source'] or media.file_identity(master) != report['output']:
        raise ValueError('Source/master changed while producing previews')
    result = {'status': 'matched_preview_pair_audited_human_pending', 'width': width, 'height': height,
              'clips': clips, 'source_unchanged': True, 'master_unchanged': True}
    (output / 'pair.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'status': result['status'], 'width': width, 'height': height, 'clips': len(clips)}))


if __name__ == '__main__':
    main()
