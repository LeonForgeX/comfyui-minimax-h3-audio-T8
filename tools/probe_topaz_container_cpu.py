"""CPU-only AAC/PNG-MOV vs FFV1-MKV evidence; never applies Topaz enhancement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from probe_topaz_source_cpu import load_module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('install', 'definitions', 'data', 'source', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence directory')
    contract, media = load_module('topaz_contract'), load_module('topaz_media')
    runtime = contract.OfficialTopaz(args.install, args.definitions, args.data)
    args.output.mkdir(parents=True)
    ffmpeg = str(runtime.executable('ffmpeg.exe'))
    env = runtime.child_environment(os.environ)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    prefix = [ffmpeg, '-v', 'error', '-nostdin', '-n', '-protocol_whitelist', 'file,pipe']
    def run(command):
        result = subprocess.run(command, cwd=runtime.install, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=flags, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf8', 'replace')[-2000:])
        return result.stdout
    source = args.output.resolve() / 'source.mp4'
    run(prefix + ['-i', str(args.source.resolve(strict=True)), '-t', '3', '-map', '0:v:0', '-map', '0:a:0',
        '-c', 'copy', str(source)])
    outputs = {'mkv': args.output.resolve() / 'ffv1.mkv', 'mov': args.output.resolve() / 'png.mov'}
    for kind, dest in outputs.items():
        codec = ['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gbrp16le'] if kind == 'mkv' else ['-c:v', 'png', '-pix_fmt', 'rgb48be']
        run(prefix + ['-copyts', '-start_at_zero', '-i', str(source), '-map', '0:v:0', '-map', '0:a:0',
            '-vf', 'scale=512:256:flags=lanczos,format=rgb48be', '-fps_mode', 'passthrough', '-enc_time_base:v', 'demux',
            *codec, '-threads', '2', '-c:a', 'copy', str(dest)])
    def stream_digest(path, kind):
        stream = ['-map', '0:a:0', '-c:a', 'pcm_f32le', '-f', 'f32le'] if kind == 'audio' else [
            '-map', '0:v:0', '-pix_fmt', 'rgb48le', '-c:v', 'rawvideo', '-fps_mode', 'passthrough', '-f', 'rawvideo']
        command = prefix + ['-i', str(path), *stream, '-threads', '2', '-']
        process = subprocess.Popen(command, cwd=runtime.install, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, creationflags=flags)
        count, digest = 0, hashlib.sha256()
        try:
            while block := process.stdout.read(1024**2):
                count += len(block)
                digest.update(block)
            if process.wait(timeout=120):
                raise RuntimeError('CPU digest decode failed')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        return {'bytes': count, 'sha256': digest.hexdigest()}
    records = {}
    for name, path in {'source': source, **outputs}.items():
        probe = json.loads(run(media.probe_command(runtime, path, packets=True)))
        (args.output / (name + '-audio.json')).write_text(json.dumps(probe, indent=2), encoding='utf8')
        records[name] = {'file': media.file_identity(path), 'pcm': stream_digest(path, 'audio'),
            'first_audio_packet': probe['packets'][0], 'packets': len(probe['packets'])}
        if name != 'source':
            records[name]['rgb48'] = stream_digest(path, 'video')
    result = {'status': 'CPU_container_evidence_not_Topaz_inference', 'records': records,
        'mov_pcm_matches_source': records['mov']['pcm'] == records['source']['pcm'],
        'mkv_pcm_matches_source': records['mkv']['pcm'] == records['source']['pcm'],
        'lossless_containers_same_rgb48': records['mkv']['rgb48'] == records['mov']['rgb48'],
        'gpu': False, 'downloads': False}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
    if not result['mov_pcm_matches_source'] or not result['lossless_containers_same_rgb48']:
        raise RuntimeError('Proposed lossless MOV route did not preserve decoded pixels/audio')


if __name__ == '__main__':
    main()
