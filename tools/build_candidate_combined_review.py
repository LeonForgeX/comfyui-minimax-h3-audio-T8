"""Copy explicitly identified audited preview media into one allowlisted review.

No generation, re-encoding, remote assets, browser actions or acceptance decisions.
The input manifest must cite an independent receipt and exact video SHA per clip.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(manifest, output):
    output = output.resolve()
    if output.exists() or not output.is_relative_to(PROJECT / 'artifacts'):
        raise ValueError('Use a new project artifact directory')
    source = json.loads(manifest.read_text(encoding='utf-8'))
    groups, copies = [], []
    for group in source['groups']:
        clips = []
        for entry in group['clips']:
            path = (PROJECT / entry['path']).resolve(strict=True)
            receipt = (PROJECT / entry['audit']).resolve(strict=True)
            if not all(p.is_relative_to(PROJECT / 'artifacts') for p in (path, receipt)):
                raise ValueError('Review must use this candidate evidence, not unrelated media')
            if digest(path) != entry['sha256'] or digest(receipt) != entry['audit_sha256']:
                raise ValueError('Bound media or audit identity changed')
            if entry['sha256'] not in receipt.read_text(encoding='utf-8'):
                raise ValueError('Independent receipt does not identify this media')
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                                   capture_output=True, text=True, check=True)
            video = next(s for s in json.loads(probe.stdout)['streams'] if s['codec_type'] == 'video')
            if video['codec_name'] != 'h264' or video['pix_fmt'] != 'yuv420p':
                raise ValueError('Use a separately audited browser-compatible preview')
            name = f'clip-{len(copies) + 1:02d}.mp4'
            copies.append((path, name, entry['sha256']))
            clips.append({'label': entry['label'], 'url': name, 'sha256': entry['sha256'],
                'width': video['width'], 'height': video['height'], 'seconds': float(video['duration'])})
        if not 1 <= len(clips) <= 3:
            raise ValueError('One to three visible videos per group')
        groups.append({'id': group['id'], 'title': group['title'], 'note': group['note'], 'clips': clips})
    if not groups or len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Missing or duplicate review groups')
    public_manifest = {'review_id': digest(manifest), 'groups': groups}
    serialized = json.dumps(public_manifest, ensure_ascii=False).replace('<', '\\u003c')
    template = (PROJECT / 'tools/candidate_combined_review.html').read_text(encoding='utf-8')
    if template.count('MANIFEST_JSON') != 1:
        raise ValueError('Unexpected review template')
    output.mkdir(parents=True)
    public = output / 'public'
    public.mkdir()
    for original, name, sha in copies:
        shutil.copyfile(original, public / name)
        if digest(public / name) != sha:
            raise RuntimeError('Review copy identity mismatch')
    (public / 'review.html').write_text(template.replace('MANIFEST_JSON', serialized), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'delivery.json').write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return public_manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output), ensure_ascii=False))
