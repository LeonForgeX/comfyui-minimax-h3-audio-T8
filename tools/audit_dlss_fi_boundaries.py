"""Read-only, complete media/record validation for fixed synthetic FI cases."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.progressive_probe_control import file_identity, ResourceGuard  # noqa: E402
from tools.run_dlss_fi_boundaries import boundary_source, expected_counts  # noqa: E402
from tools.dlss_fi_media import inspect_source, decode_frames  # noqa: E402


def check_records(rows, hashes, plan):
    if len(hashes) != plan.source_count or len(rows) != plan.output_count:
        raise ValueError('Incomplete source or output records')
    counts = dict.fromkeys(('source', 'generated', 'cut_hold', 'tail_hold'), 0)
    for row, slot in zip(rows, plan.slots()):
        right = slot.source_index+1
        input_index = right if slot.kind in ('generated', 'cut_hold') else slot.source_index
        reset = input_index == 0 or input_index in plan.cuts
        if slot.kind == 'tail_hold':
            reset = False
        if (row['slot'], Fraction(row['pts']), row['kind'], row['input_index'], row['reset']) != (
                slot.index, slot.pts, slot.kind, input_index, reset):
            raise ValueError('Record timing/kind/input/reset disagrees with exact plan')
        digest = row['rgb_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid frame digest')
        if slot.kind == 'generated':
            endpoints = hashes[slot.source_index:right+1]
            if endpoints[0] != endpoints[1] and digest in endpoints:
                raise ValueError('Moving interval duplicates an endpoint')
        elif digest != hashes[slot.source_index]:
            raise ValueError('Source or declared hold changed its input RGB')
        counts[slot.kind] += 1
    if counts != expected_counts(plan):
        raise ValueError('Incomplete frame counts')
    return counts


def audit(root, manifest, ffmpeg):
    root = Path(root)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    terminal, identities, result, recorded = [read(n+'.json') for n in ('terminal', 'identity', 'result', 'source')]
    if terminal['status'] != 'actual_boundary_complete' or terminal['server_stop']['owned_children_remaining']:
        raise ValueError('Controller did not complete and clean up')
    if file_identity(manifest) != identities['manifest']:
        raise ValueError('Fixture manifest changed')
    source = boundary_source(manifest, terminal['case'])
    if source['file'] != recorded['file']:
        raise ValueError('Recorded source is not the fixed fixture')
    for path, identity in identities['code'].items():
        if file_identity(path) != identity:
            raise ValueError('Recorded execution code changed')
    for name, identity in identities['runtime'].items():
        actual = file_identity(identity['path'])
        if any(actual[k] != identity[k] or actual[k] != result['mapped'][name][k] for k in ('path', 'bytes', 'sha256')):
            raise ValueError('Actual runtime/mapped identity differs')
    hashes = [hashlib.sha256(frame[..., :3].tobytes()).hexdigest() for frame in decode_frames(source)]
    rows = [json.loads(s) for s in (root/'frames.jsonl').read_text(encoding='utf8').splitlines()]
    counts = check_records(rows, hashes, source['plan'])
    if counts != terminal['counts'] or counts != result['ledger']['counts']:
        raise ValueError('Live and independent counts disagree')
    candidate = inspect_source(root/'candidate.mp4')
    p, q = source['plan'], candidate['plan']
    if (q.source_count, q.source_rate, q.origin, q.duration) != (p.output_count, p.target_rate, p.origin, p.duration):
        raise ValueError('Candidate time grid/count differs')
    if candidate['file'] != terminal['candidate'] or candidate['file'] != result['media']['file']:
        raise ValueError('Candidate identity differs')
    if candidate['audio_packets'] or candidate['audio_pcm']:
        raise ValueError('Silent fixture acquired an audio track')
    if (candidate['width'], candidate['height'], candidate['colors']) != (source['width'], source['height'], source['colors']):
        raise ValueError('Candidate geometry/color metadata differs')
    guard = ResourceGuard()
    guard.observe(read('startup.json'), startup=True)
    resources = [json.loads(s) for s in (root/'resources.jsonl').read_text(encoding='utf8').splitlines()]
    for sample in resources:
        guard.observe(sample)
    if guard.reason or not resources:
        raise ValueError('Raw resource monitoring failed or is missing')
    decoded = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-threads', '1',
        '-hwaccel', 'none', '-i', str(root/'candidate.mp4'), '-map', '0:v:0', '-f', 'null', '-'],
        capture_output=True, timeout=120)
    if decoded.returncode or decoded.stderr:
        raise ValueError('Complete independent FFmpeg decode failed')
    return {'status': 'fixed_boundary_media_and_records_pass', 'case': terminal['case'], 'counts': counts,
            'candidate': candidate['file'], 'fps': str(q.source_rate), 'duration': str(q.duration),
            'ffmpeg': file_identity(ffmpeg), 'quality_qualified': False, 'has_audio': False,
            'resource_samples': len(resources), 'source_kind': 'CPU_synthetic_not_H3_generation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit(args.run_root, args.manifest, args.ffmpeg)
    with args.report.open('x', encoding='utf8') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
