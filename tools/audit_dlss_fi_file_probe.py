"""Read actual media and frame records again, independently of the live encoder.

Does not start a worker or amend the original receipts. Exact RGB hashes describe
worker inputs/outputs before lossy encoding, not decoded H.264 pixel identity.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.progressive_probe_control import file_identity  # noqa: E402


def check_rows(rows, source_hashes, *, rate, origin):
    count = len(source_hashes)
    if len(rows) != 2*count or count < 2:
        raise ValueError('Wrong complete output record count')
    for i,row in enumerate(rows):
        kind = 'source' if i%2 == 0 else ('tail_hold' if i == 2*count-1 else 'generated')
        if row['slot'] != i or Fraction(row['pts']) != origin+Fraction(i)/(2*rate) or row['kind'] != kind:
            raise ValueError('Record slot/time/kind disagrees with fixed continuous 2x case')
        digest = row['rgb_sha256']
        if not isinstance(digest,str) or len(digest) != 64 or any(x not in '0123456789abcdef' for x in digest):
            raise ValueError('Invalid pre-encode RGB digest')
        if kind == 'source' and digest != source_hashes[i//2]:
            raise ValueError('Recorded source RGB differs from actual source decode')
        if kind == 'tail_hold' and digest != source_hashes[-1]:
            raise ValueError('Tail hold does not retain the final source RGB')
        if kind == 'generated':
            endpoints = source_hashes[i//2:i//2+2]
            if endpoints[0] != endpoints[1] and digest in endpoints:
                raise ValueError('Motion interval duplicates an endpoint')
    return {'source':count,'generated':count-1,'cut_hold':0,'tail_hold':1}


def audit(root, ffmpeg):
    import av
    from dlss_nr_advanced import _audio_packet_digests, _audio_pcm_digests, _validate_audio_identity
    root = Path(root)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    terminal, source, result, identities = [read(n) for n in ('terminal.json','source.json','result.json','identity.json')]
    if terminal['status'] != 'actual_73_source_72_generated_1_tail_audio_exact' or terminal['server_stop']['owned_children_remaining']:
        raise ValueError('Whole-file controller did not finish/clean up')
    original, candidate = Path(source['file']['path']), root/'candidate.mp4'
    if file_identity(original) != source['file'] or file_identity(candidate) != terminal['candidate'] or terminal['candidate'] != result['media']['file']:
        raise ValueError('Original or candidate file identity changed')
    for path,identity in identities['code'].items():
        if file_identity(path) != identity:
            raise ValueError('Current code differs from the specific audited run')
    for name,identity in identities['runtime'].items():
        actual = file_identity(identity['path'])
        if any(actual[k] != identity[k] or actual[k] != result['mapped'][name][k] for k in ('path','sha256','bytes')):
            raise ValueError('Runtime file/mapped-module identity mismatch')
    count, rate, origin = source['plan']['source_count'], Fraction(source['plan']['source_rate']), Fraction(source['plan']['origin'])
    source_hashes = []
    with av.open(str(original),options={'err_detect':'explode'}) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if Fraction(frame.pts)*frame.time_base != origin+Fraction(i)/rate:
                raise ValueError('Original timestamp grid differs')
            source_hashes.append(hashlib.sha256(frame.to_ndarray(format='rgba')[...,:3].tobytes()).hexdigest())
    if len(source_hashes) != count:
        raise ValueError('Original decode count differs')
    rows = [json.loads(line) for line in (root/'frames.jsonl').read_text(encoding='utf8').splitlines()]
    counts = check_rows(rows,source_hashes,rate=rate,origin=origin)
    if counts != result['ledger']['counts'] or counts['generated'] != terminal['generated_frames']:
        raise ValueError('Controller and actual frame records disagree')
    with av.open(str(candidate),options={'err_detect':'explode'}) as container:
        if len(container.streams.video) != 1:
            raise ValueError('Non-unique candidate video stream')
        stream = container.streams.video[0]
        if Fraction(stream.average_rate) != 2*rate or stream.width != source['width'] or stream.height != source['height']:
            raise ValueError('Candidate FPS/geometry differ')
        decoded = 0
        for i,frame in enumerate(container.decode(stream)):
            if Fraction(frame.pts)*frame.time_base != origin+Fraction(i)/(2*rate):
                raise ValueError('Candidate decoded PTS grid differs')
            decoded += 1
        if decoded != 2*count or stream.duration*stream.time_base != Fraction(count)/rate:
            raise ValueError('Candidate count/duration differ')
    audio = _validate_audio_identity(_audio_packet_digests(original),_audio_packet_digests(candidate),_audio_pcm_digests(original),_audio_pcm_digests(candidate))
    strict = subprocess.run([str(ffmpeg),'-nostdin','-v','error','-xerror','-threads','1','-hwaccel','none',
                            '-i',str(candidate),'-map','0:v:0','-map','0:a?','-f','null','-'],capture_output=True,timeout=120)
    if strict.returncode or strict.stderr:
        raise ValueError('Independent FFmpeg complete AV decode failed: '+strict.stderr.decode('utf8','replace')[:1000])
    return {'status':'actual_media_records_audio_and_full_decode_pass','counts':counts,'frames':decoded,
            'fps':str(2*rate),'duration':str(Fraction(count)/rate),'audio':audio,'ffmpeg':file_identity(ffmpeg),
            'candidate':file_identity(candidate),'quality_qualified':False,'RGB_evidence':'before_lossy_encode'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = audit(args.run_root,args.ffmpeg)
    with args.report.open('x',encoding='utf8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
