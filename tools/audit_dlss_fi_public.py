"""Read-only, independent re-decode of the actual public-node game qualification."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_dlss_fi_file_probe import check_rows  # noqa: E402
from tools.progressive_probe_control import file_identity  # noqa: E402


def audit(root):
    import av
    from dlss_nr_advanced import _audio_packet_digests, _audio_pcm_digests, _validate_audio_identity
    root = Path(root).resolve(strict=True)
    report = json.loads((root/'qualification.json').read_text(encoding='utf8'))
    isolation = report['isolation']
    if (report['status'] != 'completed' or isolation['status'] != 'complete' or
            not isolation['job_assigned_before_task'] or isolation['active_after_cleanup']):
        raise ValueError('Public node did not complete and clean its owned Job')
    source, target = Path(report['source']['path']), Path(report['saved_path'])
    if file_identity(source) != report['source'] or not target.parent == root:
        raise ValueError('Source/output location changed')
    target_identity = file_identity(target)
    if any(target_identity[k] != report['media']['file'][k] for k in ('sha256','bytes','mtime_ns')):
        raise ValueError('Atomic published output differs from the validated candidate')
    for path, value in report['code'].items():
        if file_identity(path) != value:
            raise ValueError('Public worker code changed since qualification')
    for name, value in report['runtime'].items():
        actual = file_identity(value['path'])
        if any(actual[k] != value[k] or actual[k] != report['mapped'][name][k] for k in ('path','sha256','bytes')):
            raise ValueError('Runtime/mapped identity differs')
    hashes = []
    with av.open(str(source),options={'err_detect':'explode'}) as container:
        stream = container.streams.video[0]
        rate, width, height = Fraction(stream.average_rate), stream.width, stream.height
        for index, frame in enumerate(container.decode(stream)):
            stamp = Fraction(frame.pts)*frame.time_base
            if index == 0:
                origin = stamp
            if stamp != origin+Fraction(index)/rate:
                raise ValueError('Original CFR grid changed')
            hashes.append(hashlib.sha256(frame.to_ndarray(format='rgba')[...,:3].tobytes()).hexdigest())
    rows = [json.loads(line) for line in (Path(report['diagnostics'])/'frames.jsonl').read_text().splitlines()]
    counts = check_rows(rows, hashes, rate=rate, origin=origin)
    if counts != report['ledger']['counts']:
        raise ValueError('Records and actual source count disagree')
    with av.open(str(target),options={'err_detect':'explode'}) as container:
        stream = container.streams.video[0]
        if (stream.width,stream.height,Fraction(stream.average_rate)) != (width,height,2*rate):
            raise ValueError('Output geometry/FPS differ')
        stamps = [Fraction(f.pts)*f.time_base for f in container.decode(stream)]
        if stamps != [origin+Fraction(i)/(2*rate) for i in range(2*len(hashes))]:
            raise ValueError('Output count or exact timestamps differ')
    audio = _validate_audio_identity(_audio_packet_digests(source),_audio_packet_digests(target),
                                     _audio_pcm_digests(source),_audio_pcm_digests(target))
    ffmpeg = Path('C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin/ffmpeg.exe')
    decoded = subprocess.run([str(ffmpeg),'-nostdin','-v','error','-xerror','-threads','1','-hwaccel','none',
        '-i',str(target),'-map','0:v:0','-map','0:a?','-f','null','-'],capture_output=True,timeout=120)
    if decoded.returncode or decoded.stderr:
        raise ValueError('Full independent FFmpeg AV decode failed')
    result = {'status':'public_node_output_independent_postflight_pass', 'counts':counts, 'fps':str(2*rate),
              'audio':audio, 'candidate':target_identity, 'quality_qualified':False, 'ffmpeg':file_identity(ffmpeg)}
    with (root/'postflight-v1.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(audit(parser.parse_args().root)))
