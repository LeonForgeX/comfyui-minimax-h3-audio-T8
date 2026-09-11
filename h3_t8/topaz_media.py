"""CPU media contracts for official Topaz; no FFmpeg execution at import."""
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('File changed during identity scan')
    return {'path': str(path), 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def probe_command(runtime, source, *, frames=False, packets=False):
    source = Path(source).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in ('.mp4', '.mkv', '.mov', '.webm', '.avi'):
        raise ValueError('Use an existing local video container, not a URL/playlist')
    args = [str(runtime.executable('ffprobe.exe')), '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-of', 'json', '-show_streams', '-show_format']
    if frames:
        args += ['-select_streams', 'v:0', '-show_frames', '-show_entries',
            'frame=best_effort_timestamp,width,height,pix_fmt,interlaced_frame']
    if packets:
        args += ['-select_streams', 'a', '-show_packets', '-show_data_hash', 'sha256', '-show_entries',
            'packet=stream_index,pts,duration,size,data_hash,side_data_list']
    return args + [str(source)]


def analyze_video(probe):
    videos = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
    if len(videos) != 1:
        raise ValueError('Exactly one video stream is required')
    stream = videos[0]
    width, height = stream.get('width'), stream.get('height')
    if any(type(x) is not int or not 32 <= x <= 8192 for x in (width, height)):
        raise ValueError('Unsupported source dimensions')
    if stream.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('HDR is not qualified for this SDR Topaz route')
    if stream.get('field_order', 'progressive') not in ('progressive', 'unknown'):
        raise ValueError('Deinterlace source explicitly before this progressive route')
    if Fraction(stream.get('sample_aspect_ratio', '1:1').replace(':', '/')) != 1:
        raise ValueError('Non-square pixels need explicit normalization before enhancement')
    for side in stream.get('side_data_list', []):
        if side.get('rotation', 0) != 0:
            raise ValueError('Normalize rotation before enhancement')
    rate = Fraction(stream.get('r_frame_rate', '0/1'))
    tick = Fraction(stream.get('time_base', '0/1'))
    if rate <= 0 or rate > 240 or tick <= 0 or tick * rate >= Fraction(1, 4):
        raise ValueError('Invalid or insufficiently precise video clock')
    frames = probe.get('frames', [])
    if len(frames) < 2:
        raise ValueError('Video needs at least two decoded frames')
    timestamps = []
    for frame in frames:
        if (frame.get('width'), frame.get('height')) != (width, height) or frame.get('interlaced_frame', 0):
            raise ValueError('Variable size or interlaced frames are not supported')
        pts = frame.get('best_effort_timestamp')
        if type(pts) is not int:
            raise ValueError('Every decoded frame needs an integer presentation timestamp')
        timestamps.append(pts * tick)
    origin = timestamps[0]
    # Container clocks may quantize CFR (e.g. MKV millisecond ticks at24fps).
    # One tick accounts only for the difference of two rounded timestamps;
    # dropped/duplicated frames are not tolerated or silently repaired.
    for i, pts in enumerate(timestamps):
        if (i and pts <= timestamps[i - 1]) or abs(pts - origin - i / rate) > tick:
            raise ValueError('VFR, missing/duplicate frames, or a discontinuous video clock')
    return {'width': width, 'height': height, 'frames': len(frames), 'fps': str(rate),
        'time_base': str(tick), 'origin': str(origin), 'duration': str(len(frames) / rate),
        'timestamps': [str(p) for p in timestamps], 'pixel_format': stream.get('pix_fmt'),
        'timestamp_tolerance': 'one container tick, not one frame'}


def compare_video(source, result, width, height):
    if (result['width'], result['height']) != (width, height):
        raise ValueError('Enhanced output dimensions differ from the requested size')
    if source['frames'] != result['frames'] or Fraction(source['fps']) != Fraction(result['fps']):
        raise ValueError('Enhancement changed frame count or frame rate')
    shift = Fraction(result['origin']) - Fraction(source['origin'])
    tolerance = Fraction(source['time_base']) + Fraction(result['time_base'])
    for left, right in zip(source['timestamps'], result['timestamps']):
        if abs(Fraction(right) - Fraction(left) - shift) > tolerance:
            raise ValueError('Enhancement changed presentation timing')
    return {'status': 'video_timeline_preserved', 'common_shift': str(shift),
        'tolerance': str(tolerance), 'frames': source['frames'], 'fps': source['fps']}


def compare_audio_packets(source, result, common_shift):
    def audio_streams(probe):
        return [s for s in probe.get('streams', []) if s.get('codec_type') == 'audio']
    a, b = audio_streams(source), audio_streams(result)
    if len(a) != len(b):
        raise ValueError('Audio stream count changed')
    count = 0
    for left, right in zip(a, b):
        fields = ('codec_name', 'sample_rate', 'channels', 'channel_layout')
        if any(left.get(k) != right.get(k) for k in fields):
            raise ValueError('Audio codec/layout changed instead of stream copy')
        p = [x for x in source.get('packets', []) if x['stream_index'] == left['index']]
        q = [x for x in result.get('packets', []) if x['stream_index'] == right['index']]
        if len(p) != len(q) or not p:
            raise ValueError('Audio packets were lost or cannot be verified')
        lt, rt = Fraction(left['time_base']), Fraction(right['time_base'])
        for lp, rp in zip(p, q):
            if packet_priming(lp) != packet_priming(rp):
                raise ValueError('Audio priming/skip samples/discard padding changed')
            if not lp.get('data_hash', '').startswith('SHA256:') or lp['data_hash'] != rp.get('data_hash'):
                raise ValueError('Audio packet bytes changed')
            if type(lp.get('pts')) is not int or type(rp.get('pts')) is not int:
                raise ValueError('Audio packet lacks a presentation timestamp')
            if abs(rp['pts'] * rt - lp['pts'] * lt - Fraction(common_shift)) > lt + rt:
                raise ValueError('Audio/video relative timing changed')
            count += 1
    return {'status': 'packet_bytes_and_av_timing_preserved', 'streams': len(a), 'packets': count,
        'priming_and_padding_preserved': True}


def packet_priming(packet):
    values = [item for item in packet.get('side_data_list', [])
        if item.get('side_data_type') == 'Skip Samples']
    if len(values) > 1:
        raise ValueError('Ambiguous audio priming/skip metadata')
    if not values:
        return (0, 0, 0, 0)
    fields = ('skip_samples', 'discard_padding', 'skip_reason', 'discard_reason')
    result = tuple(values[0].get(name, 0) for name in fields)
    if any(type(value) is not int or not 0 <= value <= 2**31 - 1 for value in result):
        raise ValueError('Invalid audio priming/skip/padding value')
    return result


def lossless_master_suffix(audio_probe):
    """MOV preserves AAC edit-list priming; MKV is not a universal audio remux."""
    audio = [s for s in audio_probe.get('streams', []) if s.get('codec_type') == 'audio']
    for stream in audio:
        packets = [p for p in audio_probe.get('packets', []) if p['stream_index'] == stream['index']]
        if stream.get('codec_name') != 'aac' and any(any(packet_priming(p)) for p in packets):
            raise ValueError('Non-AAC priming/padding needs a separately qualified container route')
    return '.mov' if any(s.get('codec_name') == 'aac' for s in audio) else '.mkv'


def decoded_pcm_digests(runtime, path, probe, environment, stderr_path):
    """Stream decoded float32 PCM without retaining a full audio track in RAM.

    Called only inside the owned, deadline/resource-guarded worker. Any decoder
    child is in that worker's Windows Job. No resampling/channel remix is used.
    """
    records = []
    with Path(stderr_path).open('xb') as error_log:
        for stream in probe.get('streams', []):
            if stream.get('codec_type') != 'audio':
                continue
            index = stream.get('index')
            if type(index) is not int or index < 0:
                raise ValueError('Invalid PCM source stream index')
            command = [str(runtime.executable('ffmpeg.exe')), '-v', 'error', '-xerror',
                '-err_detect', 'explode', '-nostdin', '-protocol_whitelist', 'file,pipe',
                '-i', str(path), '-map', f'0:{index}', '-c:a', 'pcm_f32le',
                '-threads', '2', '-f', 'f32le', 'pipe:1']
            process = subprocess.Popen(command, cwd=runtime.install, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=error_log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False)
            digest, count = hashlib.sha256(), 0
            try:
                while block := process.stdout.read(1024**2):
                    count += len(block)
                    digest.update(block)
                if process.wait(timeout=30):
                    raise RuntimeError('Strict PCM decode failed; output is not published')
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
            if not count:
                raise ValueError('Audio stream produced empty PCM')
            records.append({'stream': index, 'bytes': count, 'sha256': digest.hexdigest()})
    return records


def compare_pcm_digests(source, output):
    if len(source) != len(output) or any(
            a.get('bytes') != b.get('bytes') or a.get('sha256') != b.get('sha256')
            or type(a.get('bytes')) is not int or a['bytes'] <= 0
            or not isinstance(a.get('sha256'), str) or len(a['sha256']) != 64
            for a, b in zip(source, output)):
        raise ValueError('Decoded PCM differs; identical compressed packets alone are insufficient')
    return {'status': 'decoded_pcm_bitexact', 'streams': len(source), 'source': source, 'output': output}
