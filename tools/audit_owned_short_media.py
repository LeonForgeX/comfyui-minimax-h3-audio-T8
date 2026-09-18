"""Full AV and timestamp audit with real posters; no GPU or source overwrite."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=73)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--audio-policy', choices=['required', 'absent'], default='required')
    parser.add_argument('--clip', type=Path, action='append', default=[])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(project / 'artifacts'):
        raise ValueError('Fresh owned output required')
    args.output.mkdir(parents=True)
    import av
    import numpy as np
    rows = []
    paths = args.clip or sorted(args.root.rglob('*.mp4'))
    if any(not path.resolve(strict=True).is_relative_to(args.root.resolve(strict=True)) for path in paths):
        raise ValueError('Media must remain inside the explicitly scoped root')
    for path in paths:
        identity = digest(path)
        pts, means, deviations, last, rgb_hash = [], [], [], None, hashlib.sha256()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert (stream.width, stream.height) == (args.width, args.height)
            assert stream.codec_context.name == 'h264'
            for frame in container.decode(video=0):
                assert frame.pts is not None
                pts.append(str(frame.pts * frame.time_base))
                pixels = frame.to_ndarray(format='rgb24')
                rgb_hash.update(pixels.tobytes())
                means.append(float(pixels.mean()))
                deviations.append(float(pixels.std()))
                if len(pts) == 1:
                    frame.to_image().save(args.output / (path.stem + '_poster.png'))
                last = frame
            assert len(pts) == args.frames
            from fractions import Fraction
            assert all(Fraction(a) < Fraction(b) for a, b in zip(pts, pts[1:]))
            last.to_image().save(args.output / (path.stem + '_last.png'))
        samples, blocks, square, peak, pcm_hash = 0, 0, 0., 0., hashlib.sha256()
        with av.open(str(path)) as container:
            if args.audio_policy == 'absent':
                assert not container.streams.audio
                sample_rate = None
            else:
                stream = container.streams.audio[0]
                sample_rate = stream.codec_context.sample_rate
        if args.audio_policy == 'required':
            with av.open(str(path)) as container:
                for frame in container.decode(audio=0):
                    wave = frame.to_ndarray()
                    assert np.isfinite(wave).all()
                    pcm_hash.update(wave.tobytes())
                    square += float(np.square(wave.astype(np.float64)).sum())
                    peak = max(peak, float(np.abs(wave).max()))
                    samples += wave.size
                    blocks += 1
                assert blocks and samples and peak
        assert digest(path) == identity
        rows.append(dict(path=str(path), sha256=identity, frames=len(pts), dimensions=[args.width, args.height],
            all_video_decoded=True, all_audio_decoded=args.audio_policy == 'required',
            audio_absent=args.audio_policy == 'absent', pts=pts, rgb_sha256=rgb_hash.hexdigest(),
            frame_means=means, frame_stds=deviations, audio_blocks=blocks, audio_values=samples,
            sample_rate=sample_rate, rms=(square/samples)**.5 if samples else None, peak=peak,
            audio_policy=args.audio_policy, decoded_audio_sha256=pcm_hash.hexdigest(),
            poster=str(args.output / (path.stem + '_poster.png')), last=str(args.output / (path.stem + '_last.png'))))
    assert rows
    result = dict(status='complete_media_integrity_not_human_quality', media=rows, original_files_unchanged=True,
        human_qualified=False, GPU_used=False, published=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(dict(status=result['status'], clips=len(rows), all_frames=args.frames, published=False)))


if __name__ == '__main__':
    main()
