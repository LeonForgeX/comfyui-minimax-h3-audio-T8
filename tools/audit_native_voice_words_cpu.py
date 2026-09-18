"""Offline CPU ASR observations only; no waveform changes, hints or quality acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observe(model, path, language):
    segments, info = model.transcribe(
        str(path), language=language, beam_size=5, temperature=0,
        initial_prompt=None, prefix=None, hotwords=None,
        condition_on_previous_text=False, vad_filter=False,
    )
    segments = list(segments)  # Consume the real lazy inference generator.
    return dict(requested_language=language, detected_language=info.language,
                language_probability=info.language_probability, duration=info.duration,
                text=''.join(segment.text for segment in segments).strip(),
                segments=[dict(start=s.start, end=s.end, text=s.text,
                               avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob)
                          for s in segments])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip', type=Path, action='append', required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    evidence = project / 'artifacts/five-track-development-20260918'
    if args.output.exists() or not args.output.resolve().is_relative_to(evidence):
        raise ValueError('Use a fresh owned evidence output')
    if any(not path.resolve(strict=True).is_relative_to(evidence) for path in args.clip):
        raise ValueError('Generated clips must come from this task evidence')
    if not args.model.is_dir():
        raise ValueError('An existing local model directory is required; no downloading')
    model_files = [args.model / name for name in ('model.bin', 'config.json', 'tokenizer.json')]
    if any(not path.is_file() for path in model_files):
        raise ValueError('Incomplete local ASR model')
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2')
    args.output.mkdir(parents=True)
    report = dict(status='incomplete', human_qualified=False, GPU_used=False,
                  published=False, source_waveforms_modified=False,
                  interpretation='ASR can misrecognize or hallucinate; it does not qualify identity, emotion, noise, lip sync or seams',
                  model_directory=str(args.model), model_SHA256={p.name: digest(p) for p in model_files},
                  configuration=dict(device='cpu', compute_type='int8', cpu_threads=2,
                                     local_files_only=True, initial_prompt=None, prefix=None,
                                     hotwords=None, condition_on_previous_text=False, vad_filter=False),
                  upstream='https://github.com/SYSTRAN/faster-whisper',
                  versions={name: importlib.metadata.version(name) for name in ('faster-whisper', 'ctranslate2')})
    started = time.monotonic()
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(str(args.model), device='cpu', compute_type='int8',
                             cpu_threads=2, num_workers=1, local_files_only=True)
        rows = []
        inputs = [(p, False) for p in args.clip]
        if args.reference:
            inputs.append((args.reference.resolve(strict=True), True))
        for path, is_reference in inputs:
            sha = digest(path)
            row = dict(path=str(path), sha256=sha, reference_recording=is_reference,
                       observations=[observe(model, path, None),
                                     observe(model, path, 'en' if is_reference else 'zh')])
            assert digest(path) == sha, 'Input changed during ASR'
            rows.append(row)
            (args.output / 'observations.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf8')
        assert all(digest(p) == report['model_SHA256'][p.name] for p in model_files)
        report.update(status='complete_offline_CPU_ASR_observations_not_human_acceptance',
                      media=rows, input_and_model_bytes_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        print(json.dumps(dict(status=report['status'], observations=len(report.get('media', [])),
                             GPU_used=False, human_qualified=False)), flush=True)


if __name__ == '__main__':
    main()
