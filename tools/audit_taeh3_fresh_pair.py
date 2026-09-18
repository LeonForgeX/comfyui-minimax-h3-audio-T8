"""Compare independently initialized real GPU traces without altering receipts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import torch
    root = args.root.resolve(strict=True)
    project = Path(__file__).resolve().parents[1]
    if not root.is_relative_to(project / 'artifacts'):
        raise ValueError('Owned artifacts required')
    dirs = [root / ('taeh3-fresh-' + route + '-gpu-v1') for route in ('off', 'on')]
    receipts = [json.loads((path / 'terminal.json').read_text(encoding='utf8')) for path in dirs]
    assert all(row.get('sources_unchanged') and len(row['cases']) == 1 for row in receipts)
    assert receipts[0]['source_hashes'] == receipts[1]['source_hashes']
    a, b = [torch.load(path / (route + '.pt'), map_location='cpu', weights_only=True)
        for path, route in zip(dirs, ('off1', 'on'))]
    assert len(a['trace']) == len(b['trace']) == 8
    def delta(first, second):
        assert all(torch.isfinite(item).all() for item in first + second)
        return [float((x-y).abs().max()) for x, y in zip(first, second)]
    steps = [dict(phase=x['phase'], input_max_abs_difference=delta(x['input'], y['input']),
        output_max_abs_difference=delta(x['output'], y['output']),
        context_same=x['context_sha256'] == y['context_sha256'],
        refs_same=x['reference_video_sha256'] == y['reference_video_sha256'] and x['reference_audio_sha256'] == y['reference_audio_sha256'])
        for x, y in zip(a['trace'], b['trace'])]
    final = delta(a['final'], b['final'])
    events = [json.loads(line)['data'] for line in (dirs[1] / 'events.jsonl').read_text(encoding='utf8').splitlines()
        if json.loads(line)['event'] == 't8-taeh3-sampling-preview']
    frames = [event for event in events if event['kind'] == 'frames']
    # Observer metadata reports the source latent grid, not pixel dimensions.
    assert frames and all(event['display_aspect_restored'] and
        event['source_aspect'][0] * 3 == event['source_aspect'][1] * 2 for event in frames)
    assert all(row['unchanged'] and row.get('input_unchanged', True) for row in receipts[1]['decoder_rng'])
    result = dict(status='independent_fresh_pretrained_GPU_pair_bit_identity_pass' if final == [0., 0.]
        and all(step['input_max_abs_difference'] == step['output_max_abs_difference'] == [0., 0.] and step['context_same'] and step['refs_same'] for step in steps)
        else 'independent_fresh_pretrained_GPU_pair_not_bit_identical',
        final_max_abs_difference=final, steps=steps, actual_native_forwards=[8, 8],
        actual_preview_updates=len(frames), real_decoder_rng_and_input_unchanged=True,
        display_aspect_restored=True, cuda_initialized=torch.cuda.is_initialized(),
        human_qualified=False, final_media_decoded=False, historical_v6_failure_unchanged=True,
        qualification='Real pretrained8+8 GPU sampling and real decoder; tensor comparison only, no media/human claim')
    path = root / 'taeh3-fresh-pair-audit-v1.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
