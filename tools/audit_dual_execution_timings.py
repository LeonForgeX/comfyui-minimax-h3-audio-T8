"""Read-back of completed multi-segment timing evidence; never executes inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError('Evidence path leaves its job')
    return path


def signed_sidecar(path):
    payload = read(path)
    claimed = payload.pop('audit_sha256', None)
    actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    if actual != claimed:
        raise ValueError('Effects sidecar checksum mismatch')
    return payload, claimed


def timing_rows(ledger, samples, scope, segment):
    if (ledger.get('clock') != 'perf_counter' or ledger.get('gpu_synchronization_added') is not False
            or ledger.get('historical_cache_time_counted_as_current') is not False):
        raise ValueError('Unexpected timing clock/synchronization/cache accounting')
    rows = []
    for event in ledger['events']:
        start, end, seconds = (event[key] for key in ('start', 'end', 'seconds'))
        if (event['status'] != 'completed'
                or not all(type(value) in (int, float) and math.isfinite(value) for value in (start, end, seconds))
                or not samples[0]['monotonic'] <= start <= end <= samples[-1]['monotonic']
                or abs(seconds - (end - start)) > 1e-6):
            raise ValueError('Invalid/out-of-run timing interval')
        observed = [sample for sample in samples if start <= sample['monotonic'] <= end]
        rows.append({'segment': segment, 'scope': scope, **event,
            'resource_sample_count': len(observed),
            'maximum_observed_device_used_bytes': max((s['gpu_used_bytes'] for s in observed), default=None),
            'minimum_observed_system_ram_available_bytes': min((s['ram_available_bytes'] for s in observed), default=None)})
    if not rows:
        raise ValueError('Missing timing events')
    return rows


def audit(root):
    root = Path(root).resolve(strict=True)
    terminal = read(root / 'terminal.json')
    if (not terminal['status'].startswith('generation_completed')
            or terminal['server_stop']['owned_children_remaining']):
        raise ValueError('Completed generation and owned cleanup required')
    independent = root / 'independent-audit-v1.json'
    proof = read(independent)
    if proof['status'] != 'multi_segment_mechanical_pass_human_pending':
        raise ValueError('Independent multi-segment media/stage audit required')
    manifests = list((root / 'output').rglob('manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Expected one completed chain')
    chain = manifests[0].parent
    segments = read(manifests[0])['segments']
    state = read(chain / 'in_node_loop_effects_state.json')
    persisted = read(chain / 'last_execution_report.json')
    if (len(segments) < 2 or state['status'] != 'complete' or state['accepted_count'] != len(segments)
            or persisted['status'] != 'complete' or persisted['contract_sha256'] != state['contract_sha256']
            or len(proof['segments']) != len(segments)):
        raise ValueError('Completed chain, timing invocation and independent audit disagree')
    media = bound(chain, state['final_video_path'])
    if (Path(proof['media']['path']).resolve(strict=True) != media
            or sha(media) != proof['media']['sha256'] or proof['media']['sha256'] != state['final_video_sha256']):
        raise ValueError('Media identity no longer binds the independent audit')
    samples = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    stamps = [s['monotonic'] for s in samples]
    if (not stamps or any(type(s) not in (int, float) or not math.isfinite(s) for s in stamps)
            or any(a >= b for a, b in zip(stamps, stamps[1:]))
            or any(s['clock'] != 'perf_counter' for s in samples)):
        raise ValueError('Ordered same-clock resource samples required')
    rows, identities = [], []
    for index, segment in enumerate(segments):
        if segment['index'] != index or proof['segments'][index]['index'] != index:
            raise ValueError('Segment ordering differs')
        candidate = bound(chain, f"candidates/segment_{index:05d}/" + segment['candidate_id'])
        payload, identity = signed_sidecar(candidate / 'effects_audit.json')
        if (payload['segment_index'] != index or payload['candidate_id'] != candidate.name
                or payload['contract_sha256'] != state['contract_sha256']):
            raise ValueError('Timing sidecar belongs to another job/segment')
        dual = payload['sampling_plan']['dual_model']
        if dual['low_reused'] or dual['high_reused']:
            raise ValueError('Fresh uninterrupted timing does not include cached stage time')
        ledgers = {'segment': dual['execution_timings'], 'delivery': payload['delivery_execution_timings']}
        for scope, expected in (
            ('segment', ['first_conditioning', 'first_sampling', 'second_conditioning',
                         'learned_upscale', 'reconcile', 'second_sampling']),
            ('delivery', ['av_decode', 'trim', 'candidate_encode_and_save'])):
            if [event['phase'] for event in ledgers[scope]['events']] != expected:
                raise ValueError('Required measured pipeline phases are missing or reordered')
        for stage in ('first_pass', 'second_pass'):
            report = dual[stage]
            ledger = report['execution_timings']
            phases = [e['phase'] for e in ledger['events']]
            if (report['completed_network_forwards'] != 4
                    or phases.count('prepare_sampling_including_model_load') != 1
                    or phases.count('forward_including_dynamic_transfers') != 4):
                raise ValueError('Missing actual prepare/forward observations')
            ledgers[stage] = ledger
        for scope, ledger in ledgers.items():
            rows.extend(timing_rows(ledger, samples, scope, index))
        identities.append(identity)
    composition = persisted['composition_execution_timings']
    if [event['phase'] for event in composition['events']] != ['final_composition']:
        raise ValueError('Final composition timing missing')
    rows.extend(timing_rows(composition, samples, 'composition', None))
    return {'status': 'multi_segment_execution_timings_verified', 'segments': len(segments),
        'rows': rows, 'effects_sha256s': identities, 'independent_audit_sha256': sha(independent),
        'media_sha256': proof['media']['sha256'], 'human_review': 'pending',
        'scope': 'Nested host wall intervals, including dynamic transfers, are not additive pure GPU time. Periodic whole-device/system observations are not exact/per-process peaks. No samples means unobserved, not zero.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new evidence output')
    result = audit(args.root)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'status': result['status'], 'segments': result['segments'], 'intervals': len(result['rows'])}))


if __name__ == '__main__':
    main()
