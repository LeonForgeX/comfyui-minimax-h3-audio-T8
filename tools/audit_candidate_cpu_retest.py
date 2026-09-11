"""Reconcile per-ID full CPU results and a bounded failure retest, not one full rerun."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--retest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit output must be new')
    evidence = {}
    results = []
    collections = []
    for root in (args.baseline, args.retest):
        blobs = {name: (root / name).read_bytes() for name in ('collection.json', 'phases.jsonl', 'terminal.json')}
        evidence[str(root)] = {key: hashlib.sha256(value).hexdigest() for key, value in blobs.items()}
        terminal = json.loads(blobs['terminal.json'])
        if terminal['cuda_initialized']:
            raise ValueError('Unexpected CUDA initialization')
        nodes = json.loads(blobs['collection.json'])
        if len(nodes) != len(set(nodes)):
            raise ValueError('Duplicate collected IDs')
        collections.append(set(nodes))
        outcomes = {}
        teardown = set()
        for line in blobs['phases.jsonl'].splitlines():
            row = json.loads(line)
            node = row['nodeid']
            if row['when'] == 'teardown':
                teardown.add(node)
            if row['outcome'] == 'failed':
                outcomes[node] = 'failed'
            elif outcomes.get(node) != 'failed' and (row['when'] == 'call' or row['outcome'] == 'skipped'):
                outcomes[node] = row['outcome']
        if set(outcomes) != set(nodes) or teardown != set(nodes):
            raise ValueError('Incomplete per-ID execution or teardown')
        results.append(outcomes)
    if not collections[1] <= collections[0] or set(results[1].values()) != {'passed'}:
        raise ValueError('Retest must be a passing subset of the full collection')
    merged = {**results[0], **results[1]}
    if 'failed' in merged.values():
        raise ValueError('Unresolved failures remain')
    report = {'status': 'per_id_reconciled', 'evidence_sha256': evidence,
              'collected': len(merged), 'passed': list(merged.values()).count('passed'),
              'skipped': list(merged.values()).count('skipped'),
              'failed_ids_retested': sorted(key for key, value in results[0].items() if value == 'failed'),
              'scope': 'Full baseline plus bounded retest; not a single uninterrupted all-pass run or future-source certification'}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
