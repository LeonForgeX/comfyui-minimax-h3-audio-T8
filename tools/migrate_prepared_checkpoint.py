"""Explicitly adopt evidenced original pilots; no model execution or publication."""
import argparse
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--generation-directory', type=Path, required=True)
    parser.add_argument('--decode-directory', type=Path, required=True)
    parser.add_argument('--media-directory', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new local migration bundle; never overwrite')
    package = types.ModuleType('_prepared_migration')
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'h3_t8')]
    sys.modules[package.__name__] = package
    from _prepared_migration.prepared_generation_contract import read_bundle
    from _prepared_migration.prepared_checkpoint import migration
    bundle, report = migration(read_bundle(args.bundle), args.generation_directory, args.decode_directory,
        seed=args.seed, media_directory=args.media_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'kind': report['kind'], 'GPU_rerun': False,
        'helpers': len(report['math_helpers_identical']), 'evidence_files': len(report['evidence']),
        'path': str(args.output.resolve()), 'human_review': 'pending'}))


if __name__ == '__main__':
    main()
