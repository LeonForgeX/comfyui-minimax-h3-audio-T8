"""Fail closed before publishing any commit history containing local roadmaps."""
import argparse
from pathlib import PurePosixPath
import subprocess
import sys


def forbidden(paths):
    return sorted({p.strip() for p in paths if p.strip()
                   and PurePosixPath(p.strip()).name.casefold() == 'roadmap.md'})


def git(*args):
    return subprocess.check_output(['git', *args]).decode('utf-8', errors='surrogateescape')


def check_index():
    return forbidden(git('ls-files', '-z').split('\0'))


def check_history(ref):
    # Include every ancestor, not only the current tree. This prevents an old
    # pre-cleanup branch from reintroducing the file via a merge or force push.
    sha = git('rev-parse', '--verify', ref + '^{commit}').strip()
    return forbidden(git('log', '--format=', '--name-only', '--no-renames', '-z', sha).split('\0'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-push', action='store_true')
    parser.add_argument('--history')
    args = parser.parse_args()
    bad = []
    if args.pre_push:
        for line in sys.stdin:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError('Malformed pre-push input; refusing push')
            sha = fields[1]
            if set(sha) == {'0'}:  # ref deletion has no content to publish
                continue
            bad.extend(check_history(sha))
    else:
        bad.extend(check_index())
        if args.history:
            bad.extend(check_history(args.history))
    if bad:
        print('BLOCKED: local ROADMAP files must never be published (including history).', file=sys.stderr)
        print('\n'.join(sorted(set(bad))), file=sys.stderr)
        return 1
    print('No ROADMAP files in the checked publication scope.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
