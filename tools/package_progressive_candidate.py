"""Build a local candidate with the installed official Comfy zip implementation.

Uses a new isolated repository, never the user's main index; no commit, remote,
network, publication, runtime installation, weights or DLSS binaries. Tools-only
FI prototypes are deliberately excluded and are not delivered as a public node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'artifacts/acceleration-research-20260909'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    from pathspec import PathSpec
    from comfy_cli.file_utils import zip_files
    root = Path(root).resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New dedicated research output required')
    config = tomllib.loads((PROJECT/'pyproject.toml').read_text(encoding='utf8'))
    includes = config['tool']['comfy']['includes']
    ignore = PathSpec.from_lines('gitwildmatch', (PROJECT/'.comfyignore').read_text().splitlines())
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=PROJECT).decode('utf8').split('\0')
    extras = [p.relative_to(PROJECT).as_posix() for p in PROJECT.glob('*.py')]
    extras += [p.relative_to(PROJECT).as_posix() for p in (PROJECT/'dlss_fi_backend').glob('*.py')]
    # The shared index predates four already-delivered VDN workflows as well as
    # the new progressive pair. Snapshot the complete project-owned examples,
    # not the live user-workflow directory and not only newly named files.
    extras += [p.relative_to(PROJECT).as_posix() for p in (PROJECT/'examples').rglob('*') if p.is_file()]
    selected = sorted({n for n in [*tracked, *extras, *includes] if n and
        n not in ('SKILL.md', 'roadmap.md', 'ROADMAP.md') and
        (n in includes or not ignore.match_file(n))})
    index_sha = sha(PROJECT/'.git/index')
    secrets = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)')
    files = {}
    for name in selected:
        path = (PROJECT/name).resolve(strict=True)
        if not path.is_relative_to(PROJECT) or not path.is_file() or name.startswith(('.git/', 'artifacts/', 'tools/', 'tests/')):
            raise ValueError('Unsafe package member: '+name)
        if path.suffix.lower() in ('.safetensors', '.onnx', '.ckpt', '.pt', '.dll', '.exe', '.mp4', '.wav', '.pyc'):
            raise ValueError('External runtime/model/media not included: '+name)
        data = path.read_bytes()
        if len(data) > 40*1024**2 or secrets.search(data):
            raise ValueError('Oversized file or possible secret in '+name)
        files[name] = hashlib.sha256(data).hexdigest()
    required = {'nodes_progressive_sampling.py', 'progressive_sampling_runtime.py', 'progressive_sampling_contract.py',
                'acceleration_measurement.py', 'docs/PROGRESSIVE_SAMPLING_EXP.md',
                'nodes_dlss_fi.py', 'dlss_fi_backend/entry.py', 'dlss_fi_backend/file_task.py',
                'examples/workflows/29-dlss-fi/README.md'}
    if not required <= set(files):
        raise ValueError('Progressive package missing required files')
    root.mkdir(parents=True)
    repo = root/'snapshot'
    repo.mkdir()
    for name in files:
        destination = repo/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/name, destination)
        if sha(destination) != files[name]:
            raise ValueError('Source changed while copying '+name)
    subprocess.run(['git', 'init', '-q', '-b', 'codex/package-preview'], cwd=repo, check=True)
    for start in range(0, len(selected), 40):
        subprocess.run(['git', '-c', 'core.longpaths=true', '-c', 'core.autocrlf=false', '-c', 'core.safecrlf=false',
                        'add', '--', *selected[start:start+40]], cwd=repo, check=True, capture_output=True)
    archive = root/'minimax-h3-audio-T8-local-candidate.zip'
    previous = Path.cwd()
    try:
        os.chdir(repo)
        zip_files(str(archive), includes=includes)
    finally:
        os.chdir(previous)
    extracted = root/'unpacked/minimax-h3-audio-T8'
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None or len(package.namelist()) != len(set(package.namelist())) or set(package.namelist()) != set(files):
            raise ValueError('Official archive membership differs')
        for name, digest in files.items():
            if hashlib.sha256(package.read(name)).hexdigest() != digest or not (extracted/name).resolve().is_relative_to(extracted.resolve()):
                raise ValueError('Archive identity/target differs')
        package.extractall(extracted)
    if sha(PROJECT/'.git/index') != index_sha or any(sha(PROJECT/n) != digest for n, digest in files.items()):
        raise ValueError('Live source/index changed during candidate build')
    receipt = {'status': 'official_local_candidate_archive_verified_import_pending', 'version': config['project']['version'],
        'archive': str(archive), 'archive_sha256': sha(archive), 'extracted': str(extracted), 'files': files,
        'main_index_sha256': index_sha, 'main_index_unchanged': True, 'public_FI_node_included': True,
        'published': False, 'human_qualified': False, 'source_scope': 'current_local_candidate_not_a_release'}
    with (root/'receipt.json').open('x', encoding='utf8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
    return {k: v for k, v in receipt.items() if k != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False))
