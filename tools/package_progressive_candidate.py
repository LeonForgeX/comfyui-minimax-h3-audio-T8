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
SELF_LIFT_DIR = 'examples/workflows/33-selflift-taomate/'
SELF_LIFT_WORKFLOWS = {
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_Core_Sage_4plus4_EXP.json': '000d37fc4b9a1c509ccf16397e6e2d1324f6771a0d76c9de2c8242d072c3132d',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_EAV_4plus4_EXP.json': 'e479fd6771335a423898ee58af345bc62afe9068a41b1b0f928bd87c72c51936',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_Guide_Mean_4plus4_EXP.json': 'af2ac3ea9dd68bc41ce2d1246cc38d6fb5eb7797294b6f9177beeab330bdf763',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_KJ_FFN_TST_EAV_Relay_4plus4_EXP.json': '1e87f44817477d1f55cff4b0311b113fcf3712d1b92958028136ceb7dbc004e9',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_KJ_Relay_Two_Segment_8s_EXP.json': '120503ff288ab656e877cc6502cc4673a8e8e77d0c1aa1375ce226f51c07396f',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_Sol_4plus4_EXP.json': '8d5f77d59ae29d8c2917edffa5153d3f7073ba09b76435c3c537f761e287cdf1',
    SELF_LIFT_DIR+'2026-09-14_H3_SelfLift_I2VA_TST_4plus4_EXP.json': '8f329f155900303df599ae70cdcc722e57c02ac0173d07def27ac56d1afa971a',
    SELF_LIFT_DIR+'2026-09-14_H3_TaoMate_T2VA_3step_EXP.json': 'ca19c01603598698483e84a5b58e6959f40f2c4726609421fe66014f2e334570',
    SELF_LIFT_DIR+'2026-09-14_H3_TaoMate_T2VA_4step_EXP.json': '3cd92726a3312e51c67114c8d9f5e15f520f024ef5798105618c2d348549a565',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_human_promotion(files):
    expected = SELF_LIFT_WORKFLOWS
    if any(files.get(name) != digest for name, digest in expected.items()):
        raise ValueError('Packaged SelfLift workflow identity differs from the released set')
    pending = ('UNREVIEWED', '未人审', '尚待人审', '仍需CPU/UI复核', '未通过不晋级')
    for name in expected:
        text = (PROJECT/name).read_text(encoding='utf8')
        if any(token in text for token in pending):
            raise ValueError('Pending-review marker in promoted workflow: '+name)
        review = json.loads(text).get('extra', {}).get('t8_bound_review', {})
        if (review.get('status') != 'accepted_in_this_review_scope'
                or review.get('not_universal_quality_claim') is not True):
            raise ValueError('Scoped review metadata missing from promoted workflow: '+name)
    for name in expected:
        if '_EAV_' not in name:
            continue
        workflow = json.loads((PROJECT/name).read_text(encoding='utf8'))
        long_node = next(node for node in workflow['nodes'] if node['type'] == 'MiniMaxH3ProgressiveLongVideoEXPT8')
        values = long_node['widgets_values_named']
        if (values['eav_tau'], values['eav_start_video_progress'], values['eav_end_video_progress'], values['eav_g_hard_limit']) != (8.0, .15, .90, 1.5):
            raise ValueError('Released EAV recipe differs: '+name)
    return expected


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
    extras += [p.relative_to(PROJECT).as_posix() for p in (PROJECT/'h3_t8').rglob('*') if p.is_file()]
    # The shared index predates four already-delivered VDN workflows as well as
    # the new progressive pair. Snapshot the complete project-owned examples,
    # not the live user-workflow directory and not only newly named files.
    extras += [p.relative_to(PROJECT).as_posix() for p in (PROJECT/'examples').rglob('*') if p.is_file()]
    selected = sorted({n for n in [*tracked, *extras, *includes] if n and
        n not in ('SKILL.md', 'roadmap.md', 'ROADMAP.md') and
        (n in includes or not ignore.match_file(n))})
    index_path = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-path', 'index'], cwd=PROJECT).decode().strip())
    index_sha = sha(index_path)
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
    required = {'h3_t8/nodes_progressive_sampling.py', 'h3_t8/progressive_sampling_runtime.py', 'h3_t8/progressive_sampling_contract.py',
                'h3_t8/acceleration_measurement.py', 'docs/PROGRESSIVE_SAMPLING_EXP.md',
                'h3_t8/nodes_dlss_fi.py', 'h3_t8/dlss_fi_backend/entry.py', 'h3_t8/dlss_fi_backend/file_task.py',
                'h3_t8/nodes_trt_vae.py', 'h3_t8/trt_vae_compile_worker.py', 'docs/TRT_VAE_EXP.md',
                'h3_t8/nodes_long_video_dual_model.py', 'h3_t8/long_video_dual_model_runner.py',
                'h3_t8/nodes_topaz.py', 'h3_t8/topaz_worker.py', 'h3_t8/topaz_media.py',
                'docs/DUAL_MODEL_LONG_VIDEO_EXP.md', 'docs/TOPAZ_EXP.md', 'docs/R1_RELIABILITY_20260911.md',
                'docs/RELEASE_1.80.0.md', 'h3_t8/nodes_progressive_long_video.py', 'h3_t8/nodes_tst.py',
                'examples/workflows/29-dlss-fi/README.md', 'examples/workflows/33-selflift-taomate/README.md'}
    if not required <= set(files):
        raise ValueError('Progressive package missing required files')
    promoted_workflows = validate_human_promotion(files)
    workflow_count = sum(name.startswith('examples/workflows/') and name.endswith('.json') for name in files)
    if workflow_count != 254:
        raise ValueError(f'Expected 254 packaged workflows, found {workflow_count}')
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
    if sha(index_path) != index_sha or any(sha(PROJECT/n) != digest for n, digest in files.items()):
        raise ValueError('Live source/index changed during candidate build')
    receipt = {'status': 'official_local_human_reviewed_candidate_archive_verified_import_pending', 'version': config['project']['version'],
        'archive': str(archive), 'archive_sha256': sha(archive), 'extracted': str(extracted), 'files': files,
        'main_index_sha256': index_sha, 'index_path': str(index_path), 'main_index_unchanged': True, 'public_FI_node_included': True,
        'workflow_json_count': workflow_count, 'selflift_workflows': promoted_workflows,
        'human_review_scope': 'Only the exact bound short samples and the documented composition of their accepted controls.',
        'universal_quality_claim': False, 'published': False, 'human_qualified': True,
        'source_scope': 'current_local_v1.80.0_candidate_not_published'}
    with (root/'receipt.json').open('x', encoding='utf8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
    return {k: v for k, v in receipt.items() if k != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False))
