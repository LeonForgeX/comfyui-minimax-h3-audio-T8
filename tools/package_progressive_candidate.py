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
EXPECTED_RELEASE_VERSION = '1.82.0'
EXPECTED_WORKFLOW_COUNT = 255
T8_MEMORY_WORKFLOW = (
    'examples/workflows/04-long-video/'
    '2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json'
)
T8_MEMORY_REVIEW_BINDING = {
    'review_id': '280131c7a8eb41801b92c81b3c49bde5465b111b069ec63f43918e478b757100',
    'media_sha256': '0cf5413eb9715d3a9b730b8025a5f71d2f1ed8622475095927389e0895e69d11',
    'audit_sha256': '58ba1960595ff35b431d7d276f7017992af0566224634bb9d9f1b209dc46eb1e',
}
T8_MEMORY_EXECUTION_SHA256 = '4013714011c504d9df8af656af76ea43d8eb89d61400e5ae830b3238a8dd1c73'
PENDING_REVIEW_TOKENS = (
    'UNREVIEWED', '未人审', '尚待人审', '仍需CPU/UI复核', '未通过不晋级',
    'human review required', 'human review pending',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe_identity(value):
    # JSON/native serializers may emit 8 and 8.0 for the same numeric widget.
    # Booleans remain distinct; no float rounding or string coercion is allowed.
    if isinstance(value, dict):
        return {key: recipe_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recipe_identity(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def validate_human_promotion(files):
    expected = SELF_LIFT_WORKFLOWS
    if any(files.get(name) != digest for name, digest in expected.items()):
        raise ValueError('Packaged SelfLift workflow identity differs from the released set')
    for name in expected:
        text = (PROJECT/name).read_text(encoding='utf8')
        if any(token.lower() in text.lower() for token in PENDING_REVIEW_TOKENS):
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


def validate_t8_memory_workflow(workflow):
    extra = workflow.get('extra', {})
    review = extra.get('t8_bound_review', {})
    if (review.get('status') != 'accepted_in_this_review_scope'
            or review.get('not_universal_quality_claim') is not True
            or any(review.get(key) != value for key, value in T8_MEMORY_REVIEW_BINDING.items())):
        raise ValueError('T8 memory workflow lacks the exact accepted review binding')
    paths = extra.get('t8_memory_paths', {})
    expected_path = ['extra_lora', 'low_vram_attention_h4', 'chunk_ffn_c2']
    if paths.get('pass1') != expected_path or paths.get('pass2') != expected_path:
        raise ValueError('T8 memory workflow no longer binds both h4+c2 MODEL paths')
    low_vram = [node for node in workflow.get('nodes', [])
                if node.get('type') == 'MiniMaxH3LowVRAMAttentionT8Advanced']
    chunk_ffn = [node for node in workflow.get('nodes', [])
                 if node.get('type') == 'MiniMaxH3ChunkFeedForwardT8Advanced']
    long_video = [node for node in workflow.get('nodes', [])
                  if node.get('type') == 'MiniMaxH3DualModelLongVideoEXPT8']
    if (len(low_vram) != 2 or any(node.get('widgets_values') != [4] for node in low_vram)
            or len(chunk_ffn) != 2
            or any(node.get('widgets_values') != [2, 4096] for node in chunk_ffn)
            or len(long_video) != 1):
        raise ValueError('T8 memory workflow node recipe differs from reviewed h4+c2 dual path')
    values = long_video[0].get('widgets_values', [])
    if (len(values) < 52 or values[3:5] != [4, 4]
            or values[11] != 'h3_t8_h4c2_portrait_motion_color_8s_20260916'
            or values[12:17] != [8, 512, 768, 124, 22]
            or values[19] != 'apply_exp'
            or values[49] != 'high_native_mask_ramp_exp'
            or values[50] != 'accepted_picture_low_context_v1'
            or values[51] != 'bounded_motion_color_exp'):
        raise ValueError('T8 memory workflow long-video controls differ from reviewed candidate')
    seam = extra.get('seam_context', {})
    if seam != {
            'low_context_source': 'accepted_picture_low_context_v1',
            'high_video_context_mode': 'high_native_mask_ramp_exp',
            'high_release_ramp': [0.25, 0.5, 0.75],
            'audio_unchanged': True,
            'color_match_mode': 'bounded_motion_color_exp'}:
        raise ValueError('T8 memory workflow seam context differs from reviewed candidate')
    execution_nodes = [
        {'id': node['id'], 'type': node['type'], 'mode': node.get('mode', 0),
         'widgets_values': node.get('widgets_values', []),
         'input_links': [[pin['name'], pin['link']] for pin in node.get('inputs', [])
                         if pin.get('link') is not None]}
        for node in workflow['nodes'] if node['type'] != 'MarkdownNote'
    ]
    identity = hashlib.sha256(json.dumps(
        recipe_identity({'nodes': execution_nodes, 'links': workflow['links']}), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if identity != T8_MEMORY_EXECUTION_SHA256:
        raise ValueError('T8 memory workflow execution differs from the exact accepted control composition')
    return review


def validate_t8_memory_promotion(files, project=PROJECT):
    path = Path(project) / T8_MEMORY_WORKFLOW
    if files.get(T8_MEMORY_WORKFLOW) != sha(path):
        raise ValueError('Packaged T8 memory workflow identity differs from source')
    text = path.read_text(encoding='utf8')
    if any(token.lower() in text.lower() for token in PENDING_REVIEW_TOKENS):
        raise ValueError('T8 memory workflow still carries a pending human-review marker')
    review = validate_t8_memory_workflow(json.loads(text))
    return {'path': T8_MEMORY_WORKFLOW, 'review': review}


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
                'docs/RELEASE_1.80.0.md', 'docs/RELEASE_1.82.0.md', 'docs/H3_MEMORY_NODES_EXP.md',
                'h3_t8/nodes_progressive_long_video.py', 'h3_t8/nodes_tst.py', 'h3_t8/h3_memory_advanced.py',
                'h3_t8/long_video_motion_color.py', 'docs/MOTION_COLOR_EXP.md',
                T8_MEMORY_WORKFLOW,
                'examples/workflows/29-dlss-fi/README.md', 'examples/workflows/33-selflift-taomate/README.md'}
    if not required <= set(files):
        raise ValueError('Progressive package missing required files')
    promoted_workflows = validate_human_promotion(files)
    memory_workflow = validate_t8_memory_promotion(files)
    workflow_count = sum(name.startswith('examples/workflows/') and name.endswith('.json') for name in files)
    if workflow_count != EXPECTED_WORKFLOW_COUNT:
        raise ValueError(f'Expected {EXPECTED_WORKFLOW_COUNT} packaged workflows, found {workflow_count}')
    if config['project']['version'] != EXPECTED_RELEASE_VERSION:
        raise ValueError('Candidate version differs from the declared release gate')
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
        't8_memory_workflow': memory_workflow,
        'human_review_scope': 'Only the exact bound short samples and documented composition of their accepted controls, including the bound T8 h4+c2 two-segment8s sample.',
        'universal_quality_claim': False, 'published': False, 'human_qualified': True,
        'source_scope': f"current_local_v{config['project']['version']}_candidate_not_published"}
    with (root/'receipt.json').open('x', encoding='utf8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
    return {k: v for k, v in receipt.items() if k != 'files'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().root), ensure_ascii=False))
