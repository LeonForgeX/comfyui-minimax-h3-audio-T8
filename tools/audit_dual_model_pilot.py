"""Independent read-back of actual stage bytes, calls and final short media."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file


def validate_memory_backend(backend, terminal, nfe=4):
    memory = backend.get('memory_composition', {})
    heads = terminal.get('memory_head_chunks', 1)
    ffn = terminal.get('memory_ffn_chunks', 1)
    expected_ffn = [2, 4096] if ffn == 2 else None
    if (heads not in (1, 4) or ffn not in (1, 2)
            or memory.get('kind') != 'kj_memory_sage' or memory.get('head_chunks') != heads
            or memory.get('ffn_settings') != expected_ffn
            or backend['completed_calls'].get('sage:unbiased') != 50 * nfe * heads
            or not memory.get('source_sha256s')
            or backend.get('head_grouping') != 'inside_delegate; full-head Relay/EAV once per block'):
        raise ValueError('Required authenticated KJ H3 memory composition was not used')


def validate_eav_stage(report):
    audit = report.get('eav_audit', {})
    config = audit.get('config', {})
    if (report.get('completed_network_forwards') != 20
            or report.get('schedule', {}).get('mode') != 'full_stock20'
            or audit.get('status') != 'apply_exp_long_video_segment_verified'
            or audit.get('aborted') or audit.get('model_forward_count') != 20
            or audit.get('attention_calls_per_active_forward') != [50] * 20
            or config.get('mode') != 'apply_exp' or config.get('sampling_profile') != 'stock20'
            or config.get('sigma_contract', {}).get('nfe') != 20
            or config.get('composed_attention_backend') != report.get('backend')
            or audit.get('long_video', {}).get('context_frames') != 0
            or audit.get('long_video', {}).get('segment_index') != 0):
        raise ValueError('Actual combined EAV Stock20 stage audit mismatch')
    return {'model_forwards': 20, 'active_block_measurements': 1000,
            'backend_observation': report.get('backend_observation')}


def audit_first_frame(terminal, records, high_condition=None):
    reference = terminal.get('first_frame_file')
    if reference is None:
        return None
    source = Path(reference['path']).resolve(strict=True)
    if (source.stat().st_size != reference['bytes']
            or hashlib.sha256(source.read_bytes()).hexdigest() != reference['sha256']):
        raise ValueError('Declared first-frame file no longer matches the sampled reference')
    tasks = []
    for name in ('low_x0', 'high_output'):
        condition = (high_condition if name == 'high_output' and high_condition is not None
                     else records[name]['report']['conditioning'])
        if (condition.get('task') != 'i2va' or condition.get('context_active') is not False
                or condition.get('segment_index') != 0 or condition.get('warnings')):
            raise ValueError('Both actual stages must report initial I2VA conditioning without ignored-reference warnings')
        tasks.append(condition['task'])
    return {'file_sha256': reference['sha256'], 'actual_stage_tasks': tasks,
            'scope': 'first-segment file identity and stage conditioning; not perceptual image adherence'}


def read_bound_high_condition(root, records):
    """High condition is stored by the delivery loop in its hashed effects sidecar."""
    paths = list((root / 'output').rglob('effects_audit.json'))
    if len(paths) != 1:
        raise ValueError('Expected exactly one high-stage effects sidecar')
    payload = json.loads(paths[0].read_text(encoding='utf8'))
    claimed = payload.pop('audit_sha256', None)
    actual = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()
    if claimed != actual:
        raise ValueError('High-stage effects sidecar checksum mismatch')
    if (payload.get('schema') != 1 or payload.get('format') != 'minimax_h3_t8_in_node_loop_effects_segment'
            or payload.get('contract_sha256') != records['low_x0']['contract']['job']
            or payload.get('segment_index') != 0 or payload.get('candidate_id') != paths[0].parent.name):
        raise ValueError('High-stage condition belongs to another job or candidate')
    first = payload['sampling_plan']['dual_model']['first_pass']
    second = payload['sampling_plan']['dual_model']['second_pass']
    if first != records['low_x0']['report'] or second != records['high_output']['report']:
        raise ValueError('Effects sidecar does not bind the audited low and high stage reports')
    return payload['conditioning'], {'path': str(paths[0]), 'audit_sha256': claimed}


def audit_distinct_base(root, terminal):
    reference = terminal.get('second_base_file')
    if reference is None:
        return None
    graph = json.loads((root / 'generation/prompt.json').read_text(encoding='utf-8'))
    expected = json.loads((root / 'expected.json').read_text(encoding='utf-8'))['pilot_graphs']['dual_short']
    if graph != expected:
        raise ValueError('Executed graph differs from the preflight recipe')
    if (graph['1']['class_type'] != 'UNETLoader' or graph['28']['class_type'] != 'UNETLoader'
            or graph['1']['inputs']['unet_name'] == graph['28']['inputs']['unet_name']
            or graph['2']['inputs']['model'] != ['1', 0]
            or graph['3']['inputs']['model'] != ['28', 0]
            or graph['21']['inputs']['model'] != ['2', 0]
            or graph['22']['inputs']['model'] != ['3', 0]
            or graph['8']['inputs']['model_pass1'] != ['21', 0]
            or graph['8']['inputs']['model_pass2'] != ['22', 0]):
        raise ValueError('Independent loaded base/LoRA/stage bindings changed')
    paths = list((root / 'output').rglob('manifest.json'))
    if len(paths) != 1:
        raise ValueError('Expected one independently bound manifest')
    manifest = json.loads(paths[0].read_text(encoding='utf-8'))
    if len(manifest['segments']) != 1:
        raise ValueError('Distinct base audit is a one-segment case')
    identities = manifest['segments'][0]['model_id'].split(':')
    if len(identities) != 2 or any(len(x) != 16 for x in identities) or identities[0] == identities[1]:
        raise ValueError('Actual content-based MODEL identities are not distinct')
    source = Path(reference['path']).resolve(strict=True)
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if source.stat().st_size != reference['bytes'] or digest.hexdigest() != reference['sha256']:
        raise ValueError('Independently read second-base weights no longer match')
    return {'second_base_sha256': digest.hexdigest(), 'second_base_path': str(source),
            'actual_model_content_identity_prefixes': identities,
            'separate_UNET_loader_and_LoRA_bindings': True,
            'disable_pinned_memory': terminal.get('disable_pinned_memory', False),
            'scope': 'one explicit installed FL2VA/Ref2VA pair; not arbitrary model-family compatibility'}


def validate_audio_stage_contract(records, tensors):
    low, prepared, high = (tensors[key] for key in ('low_x0', 'high_input', 'high_output'))
    if not torch.equal(low['samples_audio'], prepared['samples_audio']):
        raise ValueError('First-pass audio was not preserved at the stage handoff')
    delta = (low['samples_audio'] - high['samples_audio']).abs().max().item()
    contract = records['high_output']['contract']
    policy = contract.get('audio_policy', {})
    if contract.get('audio_policy_version') == 2 and policy.get('effective_source') == 'legacy_policy':
        if records['low_x0']['contract'].get('audio_policy') != policy:
            raise ValueError('Audio policy differs between stages')
        mask = prepared.get('mask_audio')
        if mask is not None and not bool(torch.all(mask == 1)):
            raise ValueError('Short native joint continuation unexpectedly freezes audio')
        if (records['low_x0']['report']['schedule']['coarse_audio_sigmas'][-1] <= 0
                or records['high_output']['report']['schedule']['refine_audio_sigmas'][-1] != 0):
            raise ValueError('Partial first stage must finish its audio trajectory in pass2')
        if records['high_output']['report']['audio_delivery']['source'] != 'completed_second_pass_output':
            raise ValueError('Final audio was replaced by the unfinished first pass')
        if delta <= 1e-6:
            raise ValueError('Joint audio has not measurably evolved beyond coarse x0')
        return {'mode': 'joint_second_pass_continuation', 'max_absolute_delta': delta,
                'lock_tolerance': None, 'quality': 'requires_human_review'}
    if 'mask_audio' not in prepared or torch.count_nonzero(prepared['mask_audio']):
        raise ValueError('First-pass audio was not copied and locked at high-stage input')
    if delta > 1e-6:
        raise ValueError(f'Locked audio changed beyond numerical roundtrip tolerance: {delta}')
    return {'mode': 'first_pass_locked', 'max_absolute_delta': delta, 'lock_tolerance': 1e-6,
            'quality': 'requires_human_review'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chain', help='Explicit single chain inside a multi-invocation benchmark output')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new audit output')
    torch.set_num_threads(2)
    root = args.root.resolve()
    output_root = root / 'output'
    if args.chain:
        if Path(args.chain).name != args.chain or args.chain in {'.', '..'}:
            raise ValueError('Chain must be one output directory name')
        output_root = (output_root / 'minimax_h3_t8_long_video' / args.chain).resolve(strict=True)
        if not output_root.is_relative_to(root / 'output'):
            raise ValueError('Chain escaped the measured output')
    terminal = json.loads((root/'terminal.json').read_text(encoding='utf-8'))
    if not terminal['status'].startswith('generation_completed') or terminal['server_stop']['owned_children_remaining']:
        raise ValueError('Requires a completed generation and owned process cleanup')
    records, tensors = {}, {}
    for stage in ('low_x0', 'high_input', 'high_output'):
        paths = list(output_root.rglob(stage+'-*.json'))
        if len(paths) != 1:
            raise ValueError('This audit requires exactly one completed segment')
        record = json.loads(paths[0].read_text(encoding='utf-8'))
        path = (paths[0].parent / record['tensor_file']).resolve()
        if not path.is_relative_to(root/'output') or hashlib.sha256(path.read_bytes()).hexdigest() != record['tensor_sha256']:
            raise ValueError('Stage checkpoint byte identity mismatch')
        records[stage], tensors[stage] = record, load_file(path, device='cpu')
        if any(not torch.isfinite(value).all() for value in tensors[stage].values()):
            raise ValueError('Nonfinite checkpoint')
    low, prepared, high = (tensors[key] for key in ('low_x0', 'high_input', 'high_output'))
    if low['samples_video'].shape[-2:] != (16, 32) or high['samples_video'].shape[-2:] != (32, 64):
        raise ValueError('Unexpected low/high geometry')
    audio_contract = validate_audio_stage_contract(records, tensors)
    audio_delta = audio_contract['max_absolute_delta']
    backend_counts = []
    first_nfe = 20 if terminal.get('eav_stock20') else 4
    eav_proof = validate_eav_stage(records['low_x0']['report']) if terminal.get('eav_stock20') else None
    for stage in ('low_x0', 'high_output'):
        r = records[stage]['report']
        counts = r['backend']['completed_calls']
        nfe = first_nfe if stage == 'low_x0' else 4
        if r['completed_network_forwards'] != nfe:
            raise ValueError('Actual model invocation count mismatch')
        if terminal.get('relay'):
            if (not counts.get('sage:biased', 0) or sum(counts.values()) < 200
                    or set(counts) - {'sage:biased', 'sage:unbiased'}
                    or r['backend']['kind'] != 'audited_kj_selector'):
                raise ValueError('Relay did not execute the measured biased Sage route without fallback')
            if terminal.get('backend') == 'kj-memory':
                validate_memory_backend(r['backend'], terminal, nfe)
        elif terminal.get('backend', 'kj') == 'sol':
            if counts != {'sol:completed': 200} or r['backend']['kind'] != 'audited_sol_attn_selector':
                raise ValueError('Actual plain Sol kernel count mismatch or fallback')
        elif terminal.get('backend', 'kj') == 'pytorch':
            if counts != {'pytorch:completed': 200} or r['backend']['kind'] != 'native_core_pytorch_selector':
                raise ValueError('Actual native PyTorch call count mismatch')
        elif counts != {'sage:unbiased': 200}:
            raise ValueError('Actual plain KJ kernel count mismatch')
        backend_counts.append(counts)
    high_condition, condition_proof = (read_bound_high_condition(root, records)
                                      if terminal.get('first_frame_file') else (None, None))
    reference_audit = audit_first_frame(terminal, records, high_condition)
    distinct_base_audit = audit_distinct_base(root, terminal)
    if reference_audit is not None:
        reference_audit['high_condition_sidecar'] = condition_proof
    files = list(output_root.rglob('assembled/*.mp4'))
    if len(files) != 1:
        raise ValueError('Expected one final assembled clip')
    media = files[0]
    probe = subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(media)],
                           capture_output=True, text=True, check=True)
    data = json.loads(probe.stdout)
    video = next(stream for stream in data['streams'] if stream['codec_type']=='video')
    audio = next(stream for stream in data['streams'] if stream['codec_type']=='audio')
    if (video['width'],video['height'],int(video['nb_read_frames']),Fraction(video['avg_frame_rate'])) != (1024,512,72,Fraction(24)):
        raise ValueError('Final video dimensions/frame timeline differs')
    checks = []
    for selection in (['-map','0:v:0'], ['-map','0:a:0'], ['-map','0:v:0','-map','0:a:0']):
        result = subprocess.run(['ffmpeg','-v','error','-threads','4','-xerror','-err_detect','explode','-i',str(media),
                                 *selection,'-f','null','-'],capture_output=True,text=True)
        if result.returncode or result.stderr.strip():
            raise RuntimeError('Strict decoder error: '+result.stderr)
        checks.append(selection)
    payload = {'status':'short_mechanical_audit_pass_human_pending', 'relay': bool(terminal.get('relay')),
        'selected_chain': args.chain,
        'backend': terminal.get('backend', 'kj'),
        'first_frame_audit': reference_audit,
        'distinct_base_audit': distinct_base_audit,
        'stage_sha256':{k:v['tensor_sha256'] for k,v in records.items()}, 'actual_forwards':[first_nfe,4],
        'eav_stock20_audit': eav_proof,
        'backend_completed_calls':backend_counts, 'audio_input_bit_exact': True,
        'memory_compositions': [records[stage]['report']['backend'].get('memory_composition')
                                for stage in ('low_x0', 'high_output')],
        'audio_output_bit_exact':torch.equal(low['samples_audio'],high['samples_audio']),
        'audio_output_max_absolute_delta':audio_delta,'audio_output_roundtrip_tolerance':audio_contract['lock_tolerance'],
        'audio_contract': audio_contract,
        'media':{'path':str(media),'sha256':hashlib.sha256(media.read_bytes()).hexdigest(),
                 'frames':72,'width':1024,'height':512,'fps':'24/1','video_seconds':video['duration'],
                 'audio_seconds':audio['duration'],'container_seconds':data['format']['duration']},
        'strict_decode':checks,'cuda_initialized':torch.cuda.is_initialized(),
        'scope':'one short declared route; not continuation, lipsync, subjective quality or speed comparison'}
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False))


if __name__ == '__main__':
    main()
