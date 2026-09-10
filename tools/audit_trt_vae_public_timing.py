"""Independently qualify saved RGB, original audio and real output-stage timings."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402


def summarize(rows):
    expected = ['native','trt','trt','native','native','trt']
    if [r['route'] for r in rows] != expected or [r['sequence_index'] for r in rows] != list(range(6)):
        raise ValueError('Expected fixed serial AB/BA/AB order')
    stages = ('load','decode_including_transfer','verify_and_tensor_save','mp4_and_audio_copy')
    for row in rows:
        seconds = row['seconds']
        if row['actual_tile_calls'] != 60 or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in seconds.values()):
            raise ValueError('Invalid actual call count or timing')
        if not math.isclose(sum(seconds[k] for k in stages),seconds['load_to_final_mp4'],abs_tol=1e-6):
            raise ValueError('Stage times do not cover claimed output-stage total')
    result = {}
    for label in ('native','trt'):
        selected = [r['seconds'] for r in rows if r['route'] == label]
        result[label] = {'first_observation':selected[0],
            'later_two_medians':{k:statistics.median(r[k] for r in selected[1:]) for k in selected[0]},
            'all_three_medians':{k:statistics.median(r[k] for r in selected) for k in selected[0]}}
    a,b = (result[label]['later_two_medians']['load_to_final_mp4'] for label in ('native','trt'))
    result['later_output_stage_speed_ratio'] = a/b
    result['later_output_stage_time_reduction_fraction'] = 1-b/a
    result['limits'] = 'Fresh VAE each call; natural file/library caches,first observations not OS-cold. Later only two calls each. Includes loader/runtime checks,decode,tensor save,CPU x264 and originalaudio copy; excludes H3 sampling,compilation,import/startup,post-export evidence hashes and cleanup. Not whole-generation speed or pure GPU/VRAM benchmark.'
    return result


def audit(root):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    from safetensors.torch import load_file
    from tools.trt_vae_saved_reference import bind_reference
    from tools.prepare_trt_vae_video_media import inspect_video
    from dlss_nr_advanced import _audio_packet_digests,_audio_pcm_digests,_validate_audio_identity
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    request,result,terminal,resources = [read(n) for n in ('request.json','result.json','terminal.json','resource-summary.json')]
    job = terminal['isolated']
    if (request['schema'] != 't8-trt-public-timing73-v1' or terminal['status'] != 'worker_complete_result_requires_audit' or
        job['status'] != 'complete' or job['exit_code'] or job['active_after_cleanup'] or not job['job_assigned_before_task'] or
        resources['status'] != 'observations_within_policy'):
        raise ValueError('Owned serial run or resource gate failed')
    for name,digest in request['sources'].items():
        path = Path(name)
        frozen = root/'source-snapshot'/(digest+'-'+path.name) if path.suffix == '.py' else path
        if digest_file(frozen) != digest:
            raise ValueError('Frozen timing input/source changed: '+path.name)
    if bind_reference(request['rgb'],request['latent'],request['native_vae']) != request['reference_sources']:
        raise ValueError('Original latent/RGB provenance changed')
    timings = summarize(result['records'])
    original = inspect_video(request['source_media'])
    packets,pcm = _audio_packet_digests(request['source_media']),_audio_pcm_digests(request['source_media'])
    native = load_file(request['rgb'],device='cpu')['rgb']
    rows = []
    for row in result['records']:
        case = root/row['case']
        if case.parent != root or read(row['case']+'/result.json') != row:
            raise ValueError('Per-call timing record changed')
        for name,digest in row['files'].items():
            if Path(name).name != name or digest_file(case/name) != digest:
                raise ValueError('Timing output changed')
        rgb = load_file(str(case/'rgb.safetensors'),device='cpu')['rgb']
        if tuple(rgb.shape) != (1,3,73,512,1024) or not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
            raise ValueError('Invalid complete RGB result')
        square_sum,maximum = 0.,0.
        for index in range(73):
            delta = (rgb[:,:,index]-native[:,:,index]).double()
            square_sum += float(delta.square().sum())
            maximum = max(maximum,float(delta.abs().max()))
        mse = square_sum/rgb.numel()
        del rgb
        if row['route'] == 'trt':
            lease = row['leases'][0]
            manifest = json.loads((Path(request['decoder'])/'manifest.json').read_text())
            if len(row['leases']) != 1 or lease['calls'] != 60 or lease['status'] != 'complete' or lease['engine_sha256'] != manifest['engine_sha256']:
                raise ValueError('Wrong actual public engine dispatch')
        video = inspect_video(case/'review.mp4')
        if video != original:
            raise ValueError('Final media frames/PTS/geometry changed')
        audio = _validate_audio_identity(packets,_audio_packet_digests(case/'review.mp4'),pcm,_audio_pcm_digests(case/'review.mp4'))
        rows.append({'case':row['case'],'psnr_vs_bound_native_db':-10*math.log10(mse) if mse else None,
                     'max_abs_error':maximum,'audio':audio,'file_sha256':row['files']['review.mp4']})
    if torch.cuda.is_initialized():
        raise ValueError('Independent timing audit must be CPU-only')
    return {'status':'six_serial_public_output_stage_media_and_timings_verified_not_human',
        'timings':timings,'media':rows,'resources':resources,'cuda_initialized':False,
        'files':{name:digest_file(root/name) for name in ('request.json','result.json','terminal.json','resource-summary.json')},
        'first_observation_caveat':'A CPU provenance audit overlapped part of the first pair; do not treat that pair as clean cold-cache performance. Later observations occurred after that audit terminated.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir/'independent-public-timing-audit.json',report)
    print(json.dumps(report,indent=2))
