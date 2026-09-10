"""Independent CPU tensor/provenance audit of actual public VAE/FaceRefine probe."""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402
from tools.trt_vae_saved_reference import bind_reference  # noqa: E402


def audit(root):
    # Core opportunistically imports SageAttention, whose import enumerates CUDA.
    # Hide devices only in this independent CPU audit process, not user/Core config.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    def reject_cuda_initialization():
        raise RuntimeError('CPU audit attempted CUDA initialization')
    torch.cuda._lazy_init = reject_cuda_initialization
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    root = Path(root).resolve(strict=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf8'))
    request,result,terminal,resources = [read(n) for n in ('request.json','result.json','terminal.json','resource-summary.json')]
    job = terminal['isolated']
    if (request['schema'] != 't8-trt-public-face73-v1' or terminal['status'] != 'worker_complete_result_requires_audit' or
            job['status'] != 'complete' or job['exit_code'] or job['active_after_cleanup'] or not job['job_assigned_before_task'] or
            resources['status'] != 'observations_within_policy'):
        raise ValueError('Actual worker ownership/resource gate failed')
    for name,digest in request['sources'].items():
        path = Path(name)
        frozen = root/'source-snapshot'/(digest+'-'+path.name) if path.suffix == '.py' else path
        if digest_file(frozen) != digest:
            raise ValueError('Frozen source/artifact changed: '+path.name)
    if bind_reference(request['rgb'],request['latent'],request['native_vae']) != request['reference_sources']:
        raise ValueError('Native RGB provenance changed')
    for name,digest in result['files'].items():
        if Path(name).name != name or digest_file(root/name) != digest:
            raise ValueError('Result tensor artifact changed')
    sys.path.insert(0,request['core'])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    package = types.ModuleType('t8_public_face_audit')
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    face = importlib.import_module(package.__name__+'.face_refine_advanced')
    source = load_file(request['rgb'],device='cpu')['rgb'].movedim(1,-1)[0]
    inputs = load_file(str(root/'inputs.safetensors'),device='cpu')
    outputs = load_file(str(root/'outputs.safetensors'),device='cpu')
    plan = read('face-plan.json')
    rebuilt,crops,*_ = face.build_face_refine_plan(source,24,'manual_static_roi',face.MANUAL_DETECTOR,'cpu',
        .35,.30,.12,.40,.75,.28,.18,4,2,1.4,'384',True,2)
    if rebuilt != plan or not torch.equal(crops,inputs['crops']):
        raise ValueError('Actual crop input or source plan differs')
    audio = load_file(request['audio'],device='cpu')['audio_latent']
    if not torch.equal(audio,inputs['audio']) or not torch.equal(audio,outputs['audio_after']):
        raise ValueError('Locked original audio latent changed')
    for key,shape in {'native_latent':(1,24,22,24,24),'trt_latent':(1,24,22,24,24),
                      'single_image_latent':(1,24,1,24,24),'native_crops':(73,384,384,3),
                      'trt_crops':(73,384,384,3)}.items():
        if tuple(outputs[key].shape) != shape or not torch.isfinite(outputs[key]).all():
            raise ValueError('Invalid actual VAE tensor: '+key)
    for label in ('native','trt'):
        stitched,mask = outputs[label+'_stitched'],outputs[label+'_mask']
        if (tuple(stitched.shape) != tuple(source.shape) or tuple(mask.shape) != (73,512,1024) or
                not torch.isfinite(stitched).all() or not torch.isfinite(mask).all() or
                not torch.count_nonzero(mask) or not torch.equal(stitched[mask==0],source[mask==0])):
            raise ValueError('Actual source pixel protection failed')
        if not bool(((stitched>=0)&(stitched<=1)).all()) or not bool(((mask>=0)&(mask<=1)).all()):
            raise ValueError('Output or mask range invalid')
        report = result['reports'][label+'_stitch']
        if report['fallback_count'] or not report['mask_outside_bit_exact']:
            raise ValueError('Silent fallback not an integration pass')
    if result['native_calls'] != {'native_encode':20,'native_decode':16} or [r['calls'] for r in result['leases']] != [20,16,4,16]:
        raise ValueError('Actual native/public dispatch counts differ')
    expected = [request['encoder'],request['decoder'],request['t1'],request['decoder']]
    for lease,path in zip(result['leases'],expected,strict=True):
        manifest = json.loads((Path(path)/'manifest.json').read_text(encoding='utf8'))
        if lease['status'] != 'complete' or lease['engine_sha256'] != manifest['engine_sha256']:
            raise ValueError('Wrong engine dispatch')
    for key in ('native_injection','full_injection'):
        report = result['reports'][key]
        if not report['audio_tensor_reused'] or not report['noise_mask_object_reused'] or report['implicit_temporal_fit']:
            raise ValueError('Runtime audio/mask identity report failed')
    error = outputs['trt_latent'].float()-outputs['native_latent'].float()
    rms = float(error.square().mean().sqrt()/outputs['native_latent'].float().square().mean().sqrt().clamp_min(1e-12))
    rgb_error = outputs['trt_crops'].float()-outputs['native_crops'].float()
    mse = float(rgb_error.square().mean())
    if torch.cuda.is_initialized():
        raise ValueError('Independent audit must remain CPU-only')
    return {'status':'public_full_decoder_face_roundtrip_tensor_and_cleanup_pass_not_human',
            'frames':73,'canvas':[384,384],'latent_relative_rmse':rms,
            'crop_roundtrip_psnr_db':-10*math.log10(mse) if mse else None,
            'original_audio_tensor_exact':True,'source_pixels_outside_both_masks_exact':True,
            'actual_tile_calls':[20,16,4,16],'worker_checked_full_decoder_bit_equal':result['full_decoder_bit_equal'],
            'timings_seconds':result['timings_seconds'],'resources':resources,
            'files':{name:digest_file(root/name) for name in ('request.json','result.json','terminal.json','resource-summary.json',*result['files'])},
            'scope':'Real public class dispatch with parent-owned GPU leases; actual crop/encode/decode/locked audio latent/stitch. No new FaceRefine sampling or improvement/human claim; single timing observations not a benchmark.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_new_json(args.run_dir/'independent-public-face-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
