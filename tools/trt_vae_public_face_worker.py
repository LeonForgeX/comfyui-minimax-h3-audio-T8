"""Actual public VAE classes + FaceRefine on existing73-frame media, no sampler."""
import argparse
from contextlib import contextmanager
import copy
import gc
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402
from dlss_fi_backend.resources import SerialProbeLease  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    root = parser.parse_args().request.parent
    request = json.loads((root/'request.json').read_text(encoding='utf8'))
    if request['schema'] != 't8-trt-public-face73-v1' or not request['parent_holds_serial_leases']:
        raise ValueError('Use the owned public-interface controller')
    for path,digest in request['sources'].items():
        if digest_file(path) != digest:
            raise ValueError('Public interface source changed: '+Path(path).name)
    sys.path.insert(0,request['core'])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.nested_tensor
    import comfy.model_management as mm
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() != request['gpu_uuid'].removeprefix('GPU-').lower():
        raise ValueError('CUDA device differs from controller NVML device')
    package = types.ModuleType('t8_public_face_probe')
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    nodes = importlib.import_module(package.__name__+'.nodes_trt_vae')
    face = importlib.import_module(package.__name__+'.face_refine_advanced')
    with safe_open(request['rgb'],framework='pt',device='cpu') as saved:
        frames = saved.get_tensor('rgb').movedim(1,-1)[0].contiguous()
    if tuple(frames.shape) != (73,512,1024,3):
        raise ValueError('Expected complete native0.5MP short clip')
    audio = load_file(request['audio'],device='cpu')['audio_latent']
    if tuple(audio.shape) != (1,32,2,122):
        raise ValueError('Expected matching original73-frame audio latent')
    original_audio = audio.clone()
    plan,crops,*_ = face.build_face_refine_plan(frames,24,'manual_static_roi',face.MANUAL_DETECTOR,'cpu',
        .35,.30,.12,.40,.75,.28,.18,4,2,1.4,'384',True,2)
    if len(plan['shots']) != 1 or tuple(crops.shape) != (73,384,384,3):
        raise ValueError('This fixed fixture must be one complete73-frame384px repair window')
    write_new_json(root/'face-plan.json',plan)
    save_file({'crops':crops,'audio':audio},str(root/'inputs.safetensors'))
    template = torch.zeros(1,24,22,24,24)
    masks = comfy.nested_tensor.NestedTensor((torch.ones_like(template),torch.zeros_like(audio)))
    av = {'samples':comfy.nested_tensor.NestedTensor((template,audio)),'noise_mask':masks}
    def check():
        if (root/'cancel.request').exists():
            raise InterruptedError('Public VAE interface test cancelled')
    timings,counts,reports,leases = {},{}, {},[]
    def timed(name,fn):
        check()
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter()-started
        return value
    state,metadata = comfy.utils.load_torch_file(request['native_vae'],return_metadata=True)
    native = comfy.sd.VAE(sd=state,metadata=metadata,device=torch.device('cuda:0'),dtype=torch.float16)
    del state
    native.throw_exception_if_invalid()
    def hook(name):
        counts[name] = 0
        def count(module,inputs):
            counts[name] += 1
        return count
    h1 = native.first_stage_model.encoder.register_forward_pre_hook(hook('native_encode'))
    h2 = native.first_stage_model.decoder.register_forward_pre_hook(hook('native_decode'))
    print('T8_PUBLIC_FACE_PHASE native_encode_decode',flush=True)
    with torch.inference_mode():
        _,native_av,report = timed('native_inject_encode',lambda native=native:face.inject_face_refine_video_latent([],av,crops,native,plan,'require_locked',False))
        reports['native_injection'] = json.loads(report)
        native_z = native_av['samples'].unbind()[0]
        native_crops = timed('native_decode',lambda native=native:native.decode(native_z))[0].cpu()
    h1.remove()
    h2.remove()
    mm.unload_all_models()  # Only this isolated worker has models registered.
    native.first_stage_model.to('cpu')
    del native,native_av
    gc.collect()
    torch.cuda.empty_cache()

    def observe(vae,label):
        backend = vae.open_backend
        # Parent already owns the cross-copy GPU leases. Keep a distinct child
        # lock rather than attempting to acquire those same OS locks twice.
        backend.serial_lease = lambda:SerialProbeLease(Path(tempfile.gettempdir())/'T8-TRT-public-face-worker.lock')
        backend.check = check
        vae.check = check
        @contextmanager
        def open_backend(kind,shapes):
            try:
                with backend(kind,shapes) as call:
                    yield call
            finally:
                leases.append(dict(label=label,**copy.deepcopy(backend.last_report)))
        vae.open_backend = open_backend
        return vae

    print('T8_PUBLIC_FACE_PHASE actual_full_public_node',flush=True)
    result = timed('full_node_load',lambda:nodes.MiniMaxH3TRTVAEFullEXPT8.execute(
        Path(request['native_vae']).name,request['decoder'],request['t1'],request['encoder'],request['runtime_site'],1024))
    full = observe(result.result[0],'full')
    reports['full_loader'] = json.loads(result.result[1])
    with torch.inference_mode():
        _,trt_av,report = timed('full_inject_encode',lambda:face.inject_face_refine_video_latent([],av,crops,full,plan,'require_locked',False))
        reports['full_injection'] = json.loads(report)
        trt_z = trt_av['samples'].unbind()[0]
        trt_crops = timed('full_decode',lambda:full.decode(trt_z))[0].cpu()
        single_z = timed('full_image_encode',lambda:full.encode(crops[:1]))
    result = timed('decoder_node_load',lambda:nodes.MiniMaxH3TRTVAEDecoderEXPT8.execute(
        Path(request['native_vae']).name,request['decoder'],request['runtime_site'],1024))
    decoder = observe(result.result[0],'decoder_only')
    with torch.inference_mode():
        duplicate = timed('decoder_only_decode',lambda:decoder.decode(trt_z))[0].cpu()
    if not torch.equal(duplicate,trt_crops):
        raise ValueError('Actual decoder-only and full public outputs differ for the same latent')
    del duplicate
    if not torch.equal(audio,original_audio) or trt_av['samples'].unbind()[1].data_ptr() != audio.data_ptr() or trt_av['noise_mask'] is not masks:
        raise ValueError('Public face injection changed locked audio or mask identity')
    if [r['calls'] for r in leases] != [20,16,4,16] or counts != {'native_encode':20,'native_decode':16}:
        raise ValueError('Actual20video/16decode/4image tile dispatch differs')
    for item in leases:
        if item['status'] != 'complete' or item['free_before_deserialize']-item['free_after_release'] > 512*1024**2:
            raise ValueError('Public backend lease/cleanup failed')
    outputs = {'native_latent':native_z,'trt_latent':trt_z,'single_image_latent':single_z,
               'native_crops':native_crops,'trt_crops':trt_crops,'audio_after':audio}
    print('T8_PUBLIC_FACE_PHASE actual_cpu_stitch',flush=True)
    for label,decoded in (('native',native_crops),('trt',trt_crops)):
        stitched,mask,fallback,count,report = timed(label+'_stitch',lambda:face.stitch_face_refine_candidate(
            frames,decoded,plan,'ellipse',4.,1.,0.,.5,0,'cpu_memory_safe'))
        if count or torch.count_nonzero(fallback) or not torch.count_nonzero(mask) or not torch.equal(stitched[mask==0],frames[mask==0]):
            raise ValueError('Stitch fallback or source pixel protection failed')
        outputs[label+'_stitched'],outputs[label+'_mask'] = stitched,mask
        reports[label+'_stitch'] = json.loads(report)
    if any(not bool(torch.isfinite(value).all()) for value in outputs.values()):
        raise ValueError('Nonfinite public interface result')
    timed('save_outputs',lambda:save_file({k:v.detach().cpu().contiguous() for k,v in outputs.items()},str(root/'outputs.safetensors')))
    report = {'status':'actual_public_vae_face_roundtrip_requires_independent_audit','timings_seconds':timings,
        'native_calls':counts,'leases':leases,'reports':reports,'full_decoder_bit_equal':True,
        'files':{name:digest_file(root/name) for name in ('face-plan.json','inputs.safetensors','outputs.safetensors')},
        'limits':'Actual public Full and Decoder classes plus FaceRefine crop/injection/CPU stitch; original73-frame audio latent locked. No new sampler, face-improvement claim, audio decode, browser, cold/warm benchmark or long-video validation. Child backend serial lease delegated to owning parent.'}
    write_new_json(root/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('reports','files')},indent=2),flush=True)


if __name__ == '__main__':
    main()
