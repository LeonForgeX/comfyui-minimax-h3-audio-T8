"""Same short latent: actual loader, scoped decode and CPU MP4+original-audio timing.

AB/BA/AB, fresh VAE objects each call, no cache flush, no sampler. This is a VAE
output-stage benchmark, not whole H3 generation and not an OS-cold benchmark.
"""
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
from tools.prepare_trt_vae_video_media import inspect_video,encode_rgb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    root = parser.parse_args().request.parent
    request = json.loads((root/'request.json').read_text(encoding='utf8'))
    if request['schema'] != 't8-trt-public-timing73-v1' or not request['parent_holds_serial_leases']:
        raise ValueError('Use the owned timing controller')
    for path,digest in request['sources'].items():
        if digest_file(path) != digest:
            raise ValueError('Frozen source changed: '+Path(path).name)
    sys.path.insert(0,request['core'])
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    from safetensors.torch import load_file,save_file
    from dlss_nr_advanced import _packet_copy_video_and_audio
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() != request['gpu_uuid'].removeprefix('GPU-').lower():
        raise ValueError('Device identity mismatch')
    package = types.ModuleType('t8_public_timing_probe')
    package.__path__ = [str(PROJECT)]
    sys.modules[package.__name__] = package
    nodes = importlib.import_module(package.__name__+'.nodes_trt_vae')
    z = load_file(request['latent'],device='cpu')['latent_tensor'].half()
    if tuple(z.shape) != (1,24,22,32,64):
        raise ValueError('Expected complete73-frame0.5MP short latent')
    source = inspect_video(request['source_media'])
    if (source['width'],source['height'],source['frames'],source['fps']) != (1024,512,73,'24'):
        raise ValueError('Original media geometry mismatch')
    records = []
    def check():
        if (root/'cancel.request').exists():
            raise InterruptedError('Timing cancelled')
    for index,label in enumerate(('native','trt','trt','native','native','trt')):
        check()
        case = root/f'{index+1:02d}-{label}'
        case.mkdir()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if label == 'native':
            state,metadata = comfy.utils.load_torch_file(request['native_vae'],return_metadata=True)
            vae = comfy.sd.VAE(sd=state,metadata=metadata,device=torch.device('cuda:0'),dtype=torch.float16)
            del state
            vae.throw_exception_if_invalid()
            tile_calls = [0]
            def count(module,inputs):
                tile_calls[0] += 1
            hook = vae.first_stage_model.decoder.register_forward_pre_hook(count)
        else:
            node_output = nodes.MiniMaxH3TRTVAEDecoderEXPT8.execute(Path(request['native_vae']).name,
                request['decoder'],request['runtime_site'],1024)
            vae = node_output.result[0]
            del node_output
            backend = vae.open_backend
            backend.serial_lease = lambda:SerialProbeLease(Path(tempfile.gettempdir())/'T8-TRT-public-timing-worker.lock')
            backend.check = vae.check = check
            leases = []
            @contextmanager
            def lease(kind,shapes):
                try:
                    with backend(kind,shapes) as call:
                        yield call
                finally:
                    leases.append(copy.deepcopy(backend.last_report))
            vae.open_backend = lease
        loaded = time.perf_counter()
        with torch.inference_mode():
            rgb = vae.decode(z)[0].movedim(-1,0).unsqueeze(0).cpu().contiguous()
        torch.cuda.synchronize()
        decoded = time.perf_counter()
        if tuple(rgb.shape) != (1,3,73,512,1024) or not torch.isfinite(rgb).all():
            raise ValueError('Invalid complete decode')
        tensor = case/'rgb.safetensors'
        save_file({'rgb':rgb},str(tensor))
        saved = time.perf_counter()
        encode_rgb(case/'video-only.mp4',tensor,tuple(rgb.shape),source)
        _packet_copy_video_and_audio(case/'video-only.mp4',Path(request['source_media']),case/'review.mp4')
        exported = time.perf_counter()
        row = {'case':case.name,'route':label,'sequence_index':index,
            'seconds':{'load':loaded-started,'decode_including_transfer':decoded-loaded,
                'verify_and_tensor_save':saved-decoded,'mp4_and_audio_copy':exported-saved,
                'load_to_final_mp4':exported-started},
            'files':{name:digest_file(case/name) for name in ('rgb.safetensors','video-only.mp4','review.mp4')}}
        if label == 'native':
            hook.remove()
            row['actual_tile_calls'] = tile_calls[0]
        else:
            row['leases'] = leases
            if len(leases) != 1 or leases[0]['status'] != 'complete':
                raise ValueError('Scoped engine did not finish exactly one lease')
            row['actual_tile_calls'] = leases[0]['calls']
        if row['actual_tile_calls'] != 60:
            raise ValueError('Expected60 actual tile calls each complete short decode')
        mm.unload_all_models()  # Only models in this controller-owned worker.
        native = vae if label == 'native' else vae.native_vae
        native.first_stage_model.to('cpu')
        del native,vae,rgb
        if label == 'trt':
            del lease,backend,leases
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        row['seconds']['post_export_hash_and_cleanup'] = time.perf_counter()-exported
        write_new_json(case/'result.json',row)
        records.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('files','leases')},ensure_ascii=False),flush=True)
    write_new_json(root/'result.json',{'status':'six_serial_output_stage_measurements_require_independent_audit',
        'records':records,'order':['native','trt','trt','native','native','trt'],
        'scope':'73frames0.5MP same existing latent; real fresh nativeVAE/publicDecoder loading,60tile decode,CPU tensor save,identical x264 and originalaudio copy. Parent owns serialGPU locks. No resampling,whole-generation or OS-cold-cache claim. Library/kernel/OS file caches remain; first observations and later calls reported separately. Validation hashes and cleanup are outside load-to-MP4 measurement; runtime own checks remain inside.'})


if __name__ == '__main__':
    main()
