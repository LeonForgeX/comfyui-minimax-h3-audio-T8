"""Guarded serial public-node/FaceRefine short integration; no new sampling."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
RESEARCH = PROJECT/'artifacts/acceleration-research-20260909'
sys.path.insert(0,str(PROJECT))
from trt_vae_build import digest_file,write_new_json  # noqa: E402
from dlss_fi_backend.process import run_isolated,IsolatedTaskError  # noqa: E402
from dlss_fi_backend.resources import GuardPolicy,NvmlResourceReader,ResourceGuard,SerialProbeLease  # noqa: E402
from tools.trt_vae_saved_reference import bind_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rgb','latent','audio','native-vae','decoder','t1','encoder','runtime-site','run-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--task',choices=('face','timing','text'),default='face')
    parser.add_argument('--font',type=Path,help='Explicit installed font for the synthetic text fixture only')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH) or shutil.disk_usage(RESEARCH).free < 8*1024**3:
        raise ValueError('New research directory and8GiB disk required')
    reference = bind_reference(args.rgb,args.latent,args.native_vae)
    capture_path = args.latent.parent/'capture.json'
    capture = json.loads(capture_path.read_text(encoding='utf8'))
    if capture['status'] != 'actual_sampler_output_captured_bit_exact':
        raise ValueError('Need actual same sampler AV capture')
    for name,path in (('video',args.latent),('audio',args.audio)):
        entry = capture['files'][name]
        if Path(entry['path']).resolve() != path.resolve() or digest_file(path) != entry['sha256']:
            raise ValueError('Audio/video capture identity differs')
    core = PROJECT.parents[1]
    worker = Path(__file__).with_name({'face':'trt_vae_public_face_worker.py',
        'timing':'trt_vae_public_timing_worker.py','text':'trt_vae_public_text_worker.py'}[args.task])
    files = [Path(__file__),worker,capture_path,*map(Path,reference),args.audio,args.native_vae,
             *PROJECT.glob('*.py'),*list((PROJECT/'dlss_fi_backend').glob('*.py'))]
    files += [core/name for name in ('comfy/ldm/minimax/vae.py','comfy/sd.py','comfy/model_management.py',
              'comfy/model_patcher.py','comfy/utils.py','comfy/ops.py','comfy/nested_tensor.py','folder_paths.py')]
    files += [path for path in args.runtime_site.rglob('*') if path.is_file() and path.suffix in ('.py','.pyd','.dll')]
    if args.task == 'text':
        if not args.font or not args.font.is_file():
            raise ValueError('Text fixture needs an explicit existing font')
        files.append(args.font)
    media = None
    if args.task == 'timing':
        media_audit = args.rgb.parent/'media-audit.json'
        evidence = json.loads(media_audit.read_text(encoding='utf8'))
        if evidence['status'] != 'complete_video_and_original_audio_verified_not_human_qualification':
            raise ValueError('Audited original audio source required for timing')
        media = Path(evidence['source']).resolve(strict=True)
        if digest_file(media) != evidence['source_sha256']:
            raise ValueError('Original source media changed')
        files += [media_audit,media,Path(__file__).with_name('prepare_trt_vae_video_media.py')]
    for bundle in (args.decoder,args.t1,args.encoder):
        files.extend((bundle/'manifest.json',bundle/'model.engine'))
    request = {'schema':{'face':'t8-trt-public-face73-v1','timing':'t8-trt-public-timing73-v1',
               'text':'t8-trt-public-text1-v1'}[args.task],'core':str(core),'parent_holds_serial_leases':True,
               'reference_sources':reference,'sources':{str(path.resolve()):digest_file(path) for path in files}}
    if args.task == 'text':
        request['font'] = str(args.font.resolve())
    if media:
        request['source_media'] = str(media)
    for key in ('rgb','latent','audio','native_vae','decoder','t1','encoder','runtime_site'):
        request[key] = str(getattr(args,key).resolve())
    root.mkdir(parents=True)
    snapshot = root/'source-snapshot'
    snapshot.mkdir()
    for path,digest in request['sources'].items():
        if Path(path).suffix == '.py':
            target = snapshot/(digest+'-'+Path(path).name)
            shutil.copyfile(path,target)
            if digest_file(target) != digest:
                raise ValueError('Source snapshot mismatch')
    terminal = {'status':'not_started'}
    try:
        with ExitStack() as stack:
            for lock in (RESEARCH/'serial-gpu.lock',Path(tempfile.gettempdir())/'T8-DLSS-FI-serial.lock',Path(tempfile.gettempdir())/'T8-TRT-VAE-serial.lock'):
                stack.enter_context(SerialProbeLease(lock))
            reader = stack.enter_context(NvmlResourceReader())
            guard = ResourceGuard(GuardPolicy(startup_free_gpu_bytes=10*1024**3,
                minimum_free_gpu_bytes=1024**3,minimum_free_ram_bytes=8*1024**3))
            first = reader.sample()
            reason = guard.observe(first,startup=True)
            write_new_json(root/'preflight.json',{'snapshot':first,'reason':reason})
            if reason:
                raise RuntimeError(reason)
            request['gpu_uuid'] = first['gpu_uuid']
            write_new_json(root/'request.json',request)
            if not args.execute:
                terminal = {'status':'preflight_only'}
                return
            cancel = threading.Event()
            last = time.perf_counter()
            with (root/'resources.jsonl').open('x',encoding='utf8') as log:
                log.write(json.dumps(first)+'\n')
                def check():
                    nonlocal last
                    if (root/'cancel.request').exists():
                        cancel.set()
                        raise InterruptedError('Public face probe cancelled')
                    if time.perf_counter()-last < .25:
                        return
                    row = reader.sample()
                    last = time.perf_counter()
                    log.write(json.dumps(row)+'\n')
                    log.flush()
                    if error := guard.observe(row):
                        raise RuntimeError(error)
                try:
                    job = run_isolated(worker,['--request',str(root/'request.json')],timeout=900,cancel=cancel,check=check)
                    terminal = {'status':'worker_complete_result_requires_audit','isolated':job}
                finally:
                    write_new_json(root/'resource-summary.json',guard.report())
    except IsolatedTaskError as error:
        terminal = {'status':'worker_failed','isolated':error.receipt}
        raise
    except BaseException as error:
        terminal = {'status':'controller_failed','error':str(error)}
        raise
    finally:
        write_new_json(root/'terminal.json',terminal)
    print(json.dumps(terminal,indent=2))


if __name__ == '__main__':
    main()
