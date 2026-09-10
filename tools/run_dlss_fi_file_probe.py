"""One serial existing-game full-file DLSSG 2x test; no H3 regeneration.

Default is CPU plan only. The initial whole-file qualification deliberately uses
the same fixed game source as the successful three-frame diagnostic. No browser,
installation, retry, automatic cut detection, NR/SR, or source replacement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss_fi_transport import BinarySession, runtime_identity  # noqa: E402
from dlss_fi_device_binding import cuda_device_inventory, bind_probe  # noqa: E402
from tools.dlss_fi_frame_stream import FrameStream  # noqa: E402
from tools.dlss_fi_media import inspect_source, decode_frames, encode_video, mux_and_validate  # noqa: E402
from run_progressive_pilot import RESEARCH, OwnedServer, ContinuousGuard, write_json  # noqa: E402
from progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402


def fixed_source(runtime):
    probe = RESEARCH/'dlss-fi-frame-probe-20260910-v2'
    terminal = json.loads((probe/'terminal.json').read_text(encoding='utf8'))
    recorded = json.loads((probe/'source.json').read_text(encoding='utf8'))
    result = json.loads((probe/'result.json').read_text(encoding='utf8'))
    if terminal['status'] != 'actual_two_intermediate_frames_noncopy_RGB_device_bound' or terminal['generated_frames'] != 2:
        raise ValueError('Actual first-frame qualification is missing')
    identities = runtime_identity(runtime)
    if recorded['runtime'] != identities:
        raise ValueError('Full-file candidate requires the same qualified runtime')
    for name, value in identities.items():
        if any(result['mapped'][name][key] != value[key] for key in ('path', 'bytes', 'sha256')):
            raise ValueError('Frame diagnostic did not bind the same runtime modules')
    identity = recorded['source_media']['file']
    if file_identity(identity['path']) != identity:
        raise ValueError('Fixed source differs from the real frame diagnostic')
    return identity, identities


def serialize_source(source):
    plan = asdict(source['plan'])
    plan.update(source_rate=str(plan['source_rate']), origin=str(plan['origin']), cuts=sorted(plan['cuts']))
    return {**source, 'plan': plan}


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan','gpu'), default='plan')
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    identity, identities = fixed_source(args.runtime)
    source = inspect_source(identity['path'])
    if source['file'] != identity or (source['width'],source['height'],source['plan'].source_count) != (1024,512,73):
        raise ValueError('Initial whole-file case is fixed at original 1024x512x73')
    if args.mode == 'plan':
        print(json.dumps({'mode':'CPU_plan_no_worker', 'source': serialize_source(source)}, ensure_ascii=False))
        return
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('A new dedicated research output directory is required')
    with SerialProbeLease(RESEARCH/'serial-gpu.lock'):
        root.mkdir(parents=True)
        started_at = time.perf_counter()
        owner, monitor, session = OwnedServer(root,0,False), None, None
        receipt = {'status':'incomplete', 'quality_qualified':False}
        try:
            write_json(root/'source.json', serialize_source(source))
            code_paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
                ('dlss_fi_transport.py','dlss_fi_device_binding.py','dlss_fi_frame_stream.py','dlss_fi_guides.py','dlss_fi_media.py')],
                Path(__file__).resolve().parents[1]/'dlss_fi_contract.py',Path(__file__).resolve().parents[1]/'dlss_nr_advanced.py']
            code = {str(path):file_identity(path) for path in code_paths}
            write_json(root/'identity.json', {'runtime':identities,'code':code})
            with NvmlResourceReader() as reader:
                guard = ResourceGuard()
                startup = reader.sample()
                guard.observe(startup,startup=True)
                write_json(root/'startup.json',startup)
                if guard.reason:
                    raise RuntimeError(guard.reason)
                inventory = cuda_device_inventory()
                write_json(root/'inventory.json',inventory)
                def check():
                    if (root/'STOP').exists():
                        raise RuntimeError('Explicit STOP requested')
                    if time.perf_counter()-started_at > 600:
                        raise TimeoutError('Whole-file test exceeded its ten-minute progress deadline')
                    if monitor:
                        monitor.check()
                def registered(process):
                    nonlocal monitor
                    owner.process = process
                    monitor = ContinuousGuard(reader,guard,root/'resources.jsonl',owner)
                    monitor.start()
                    print(f'Owned DLSSG whole-file worker PID {process.pid}',flush=True)
                try:
                    check()
                    session = BinarySession([identities['dlssg-worker.exe']['path'],'--serve'],cwd=args.runtime.resolve(),
                        width=source['width'],height=source['height'],frame_count=source['plan'].source_count,on_start=registered)
                    mapped_paths = set()
                    with (root/'frames.jsonl').open('x',encoding='utf8') as log:
                        def observed(row):
                            log.write(json.dumps(row)+'\n')
                            log.flush()
                            if row['slot'] in (0,72,144):
                                mapped_paths.update(m.path for m in psutil.Process(session.process.pid).memory_maps(grouped=True) if 'dlss' in m.path.lower())
                                print(f"Encoded sequence progress {row['slot']+1}/146",flush=True)
                        stream = FrameStream(source['plan'],width=source['width'],height=source['height'],session=session,observer=observed,check=check)
                        encoded = encode_video(root/'video-only.mp4',source,stream.outputs(decode_frames(source,check=check)),check=check)
                    if stream.report is None or stream.report['counts']['generated'] != 72 or session.next_index != 73:
                        raise ValueError('Whole-file runtime did not produce all 72 intermediate frames')
                    session.close()
                    logs = b''.join(session.logs).decode('utf8','replace')
                    write_json(root/'worker-log.json',{'stderr_tail':logs,'stop':session.stop_receipt})
                    binding = bind_probe({'supports_native_2x_by_report':True,'stderr_tail':logs},inventory,startup['gpu_uuid'])
                    mapped = {Path(path).name:file_identity(path) for path in mapped_paths}
                    for name, value in identities.items():
                        if name not in mapped or any(mapped[name][key] != value[key] for key in ('path','sha256','bytes')):
                            raise ValueError('Whole-file worker module identity mismatch')
                    check()
                    media = mux_and_validate(root/'video-only.mp4',source,root/'candidate.mp4',check=check)
                    if runtime_identity(args.runtime) != identities or any(file_identity(path) != value for path,value in code.items()):
                        raise ValueError('Runtime/code changed during the whole-file test')
                    write_json(root/'result.json',{'media':media,'ledger':stream.report,'encoder':encoded,'device_binding':binding,'mapped':mapped})
                    receipt.update(status='actual_73_source_72_generated_1_tail_audio_exact',candidate=media['file'],
                        audio_identity_qualified=True,generated_frames=72)
                finally:
                    if session:
                        session.close()
                    owner.stop()
                    if monitor:
                        monitor.close()
                        monitor.check()
                    receipt['resources'] = guard.report()
        except BaseException as error:
            receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if session:
                session.close()
                if not (root/'worker-log.json').exists():
                    write_json(root/'worker-log.json',{'stderr_tail':b''.join(session.logs).decode('utf8','replace'),'stop':session.stop_receipt})
            owner.stop()
            receipt.update(server_stop=owner.stop_receipt,wall_seconds=time.perf_counter()-started_at)
            write_json(root/'terminal.json',receipt)
            print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
