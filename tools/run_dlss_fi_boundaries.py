"""Run one predeclared synthetic FI boundary case, never an H3 generation.

Default is CPU plan. GPU cases are exclusive, serial and not retried. This is
research qualification, not a public arbitrary-file node or quality acceptance.
The original successful game file probe and its recorded sources are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_dlss_fi_file_probe import serialize_source, fixed_source  # noqa: E402
from tools.dlss_fi_transport import BinarySession, runtime_identity  # noqa: E402
from tools.dlss_fi_device_binding import cuda_device_inventory, bind_probe  # noqa: E402
from tools.dlss_fi_frame_stream import FrameStream  # noqa: E402
from tools.dlss_fi_media import inspect_source, decode_frames, encode_video, mux_and_validate  # noqa: E402
from tools.prepare_dlss_fi_boundaries import CASES, WIDTH, HEIGHT  # noqa: E402
from tools.run_progressive_pilot import RESEARCH, OwnedServer, ContinuousGuard, write_json  # noqa: E402
from tools.progressive_probe_control import SerialProbeLease, NvmlResourceReader, ResourceGuard, file_identity  # noqa: E402


def boundary_source(manifest_path, case):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if case not in CASES or manifest['status'] != 'CPU_synthetic_sources_only_no_FI_execution':
        raise ValueError('Expected a declared CPU boundary source')
    record = manifest['cases'][case]
    path = manifest_path.parent/(case+'.mp4')
    rate, count, cuts = CASES[case]
    if file_identity(path) != record['file']:
        raise ValueError('Boundary source changed or is outside its fixture directory')
    if (record['width'], record['height'], record['source_frames'], record['source_fps'],
            tuple(record['cuts']), record['has_audio']) != (WIDTH, HEIGHT, count, rate, cuts, False):
        raise ValueError('Boundary manifest changed the predeclared recipe')
    source = inspect_source(path, cuts=cuts)
    if (source['width'], source['height'], source['plan'].source_count, source['plan'].source_rate,
            tuple(sorted(source['plan'].cuts)), bool(source['audio_packets'])) != (WIDTH, HEIGHT, count, rate, cuts, False):
        raise ValueError('Decoded boundary source differs from its declaration')
    return source


def expected_counts(plan):
    return {'source': plan.source_count, 'generated': plan.generated_count,
            'cut_hold': len(plan.cuts), 'tail_hold': 1}


def run(args):
    import psutil
    source = boundary_source(args.manifest, args.case)
    _, identities = fixed_source(args.runtime)
    if args.mode == 'plan':
        return {'status': 'CPU_plan_no_worker', 'case': args.case, 'source': serialize_source(source),
                'expected_counts': expected_counts(source['plan'])}
    root = args.run_root.resolve()
    if root.exists() or root == RESEARCH or not root.is_relative_to(RESEARCH):
        raise ValueError('New exclusive research output directory required')
    with SerialProbeLease(RESEARCH/'serial-gpu.lock'):
        root.mkdir(parents=True)
        started = time.perf_counter()
        owner, monitor, session = OwnedServer(root, 0, False), None, None
        receipt = {'status': 'incomplete', 'case': args.case, 'quality_qualified': False,
                   'source_kind': 'CPU_synthetic_not_H3_generation'}
        try:
            write_json(root/'source.json', serialize_source(source))
            paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
                ('dlss_fi_transport.py', 'dlss_fi_device_binding.py', 'dlss_fi_frame_stream.py',
                 'dlss_fi_guides.py', 'dlss_fi_media.py', 'prepare_dlss_fi_boundaries.py', 'run_dlss_fi_file_probe.py')],
                Path(__file__).resolve().parents[1]/'dlss_fi_contract.py',
                Path(__file__).resolve().parents[1]/'dlss_nr_advanced.py']
            code = {str(p): file_identity(p) for p in paths}
            write_json(root/'identity.json', {'runtime': identities, 'code': code,
                                              'manifest': file_identity(args.manifest)})
            with NvmlResourceReader() as reader:
                guard = ResourceGuard()
                startup = reader.sample()
                guard.observe(startup, startup=True)
                write_json(root/'startup.json', startup)
                if guard.reason:
                    raise RuntimeError(guard.reason)
                inventory = cuda_device_inventory()
                write_json(root/'inventory.json', inventory)

                def check():
                    if (root/'STOP').exists():
                        raise RuntimeError('Explicit STOP requested')
                    if time.perf_counter()-started > 180:
                        raise TimeoutError('Boundary progress deadline exceeded')
                    if monitor:
                        monitor.check()

                def registered(process):
                    nonlocal monitor
                    owner.process = process
                    monitor = ContinuousGuard(reader, guard, root/'resources.jsonl', owner)
                    monitor.start()
                    print(f'Owned boundary worker PID {process.pid}', flush=True)

                try:
                    check()
                    session = BinarySession([identities['dlssg-worker.exe']['path'], '--serve'],
                        cwd=args.runtime.resolve(), width=source['width'], height=source['height'],
                        frame_count=source['plan'].source_count, on_start=registered)
                    mapped_paths = set()
                    with (root/'frames.jsonl').open('x', encoding='utf8') as log:
                        def observed(row):
                            log.write(json.dumps(row)+'\n')
                            log.flush()
                            if row['slot'] in (0, source['plan'].output_count-2):
                                mapped_paths.update(m.path for m in psutil.Process(session.process.pid).memory_maps(grouped=True)
                                                    if 'dlss' in m.path.lower())
                        stream = FrameStream(source['plan'], width=source['width'], height=source['height'],
                                             session=session, observer=observed, check=check)
                        encoded = encode_video(root/'video-only.mp4', source,
                                               stream.outputs(decode_frames(source, check=check)), check=check)
                    if stream.report is None or stream.report['counts'] != expected_counts(source['plan']) or session.next_index != source['plan'].source_count:
                        raise ValueError('Actual frame counts differ from fixed boundary case')
                    session.close()
                    logs = b''.join(session.logs).decode('utf8', 'replace')
                    write_json(root/'worker-log.json', {'stderr_tail': logs, 'stop': session.stop_receipt})
                    binding = bind_probe({'supports_native_2x_by_report': True, 'stderr_tail': logs}, inventory, startup['gpu_uuid'])
                    mapped = {Path(p).name: file_identity(p) for p in mapped_paths}
                    for name, identity in identities.items():
                        if name not in mapped or any(mapped[name][k] != identity[k] for k in ('path', 'sha256', 'bytes')):
                            raise ValueError('Actual mapped runtime differs')
                    media = mux_and_validate(root/'video-only.mp4', source, root/'candidate.mp4', check=check)
                    if runtime_identity(args.runtime) != identities or any(file_identity(p) != v for p, v in code.items()):
                        raise ValueError('Runtime or source code changed during execution')
                    write_json(root/'result.json', {'media': media, 'ledger': stream.report, 'encoder': encoded,
                                                   'device_binding': binding, 'mapped': mapped})
                    receipt.update(status='actual_boundary_complete', candidate=media['file'],
                                   counts=stream.report['counts'], no_audio_source=True)
                finally:
                    if session:
                        session.close()
                    owner.stop()
                    if monitor:
                        monitor.close()
                        monitor.check()
                    receipt['resources'] = guard.report()
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if session:
                session.close()
                if not (root/'worker-log.json').exists():
                    write_json(root/'worker-log.json', {'stderr_tail': b''.join(session.logs).decode('utf8', 'replace'),
                                                       'stop': session.stop_receipt})
            owner.stop()
            receipt.update(server_stop=owner.stop_receipt, wall_seconds=time.perf_counter()-started)
            write_json(root/'terminal.json', receipt)
            print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'gpu'), default='plan')
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
